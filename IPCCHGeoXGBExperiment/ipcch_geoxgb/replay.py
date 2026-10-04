"""P5 independent replay of a completed run from its saved keyed evidence.

Deliberately does NOT call the training-path code that produced the evidence
(projection, metrics, stage1/stage3 gate functions): projection, counting,
gate dates/decisions and selection are re-implemented here in plain form, so
an error in the production path cannot certify itself. Every check failure is
collected; a run passes only with zero failures. No model is fitted.

Detects, among others: a dropped/extra/duplicated prediction row, truth not
matching the prepared keys, a wrong target column order, an origin or
persistence date after O (future information), a local model whose prefix is
not the same-origin global, a gate decision that does not follow from its
saved pairs (stale gate), routing that disagrees with the frozen map, a Stage1
winner not reproducible from saved S predictions, and report metrics or
bootstrap intervals not reproducible from saved predictions/draws.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_geoxgb.artifacts import sha256_file
from ipcch_geoxgb.learnmap import verify_prepared
from ipcch_geoxgb.modelstore import array_digest

Q = ("q2", "q3", "q4", "q5")


class Checks:
    def __init__(self):
        self.failures: list[str] = []
        self.passed: dict[str, int] = {}

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        if ok:
            self.passed[name] = self.passed.get(name, 0) + 1
        else:
            self.failures.append(f"{name}: {detail}")


def _pava(values) -> list[float]:
    """Plain pool-adjacent-violators (sum/count blocks), non-increasing."""
    blocks = []
    for v in values:
        blocks.append([float(v), 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
    out = []
    for s, n in blocks:
        out += [s / n] * n
    return out


def _project(raw: np.ndarray) -> np.ndarray:
    return np.clip(np.array([_pava(r) for r in raw]), 0.0, 1.0)


def _phase(star: np.ndarray) -> np.ndarray:
    return 1 + (star >= 0.20).sum(axis=1)


def _counts(truth, pred) -> tuple[int, int, int, int]:
    t, p = np.asarray(truth) >= 3, np.asarray(pred) >= 3
    return int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum()), int((~t & ~p).sum())


def _f1(tp, fp, fn) -> Fraction | None:
    return Fraction(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) else None


def _month(label: str) -> int:
    y, m = label.split("-")
    return int(y) * 12 + int(m) - 1


def replay_run(run_dir: Path, contract: dict) -> dict:
    c = Checks()
    prepared, s1, s3 = run_dir / "prepared", run_dir / "stage1", run_dir / "stage3"
    summary3 = json.loads((s3 / "stage3-summary.json").read_text(encoding="utf-8"))
    val_floor = contract["support"]["validation"]
    min_dates = contract["support"]["stage3_min_successful_local_dates"]
    max_dates = contract["calendar"]["historical_gate_max_dates"]
    records = _model_records(run_dir / "models")
    keys_by_h, regions_by_h = {}, {}
    try:
        verify_prepared(prepared, [int(h) for h in summary3["horizons"]])
        c.check("prepared.inventory_and_digests", True)
    except Exception as error:  # recorded as a failed check, replay continues
        c.check("prepared.inventory_and_digests", False, str(error))

    for h_str, info in summary3["horizons"].items():
        h = int(h_str)
        keys = pd.read_csv(prepared / f"keys_h{h:02d}.csv.gz")
        pred = pd.read_csv(s3 / f"h{h:02d}" / "predictions.csv.gz")
        ledger = pd.read_csv(s3 / f"h{h:02d}" / "fold_ledger.csv")
        frozen = json.loads((s1 / f"frozen_h{h:02d}.json").read_text(encoding="utf-8"))
        fmap = pd.read_csv(s1 / f"frozen_map_h{h:02d}.csv", dtype={"node_id": str})
        region_of = dict(zip(fmap["admin_code"].astype(int), fmap["node_id"]))
        keys_by_h[h], regions_by_h[h] = keys, fmap
        c.check(f"H{h}.predictions_digest", sha256_file(s3 / f"h{h:02d}" / "predictions.csv.gz") == info["predictions_sha256"])
        c.check(f"H{h}.frozen_map_digest", sha256_file(s1 / f"frozen_map_h{h:02d}.csv") == frozen["map_sha256"])

        # A. keys: every scored fold has exactly the prepared valid keys at T; truth equals prepared
        key_cols = ["admin_code", "target_ord"]
        c.check(f"H{h}.no_duplicate_keys", not pred.duplicated(key_cols).any(), "duplicated prediction key")
        for row in ledger.to_dict("records"):
            expected = keys[keys["target_ord"] == row["target_ord"]]
            got = pred[(pred["target_ord"] == row["target_ord"]) & (pred["fold_id"] == row["fold_id"])]
            c.check(f"H{h}.fold_keys", row["eval_keys"] == len(expected) and
                    set(map(tuple, got[key_cols].to_numpy())) == set(map(tuple, expected[key_cols].to_numpy())),
                    f"{row['fold_id']}: {len(got)} predictions vs {len(expected)} prepared keys")
        merged = pred.merge(keys[key_cols + ["phase_truth", "q3"]], on=key_cols, how="left", suffixes=("", "_prep"))
        c.check(f"H{h}.truth_matches_prepared", merged["phase_truth"].equals(merged["phase_truth_prep"]) and
                np.array_equal(merged["q3_truth"].to_numpy(), merged["q3"].to_numpy()), "truth differs from prepared")

        # B. information boundary
        c.check(f"H{h}.origin_is_T_minus_H", (pred["origin_ord"] == pred["target_ord"] - h).all())
        avail = pred[pred["persistence_available"] == 1]
        src = np.array([_month(m) for m in avail["persistence_source_month"]]) if len(avail) else np.zeros(0)
        c.check(f"H{h}.persistence_not_after_origin", bool(np.all(src <= avail["origin_ord"].to_numpy())) and
                bool(np.all(avail["persistence_age_months"].to_numpy() == avail["origin_ord"].to_numpy() - src)))

        # C. projection/decoding from saved raw quartets
        for arm in ("geo", "pool"):
            raw = pred[[f"{arm}_{q}_raw" for q in Q]].to_numpy()
            star = pred[[f"{arm}_{q}_star" for q in Q]].to_numpy()
            again = _project(raw)
            c.check(f"H{h}.{arm}_projection", np.allclose(again, star, atol=1e-12, rtol=0), "q_star mismatch")
            c.check(f"H{h}.{arm}_decode", np.array_equal(_phase(star), pred[f"{arm}_phase"].to_numpy()), "phase mismatch")
            c.check(f"H{h}.{arm}_monotone", bool(np.all(np.diff(star, axis=1) <= 0)), "q_star not ordered q2..q5")

        # D. routing
        expected_region = pred["admin_code"].map(lambda a: region_of.get(int(a), "")).fillna("")
        c.check(f"H{h}.region_from_frozen_map", (pred["region"].fillna("") == expected_region).all())
        local = pred["route"] == "local"
        same = np.all([np.array_equal(pred.loc[~local, f"geo_{q}_raw"], pred.loc[~local, f"pool_{q}_raw"]) for q in Q])
        c.check(f"H{h}.non_local_equals_pooled", bool(same))
        c.check(f"H{h}.local_provider_differs", (pred.loc[local, "provider"] != pred.loc[local, "global_identity"]).all())
        c.check(f"H{h}.unmapped_routes_global", (pred.loc[expected_region == "", "route"] == "unmapped_area_global").all())

        # E. gate replay
        observed = np.unique(keys["target_ord"].to_numpy())
        decisions = [json.loads(line) for line in (s3 / f"h{h:02d}" / "gate_decisions.jsonl").read_text().splitlines()]
        by_fold = {}
        for d in decisions:
            by_fold.setdefault(d["fold_id"], []).append(d)
        for row in ledger[ledger["status"] == "scored"].to_dict("records"):
            origin = int(row["origin_ord"])
            dates = sorted(observed[observed < origin].tolist(), reverse=True)[:max_dates]
            c.check(f"H{h}.gate_dates", json.loads(row["gate_dates"].replace("'", '"')) == dates if isinstance(row["gate_dates"], str)
                    else list(row["gate_dates"]) == dates, f"{row['fold_id']}")
            if not frozen["accepted_split"]:
                c.check(f"H{h}.no_gate_without_split", row["fold_id"] not in by_fold)
                continue
            ppath = s3 / f"h{h:02d}" / f"pairs_{row['fold_id']}.csv.gz"
            pairs = pd.read_csv(ppath) if ppath.is_file() else pd.DataFrame()
            if len(pairs):
                c.check(f"H{h}.pair_dates", set(pairs["validation_month"]) <= set(dates) and
                        (pairs["internal_origin"] == pairs["validation_month"] - h).all())
            fold_pred = pred[pred["fold_id"] == row["fold_id"]]
            for d in by_fold.get(row["fold_id"], []):
                part = pairs[pairs["region"] == d["region"]] if len(pairs) else pairs
                enabled = _gate(part, val_floor, min_dates)
                c.check(f"H{h}.gate_decision", enabled == d["enabled"], f"{row['fold_id']}/{d['region']}")
                region_rows = fold_pred[fold_pred["region"] == d["region"]]
                if d.get("route") == "local":
                    c.check(f"H{h}.local_route_matches_gate", (region_rows["route"] == "local").all() and
                            (region_rows["provider"] == d["local_identity"]).all())
                    _check_local_model(c, h, records, d["local_identity"], origin)
                else:
                    c.check(f"H{h}.no_local_without_gate", not (region_rows["route"] == "local").any())

        # G. Stage1 selection reproducible from saved S predictions
        sel = json.loads((s1 / f"h{h:02d}" / "selection.json").read_text(encoding="utf-8"))
        entries = []
        for e in sel["candidates"]:
            sp = pd.read_csv(s1 / f"h{h:02d}" / e["candidate"] / "s_predictions.csv.gz")
            star = sp[[f"{q}_star" for q in Q]].to_numpy()
            c.check(f"H{h}.stage1_projection", np.allclose(_project(sp[[f"{q}_raw" for q in Q]].to_numpy()), star, atol=1e-12, rtol=0))
            tp, fp, fn, _ = _counts(sp["phase_truth"], _phase(star))
            f1 = _f1(tp, fp, fn)
            c.check(f"H{h}.stage1_f1", (str(f1) if f1 is not None else None) == e["f1_exact"], e["candidate"])
            if f1 is not None:
                entries.append((-f1, e["terminal_regions"], e["global_rounds"] + e["local_rounds"],
                                e["global_depth"], e["local_depth"], e["g_id"], e["l_id"], e["candidate"]))
        c.check(f"H{h}.stage1_winner", bool(entries) and sorted(entries)[0][-1] == frozen["candidate"])

    # F. every Stage3 model request's window/origin/prefix provenance
    for line in (s3 / "model_requests.jsonl").read_text().splitlines():
        use = json.loads(line)
        ident = records[use["identity_sha256"]]["identity"]
        c.check("models.window_ends_at_origin", ident["window"][1] == ident["fitting_origin"] == use["fitting_origin"])
        if use.get("use") == "gate":
            c.check("models.gate_origin_is_U_minus_H", use["fitting_origin"] == use["gate_month"] - use["H"])
        if use["purpose"] == "local":
            _check_local_model(c, use["H"], records, use["identity_sha256"], use["fitting_origin"])
        _check_model_targets(c, records[use["identity_sha256"]], keys_by_h[use["H"]], regions_by_h[use["H"]])

    # H. report recomputation
    report_path = run_dir / "report" / "report.json"
    if report_path.is_file():
        _check_report(c, run_dir, json.loads(report_path.read_text(encoding="utf-8")))
    return {"failures": c.failures, "passed_checks": c.passed, "status": "passed" if not c.failures else "failed"}


def _model_records(root: Path) -> dict:
    out = {}
    for path in root.glob("*/*/record.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        out[record["identity_sha256"]] = record
    return out


def _check_local_model(c: Checks, h: int, records: dict, digest: str, origin: int) -> None:
    rec = records.get(digest)
    c.check("models.local_exists", rec is not None, digest)
    if rec is None:
        return
    ident = rec["identity"]
    glob = records.get(ident["global_identity"])
    c.check("models.local_has_global", glob is not None and glob["identity"]["scope"].endswith("-global"))
    if glob is None:
        return
    c.check("models.local_same_origin_as_global", glob["identity"]["fitting_origin"] == ident["fitting_origin"] == origin)
    for q in Q:
        fit = rec["fit_records"][q]
        c.check("models.local_prefix_is_global", fit["kind"] == "local" and
                fit["parent_booster_sha256"] == glob["booster_sha256"][q] and
                fit["parent_rounds"] == glob["fit_records"][q]["rounds_total"] and
                fit["parent_structure_sha256"] == glob["fit_records"][q]["structure_sha256"], f"{digest}/{q}")


def _check_model_targets(c: Checks, rec: dict, keys: pd.DataFrame, fmap: pd.DataFrame) -> None:
    """Rebuild the fitting rows from prepared keys; each target's y digest must match."""
    ident = rec["identity"]
    lo, hi = ident["window"]
    months = keys["target_ord"].to_numpy()
    rows = np.flatnonzero((months >= lo) & (months <= hi))
    if ident["scope"] == "stage3-local":
        areas = fmap.loc[fmap["node_id"] == ident["region_node"], "admin_code"].to_numpy(dtype=np.int64)
        c.check("models.region_members", array_digest(np.sort(areas)) == ident["region_areas"], ident["region_node"])
        rows = rows[np.isin(keys["admin_code"].to_numpy()[rows], areas)]
    c.check("models.fit_rows_rebuilt", array_digest(rows.astype(np.int64)) == ident["fit_rows"], rec["identity_sha256"])
    for q in Q:
        y = np.ascontiguousarray(keys[q].to_numpy(dtype=np.float64)[rows])
        c.check("models.target_order", hashlib.sha256(y.tobytes()).hexdigest() == rec["fit_records"][q]["y_sha256"],
                f"{rec['identity_sha256']}/{q}")


def _gate(pairs: pd.DataFrame, floor: dict, min_dates: int) -> bool:
    n = len(pairs)
    if n == 0:
        return False
    truth = pairs["phase_truth"].to_numpy()
    crisis = int((truth >= 3).sum())
    months = pairs["validation_month"].nunique()
    ok_dates = pairs[pairs["local_fit_ok"].astype(bool)]["validation_month"].nunique()
    support = {"keys": n, "areas": pairs["admin_code"].nunique(), "target_months": months,
               "crisis_keys": crisis, "noncrisis_keys": n - crisis}
    if any(support[k] < v for k, v in floor.items()) or ok_dates < min_dates:
        return False
    tg, fg, ng, _ = _counts(truth, pairs["phase_global"])
    tl, fl, nl, _ = _counts(truth, pairs["phase_local_routed"])
    f_g, f_l = _f1(tg, fg, ng), _f1(tl, fl, nl)
    return f_g is not None and f_l is not None and (f_l - f_g) > Fraction(1, 100)


def _check_report(c: Checks, run_dir: Path, report: dict) -> None:
    for h, hrep in report["horizons"].items():
        pred = pd.read_csv(run_dir / "stage3" / f"h{int(h):02d}" / "predictions.csv.gz")
        for period in ("main", "supplementary"):
            entry = hrep[period]
            e_all = pred[pred["period"] == period]
            e_persist = e_all[e_all["persistence_available"] == 1]
            c.check("report.coverage", entry["coverage"]["E_all_keys"] == len(e_all) and
                    entry["coverage"]["E_persist_keys"] == len(e_persist), f"H{h}/{period}")
            for cohort, frame, arms in (("E_all", e_all, {"geo": "geo_phase", "pool": "pool_phase"}),
                                        ("E_persist", e_persist, {"geo": "geo_phase", "persistence": "persistence_phase"})):
                if cohort not in entry:
                    continue
                for arm, col in arms.items():
                    tp, fp, fn, tn = _counts(frame["phase_truth"], frame[col])
                    saved = entry[cohort][arm]["binary"]["counts"]
                    c.check("report.counts", saved == {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, f"H{h}/{period}/{cohort}/{arm}")
            for name, rec in entry.get("bootstrap", {}).items():
                path = run_dir / "report" / f"bootstrap_h{int(h):02d}_{name}.csv.gz"
                if not path.is_file():
                    continue
                draws = pd.read_csv(path)
                mult = draws[[f"m::{x}" for x in rec["countries"]]].to_numpy()
                arms = list(rec["country_counts"])
                ca, cb = (np.array(rec["country_counts"][a]) for a in arms)
                fa = _vector_f1(mult @ ca)
                fb = _vector_f1(mult @ cb)
                c.check("report.bootstrap_deltas", np.allclose(fa - fb, draws["delta"].to_numpy(), equal_nan=True), name)
                expected = rng_multiplicities(len(rec["countries"]), rec["draws"], rec["seed"])
                c.check("report.bootstrap_rng", np.array_equal(expected, mult), name)
                if rec["interval"] is not None:
                    lo, hi = np.percentile(fa - fb, rec["percentiles"], method="linear")
                    c.check("report.bootstrap_interval", np.allclose([lo, hi], rec["interval"]), name)


def _vector_f1(counts: np.ndarray) -> np.ndarray:
    den = 2 * counts[:, 0] + counts[:, 1] + counts[:, 2]
    return np.where(den > 0, 2 * counts[:, 0] / np.where(den > 0, den, 1), np.nan)


def rng_multiplicities(k: int, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([np.bincount(rng.integers(0, k, size=k), minlength=k) for _ in range(draws)])
