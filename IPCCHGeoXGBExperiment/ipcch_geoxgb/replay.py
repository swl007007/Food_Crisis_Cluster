"""P5 independent replay of a completed run from its saved keyed evidence.

Authority is the frozen contract, the verified prepared data and the actual
saved boosters -- never a stage's own summary or report. Scientific
recomputation (projection, decoding, crisis/four-class/R² metrics and NA
rules, support, gates, Stage1 selection, bootstrap) is re-implemented here in
plain form and does NOT call the production functions that produced the
evidence. Artifact loading/integrity reuses the store's primitives
(byte digests and ``validate_fit_records``), as R48 defines what a valid model
artifact is. Predictions are re-derived from the referenced fitted quartets on
the prepared inputs; nothing is refitted.

Every check failure is collected; missing required evidence is a failure, not
a skipped check. A run passes only with zero failures.
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
from ipcch_geoxgb.modelstore import array_digest, target_digests, validate_fit_records
from ipcch_geoxgb.quartet import TARGETS, Quartet

Q = TARGETS
DRAWS, SEED = 2000, 42
# Saved prediction CSVs are read with float_precision="round_trip": pandas' default
# parser is not round-trip exact, and lineage is checked by exact equality.


class Checks:
    def __init__(self):
        self.failures: list[str] = []
        self.passed: dict[str, int] = {}

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passed[name] = self.passed.get(name, 0) + 1
        else:
            self.failures.append(f"{name}: {detail}")
        return bool(ok)

    def guard(self, name: str, fn, *args):
        """Run a check group; an exception (e.g. missing evidence) is a failure."""
        try:
            return fn(*args)
        except Exception as error:  # noqa: BLE001 - replay records, never crashes
            self.failures.append(f"{name}: {type(error).__name__}: {error}")
            return None


# ------------------------------------------------------------- independent science


def _pava(values) -> list[float]:
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
    return np.clip(np.array([_pava(r) for r in raw]).reshape(-1, 4), 0.0, 1.0)


def _phase(star: np.ndarray) -> np.ndarray:
    return 1 + (np.asarray(star) >= 0.20).sum(axis=1)


def _counts(truth, pred) -> dict:
    t, p = np.asarray(truth) >= 3, np.asarray(pred) >= 3
    return {"tp": int((t & p).sum()), "fp": int((~t & p).sum()), "fn": int((t & ~p).sum()), "tn": int((~t & ~p).sum())}


def _f1(c) -> Fraction | None:
    den = 2 * c["tp"] + c["fp"] + c["fn"]
    return Fraction(2 * c["tp"], den) if den else None


def _ratio(num, den):
    return num / den if den else None


def _r2(y, f):
    y, f = np.asarray(y, dtype=float), np.asarray(f, dtype=float)
    if len(y) < 2 or np.all(y == y[0]):
        return None
    with np.errstate(over="ignore", under="ignore"):
        sst, sse = float(((y - y.mean()) ** 2).sum()), float(((y - f) ** 2).sum())
    if not (np.isfinite(sst) and np.isfinite(sse)) or sst == 0.0:
        return None
    return 1.0 - sse / sst


def _panel(truth, pred, q3_true, q3_star, q3_raw) -> dict:
    c = _counts(truth, pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    t4 = np.minimum(np.asarray(truth), 4) - 1
    p4 = np.minimum(np.asarray(pred), 4) - 1
    f1s = []
    for k in range(4):
        tpk = int(((t4 == k) & (p4 == k)).sum())
        den = 2 * tpk + int(((t4 != k) & (p4 == k)).sum()) + int(((t4 == k) & (p4 != k)).sum())
        f1s.append(2 * tpk / den if den else None)
    n = len(t4)
    return {
        "binary.accuracy": _ratio(tp + tn, n), "binary.precision": _ratio(tp, tp + fp),
        "binary.recall": _ratio(tp, tp + fn), "binary.f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "binary.f2": _ratio(5 * tp, 5 * tp + 4 * fn + fp),
        "four_class.accuracy": _ratio(int((t4 == p4).sum()), n),
        "four_class.macro_f1": None if any(v is None for v in f1s) else float(np.mean(f1s)),
        "q3_r2_projected": _r2(q3_true, q3_star), "q3_r2_raw": _r2(q3_true, q3_raw), "counts": c,
    }


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.isclose(float(a), float(b), rtol=1e-12, atol=1e-12))


def _arm_cols(arm):
    if arm == "persistence":
        return "persistence_phase", "persistence_q3", "persistence_q3"
    return f"{arm}_phase", f"{arm}_q3_star", f"{arm}_q3_raw"


def _arm_panel(frame, arm):
    ph, qs, qr = _arm_cols(arm)
    return _panel(frame["phase_truth"].to_numpy(), frame[ph].to_numpy(), frame["q3_truth"].to_numpy(),
                  frame[qs].to_numpy(), frame[qr].to_numpy())


def _rng_multiplicities(k: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return np.array([np.bincount(rng.integers(0, k, size=k), minlength=k) for _ in range(DRAWS)])


def _support(frame) -> dict:
    crisis = int((frame["phase_truth"] >= 3).sum())
    return {"keys": len(frame), "areas": frame["admin_code"].nunique(), "target_months": frame["target_ord"].nunique(),
            "crisis_keys": crisis, "noncrisis_keys": len(frame) - crisis}


def _meets(support, floor) -> bool:
    return all(support[k] >= v for k, v in floor.items())


# ------------------------------------------------------------- artifact access


class Models:
    """Loads saved quartets once, with byte and fit-record validation (R48 primitives)."""

    def __init__(self, root: Path):
        self.root = root
        self.records = {}
        for path in root.glob("*/*/record.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            self.records[record["identity_sha256"]] = record
        self._loaded: dict[str, Quartet] = {}

    def quartet(self, digest: str) -> Quartet:
        if digest not in self._loaded:
            record = self.records.get(digest)
            if record is None:
                raise FileNotFoundError(f"model {digest} has no saved record")
            directory = self.root / digest[:2] / digest
            payloads = {}
            for q in Q:
                payload = (directory / f"{q}.ubj").read_bytes()
                if hashlib.sha256(payload).hexdigest() != record["booster_sha256"][q]:
                    raise ValueError(f"model {digest} {q} bytes differ from the record")
                payloads[q] = payload
            validate_fit_records(record["identity"], payloads, record["fit_records"])
            self._loaded[digest] = Quartet(payloads, record["fit_records"])
        return self._loaded[digest]

    def identity(self, digest: str) -> dict:
        record = self.records.get(digest)
        if record is None:
            raise FileNotFoundError(f"model {digest} has no saved record")
        return record["identity"]


# ------------------------------------------------------------- replay


def replay_run(run_dir: Path, contract: dict) -> dict:
    c = Checks()
    run_dir = Path(run_dir)
    horizons = [int(h) for h in contract["calendar"]["horizons_months"]]
    prepared = run_dir / "prepared"
    bound = c.guard("prepared.inventory_and_digests", verify_prepared, prepared, horizons)
    if bound is None:
        return _result(c)
    artifacts = bound["manifest"]["artifacts_sha256"]
    calendar = pd.read_csv(prepared / "fold_calendar.csv")
    split = pd.read_csv(prepared / "stage1_split.csv.gz")
    models = Models(run_dir / "models")
    c.check("inventory.stage1_summary", (run_dir / "stage1" / "stage1-summary.json").is_file())
    c.check("inventory.stage3_summary", (run_dir / "stage3" / "stage3-summary.json").is_file())
    c.check("inventory.report", (run_dir / "report" / "report.json").is_file())
    report = c.guard("inventory.report_readable", lambda: json.loads((run_dir / "report" / "report.json").read_text()))
    summary3 = c.guard("inventory.stage3_readable",
                       lambda: json.loads((run_dir / "stage3" / "stage3-summary.json").read_text()))
    if summary3 is not None:
        c.check("inventory.stage3_horizons", sorted(int(h) for h in summary3["horizons"]) == sorted(horizons))
    for h in horizons:
        keys = pd.read_csv(prepared / f"keys_h{h:02d}.csv.gz")
        X = np.load(prepared / f"X_rich561_h{h:02d}.npy", mmap_mode="r")
        sha = {"X": artifacts[f"X_rich561_h{h:02d}.npy"], "keys": artifacts[f"keys_h{h:02d}.csv.gz"]}
        frozen = c.guard(f"H{h}.stage1", _replay_stage1, c, run_dir, h, keys, X, split, contract, models)
        if frozen is None:
            continue
        pred = c.guard(f"H{h}.stage3", _replay_stage3, c, run_dir, h, keys, X, sha, calendar, frozen, contract, models)
        if pred is not None and report is not None:
            c.guard(f"H{h}.report", _replay_report, c, run_dir, h, pred, calendar, frozen, report)
    c.guard("models.requests", _replay_requests, c, run_dir, models, horizons, artifacts)
    return _result(c)


def _result(c: Checks) -> dict:
    return {"failures": c.failures, "passed_checks": c.passed, "status": "passed" if not c.failures else "failed"}


def _replay_stage1(c, run_dir, h, keys, X, split, contract, models):
    s1 = run_dir / "stage1"
    frozen = json.loads((s1 / f"frozen_h{h:02d}.json").read_text(encoding="utf-8"))
    fmap = pd.read_csv(s1 / f"frozen_map_h{h:02d}.csv", dtype={"node_id": str})
    c.check(f"H{h}.frozen_binding", frozen.get("H") == h and sha256_file(s1 / f"frozen_map_h{h:02d}.csv") ==
            frozen["map_sha256"] and not fmap["admin_code"].duplicated().any())
    selection = json.loads((s1 / f"h{h:02d}" / "selection.json").read_text(encoding="utf-8"))
    expected = {g + l for g in contract["model"]["global_recipes"] for l in contract["model"]["local_recipes"]}
    got = {e["candidate"] for e in selection["candidates"]}
    c.check(f"H{h}.stage1_candidate_inventory", got == expected, f"{sorted(got)} vs {sorted(expected)}")
    s_keys = split[split["split_role"] == "validation"][["admin_code", "month_ord"]]
    row_of = {k: i for i, k in enumerate(map(tuple, keys[["admin_code", "target_ord"]].to_numpy()))}
    expected_s = set(map(tuple, s_keys.to_numpy()))
    ranking = []
    for e in selection["candidates"]:
        cdir = s1 / f"h{h:02d}" / e["candidate"]
        sp = pd.read_csv(cdir / "s_predictions.csv.gz", float_precision="round_trip")
        c.check(f"H{h}.stage1_complete_S", set(map(tuple, sp[["admin_code", "target_ord"]].to_numpy())) == expected_s
                and len(sp) == len(expected_s), e["candidate"])
        raw = sp[[f"{q}_raw" for q in Q]].to_numpy()
        for provider, part in sp.groupby("provider"):
            rows = np.array([row_of[k] for k in map(tuple, part[["admin_code", "target_ord"]].to_numpy())])
            again = models.quartet(provider).predict_raw(np.asarray(X[rows]))
            c.check(f"H{h}.stage1_provider_lineage", np.array_equal(again, part[[f"{q}_raw" for q in Q]].to_numpy()),
                    f"{e['candidate']}/{provider[:12]}")
        star = _project(raw)
        c.check(f"H{h}.stage1_projection", np.allclose(star, sp[[f"{q}_star" for q in Q]].to_numpy(), atol=1e-12, rtol=0)
                and np.array_equal(_phase(star), sp["phase_pred"].to_numpy()), e["candidate"])
        f1 = _f1(_counts(sp["phase_truth"], _phase(star)))
        c.check(f"H{h}.stage1_f1", (None if f1 is None else str(f1)) == e["f1_exact"], e["candidate"])
        if f1 is not None:
            ranking.append((-f1, e["terminal_regions"], e["global_rounds"] + e["local_rounds"],
                            e["global_depth"], e["local_depth"], e["g_id"], e["l_id"], e["candidate"]))
    c.check(f"H{h}.stage1_winner", bool(ranking) and sorted(ranking)[0][-1] == frozen["candidate"] ==
            selection["selection"]["winner"])
    winner_map = pd.read_csv(s1 / f"h{h:02d}" / frozen["candidate"] / "terminal_map.csv", dtype={"node_id": str})
    c.check(f"H{h}.frozen_is_winner_map", winner_map[["admin_code", "node_id"]].sort_values("admin_code")
            .reset_index(drop=True).equals(fmap[["admin_code", "node_id"]].reset_index(drop=True)))
    return {"record": frozen, "map": fmap}


def _replay_stage3(c, run_dir, h, keys, X, sha, calendar, frozen, contract, models):
    s3 = run_dir / "stage3" / f"h{h:02d}"
    pred = pd.read_csv(s3 / "predictions.csv.gz", float_precision="round_trip")
    ledger = pd.read_csv(s3 / "fold_ledger.csv")
    record, fmap = frozen["record"], frozen["map"]
    region_of = dict(zip(fmap["admin_code"].astype(int), fmap["node_id"]))
    regions = {n: set(g["admin_code"].astype(int)) for n, g in fmap.groupby("node_id")}
    months = keys["target_ord"].to_numpy()
    area = keys["admin_code"].to_numpy()
    row_of = {k: i for i, k in enumerate(map(tuple, keys[["admin_code", "target_ord"]].to_numpy()))}
    cal = calendar[calendar["horizon_months"] == h]
    c.check(f"H{h}.schedule_complete", sorted(cal["fold_id"]) == sorted(ledger["fold_id"]),
            f"{len(ledger)} ledger folds vs {len(cal)} scheduled")
    c.check(f"H{h}.predictions_only_scheduled", set(pred["fold_id"]) <= set(cal["fold_id"]))
    c.check(f"H{h}.no_duplicate_keys", not pred.duplicated(["admin_code", "target_ord"]).any())
    observed = np.unique(months)
    floors_fit = contract["support"]["local_fit"]
    floors_val = contract["support"]["validation"]
    min_dates = contract["support"]["stage3_min_successful_local_dates"]
    window = contract["calendar"]["rolling_window_calendar_months"]
    max_dates = contract["calendar"]["historical_gate_max_dates"]
    gate_path = s3 / "gate_decisions.jsonl"
    decisions = [json.loads(x) for x in gate_path.read_text(encoding="utf-8").splitlines()] if gate_path.is_file() else []
    by_fold: dict = {}
    for d in decisions:
        by_fold.setdefault(d["fold_id"], []).append(d)

    for fold in cal.to_dict("records"):
        fid, target, origin = fold["fold_id"], int(fold["target_ord"]), int(fold["origin_ord"])
        lrow = ledger[ledger["fold_id"] == fid]
        expected_rows = np.flatnonzero(months == target)
        got = pred[pred["fold_id"] == fid]
        if len(expected_rows) == 0:
            c.check(f"H{h}.empty_fold_semantics", len(lrow) == 1 and lrow["status"].iloc[0] == "no_valid_target"
                    and len(got) == 0 and fid not in by_fold, fid)
            continue
        c.check(f"H{h}.fold_scored", len(lrow) == 1 and lrow["status"].iloc[0] == "scored"
                and int(lrow["eval_keys"].iloc[0]) == len(expected_rows), fid)
        if not c.check(f"H{h}.fold_keys", set(map(tuple, got[["admin_code", "target_ord"]].to_numpy()))
                       == set(map(tuple, keys.iloc[expected_rows][["admin_code", "target_ord"]].to_numpy()))
                       and len(got) == len(expected_rows), f"{fid}: {len(got)} vs {len(expected_rows)}"):
            continue
        rows = np.array([row_of[k] for k in map(tuple, got[["admin_code", "target_ord"]].to_numpy())])
        k = keys.iloc[rows]
        c.check(f"H{h}.truth_matches_prepared", np.array_equal(got["phase_truth"].to_numpy(), k["phase_truth"].to_numpy())
                and np.array_equal(got["q3_truth"].to_numpy(), k["q3"].to_numpy()), fid)
        c.check(f"H{h}.origin_is_T_minus_H", bool((got["origin_ord"] == target - h).all()) and origin == target - h, fid)
        avail = got[got["persistence_available"] == 1]
        src = (np.array([int(m[:4]) * 12 + int(m[5:7]) - 1 for m in avail["persistence_source_month"]])
               if len(avail) else np.zeros(0))
        c.check(f"H{h}.persistence_not_after_origin", bool(np.all(src <= origin)) and
                bool(np.all(avail["persistence_age_months"].to_numpy() == origin - src)), fid)
        for col in ("persistence_available", "persistence_phase", "persistence_source_month"):
            c.check(f"H{h}.persistence_matches_prepared", np.array_equal(
                got[col].fillna("").astype(str).to_numpy(), k[col].fillna("").astype(str).to_numpy()), f"{fid}/{col}")
        for arm in ("geo", "pool"):
            raw = got[[f"{arm}_{q}_raw" for q in Q]].to_numpy()
            star = got[[f"{arm}_{q}_star" for q in Q]].to_numpy()
            c.check(f"H{h}.{arm}_projection", np.allclose(_project(raw), star, atol=1e-12, rtol=0)
                    and np.array_equal(_phase(star), got[f"{arm}_phase"].to_numpy()), fid)
        g_digests = got["global_identity"].unique()
        if not c.check(f"H{h}.one_global_per_fold", len(g_digests) == 1, fid):
            continue
        gid = g_digests[0]
        ident = models.identity(gid)
        c.check(f"H{h}.current_global_identity", ident["scope"] == "stage3-global" and ident["H"] == h and
                ident["fitting_origin"] == origin and ident["window"] == [origin - window + 1, origin] and
                ident["X_artifact_sha256"] == sha["X"] and ident["keys_artifact_sha256"] == sha["keys"], fid)
        pooled = models.quartet(gid).predict_raw(np.asarray(X[rows]))
        c.check(f"H{h}.pooled_lineage", np.array_equal(pooled, got[[f"pool_{q}_raw" for q in Q]].to_numpy()), fid)
        expect_region = np.array([region_of.get(int(a), "") for a in got["admin_code"]], dtype=object).astype(str)
        got_region = got["region"].fillna("").astype(str).to_numpy()
        c.check(f"H{h}.region_from_frozen_map", np.array_equal(got_region, expect_region), fid)
        local = (got["route"] == "local").to_numpy()
        c.check(f"H{h}.non_local_equals_pooled", all(np.array_equal(got.loc[~local, f"geo_{q}_raw"],
                                                                     got.loc[~local, f"pool_{q}_raw"]) for q in Q), fid)
        c.check(f"H{h}.unmapped_routes_global", bool((got.loc[expect_region == "", "route"] == "unmapped_area_global").all()), fid)
        fold_decisions = by_fold.get(fid, [])
        if not record["accepted_split"]:
            c.check(f"H{h}.no_gate_without_split", not fold_decisions and not local.any()
                    and not (s3 / f"pairs_{fid}.csv.gz").exists(), fid)
            continue
        c.check(f"H{h}.gate_decision_inventory", sorted(str(d["region"]) for d in fold_decisions) == sorted(regions), fid)
        dates = sorted(observed[observed < origin].tolist(), reverse=True)[:max_dates]
        c.check(f"H{h}.gate_dates", json.loads(str(lrow["gate_dates"].iloc[0]).replace("'", '"')) == dates, fid)
        pairs_path = s3 / f"pairs_{fid}.csv.gz"
        pairs = pd.read_csv(pairs_path, float_precision="round_trip") if pairs_path.is_file() else pd.DataFrame(
            columns=["region", "validation_month", "admin_code", "target_ord"])
        pairs["region"] = pairs["region"].astype(str)
        for d in fold_decisions:
            region = str(d["region"])
            enabled_expected = _replay_region_gate(
                c, h, fid, region, regions.get(region, set()), pairs[pairs["region"] == region], dates, keys, X, months,
                area, row_of, models, floors_fit, floors_val, min_dates, window)
            c.check(f"H{h}.gate_decision", d["enabled"] == enabled_expected, f"{fid}/{region}")
            rows_te = got_region == region
            cur = np.flatnonzero((months >= origin - window + 1) & (months <= origin))
            cur = cur[np.isin(area[cur], list(regions.get(region, set())))]
            cur_ok = _meets(_support(keys.iloc[cur]), floors_fit)
            if d.get("route") == "local":
                c.check(f"H{h}.local_requires_gate_and_support", enabled_expected and cur_ok, f"{fid}/{region}")
                lid = d.get("local_identity", "")
                lident = models.records.get(lid, {}).get("identity")
                c.check(f"H{h}.local_provider_identity", lident is not None and lident["scope"] == "stage3-local" and
                        lident["fitting_origin"] == origin and lident["region_node"] == region and
                        lident["global_identity"] == gid, f"{fid}/{region}")
                region_rows = got[rows_te]
                c.check(f"H{h}.local_route_matches_gate", bool((region_rows["route"] == "local").all()) and
                        bool((region_rows["provider"] == lid).all()), f"{fid}/{region}")
                if lident is not None and len(region_rows):
                    again = models.quartet(lid).predict_raw(np.asarray(X[rows[rows_te]]))
                    c.check(f"H{h}.local_lineage", np.array_equal(again, region_rows[[f"geo_{q}_raw" for q in Q]].to_numpy()),
                            f"{fid}/{region}")
            else:
                c.check(f"H{h}.no_local_without_gate", not (got.loc[rows_te, "route"] == "local").any(), f"{fid}/{region}")
                if d.get("reason") == "current_fit_support":
                    c.check(f"H{h}.current_support_fallback", enabled_expected and not cur_ok, f"{fid}/{region}")
        adopted = {str(d["region"]) for d in fold_decisions if d.get("route") == "local"}
        c.check(f"H{h}.local_rows_have_decisions", set(got.loc[local, "region"].astype(str)) <= adopted, fid)
    return pred


def _replay_region_gate(c, h, fid, region, members, part, dates, keys, X, months, area, row_of, models,
                        floors_fit, floors_val, min_dates, window) -> bool:
    """Full historical keys per date, lineage of both quartets, support and the gate decision."""
    truth_all, global_all, local_all, ok_dates, months_seen, areas_seen = [], [], [], 0, set(), set()
    for u in dates:
        v = u - h
        expected = np.flatnonzero((months == u) & np.isin(area, list(members)))
        rows_u = part[part["validation_month"] == u]
        if len(expected) == 0:
            c.check(f"H{h}.pair_keys", len(rows_u) == 0, f"{fid}/{region}/{u}")
            continue
        if not c.check(f"H{h}.pair_keys", set(map(tuple, rows_u[["admin_code", "target_ord"]].to_numpy()))
                       == set(map(tuple, keys.iloc[expected][["admin_code", "target_ord"]].to_numpy()))
                       and len(rows_u) == len(expected), f"{fid}/{region}/{u}"):
            continue
        rows = np.array([row_of[k] for k in map(tuple, rows_u[["admin_code", "target_ord"]].to_numpy())])
        gdig = rows_u["global_identity"].unique()
        gident = models.records.get(gdig[0], {}).get("identity") if len(gdig) == 1 else None
        if not c.check(f"H{h}.pair_global_identity", gident is not None and gident["fitting_origin"] == v
                       and gident["scope"] == "stage3-global", f"{fid}/{region}/{u}"):
            continue
        g_raw = models.quartet(gdig[0]).predict_raw(np.asarray(X[rows]))
        c.check(f"H{h}.pair_global_lineage", np.array_equal(g_raw, rows_u[[f"global_{q}_raw" for q in Q]].to_numpy()),
                f"{fid}/{region}/{u}")
        fit_rows = np.flatnonzero((months >= v - window + 1) & (months <= v))
        fit_rows = fit_rows[np.isin(area[fit_rows], list(members))]
        supported = _meets(_support(keys.iloc[fit_rows]), floors_fit)
        ok = bool(rows_u["local_fit_ok"].astype(bool).iloc[0])
        c.check(f"H{h}.pair_local_support", ok == supported and rows_u["local_fit_ok"].nunique() == 1, f"{fid}/{region}/{u}")
        provider = list(rows_u["local_routed_provider"].unique())
        if ok:
            lid = rows_u["local_identity"].iloc[0]
            lident = models.records.get(lid, {}).get("identity")
            c.check(f"H{h}.pair_local_identity", lident is not None and lident["fitting_origin"] == v and
                    lident["region_node"] == region and lident["global_identity"] == gdig[0] and provider == [lid],
                    f"{fid}/{region}/{u}")
            l_raw = models.quartet(lid).predict_raw(np.asarray(X[rows])) if lident is not None else None
        else:
            c.check(f"H{h}.pair_fallback_is_global", provider == [gdig[0]], f"{fid}/{region}/{u}")
            l_raw = g_raw
        if l_raw is not None:
            c.check(f"H{h}.pair_local_lineage", np.array_equal(l_raw, rows_u[[f"local_routed_{q}_raw" for q in Q]].to_numpy()),
                    f"{fid}/{region}/{u}")
        g_phase = _phase(_project(rows_u[[f"global_{q}_raw" for q in Q]].to_numpy()))
        l_phase = _phase(_project(rows_u[[f"local_routed_{q}_raw" for q in Q]].to_numpy()))
        c.check(f"H{h}.pair_phases", np.array_equal(g_phase, rows_u["phase_global"].to_numpy()) and
                np.array_equal(l_phase, rows_u["phase_local_routed"].to_numpy()), f"{fid}/{region}/{u}")
        truth_all.append(keys.iloc[rows]["phase_truth"].to_numpy())
        global_all.append(g_phase)
        local_all.append(l_phase)
        ok_dates += int(ok)
        months_seen.add(u)
        areas_seen |= set(rows_u["admin_code"].astype(int))
    c.check(f"H{h}.pair_dates_complete", set(part["validation_month"].astype(int)) <= set(dates), f"{fid}/{region}")
    if not truth_all:
        return False
    truth, g, l = np.concatenate(truth_all), np.concatenate(global_all), np.concatenate(local_all)
    support = {"keys": len(truth), "areas": len(areas_seen), "target_months": len(months_seen),
               "crisis_keys": int((truth >= 3).sum()), "noncrisis_keys": int((truth < 3).sum())}
    if not _meets(support, floors_val) or ok_dates < min_dates:
        return False
    f_g, f_l = _f1(_counts(truth, g)), _f1(_counts(truth, l))
    return f_g is not None and f_l is not None and (f_l - f_g) > Fraction(1, 100)


def _replay_report(c, run_dir, h, pred, calendar, frozen, report):
    hrep = report["horizons"].get(str(h))
    if not c.check(f"H{h}.report_present", hrep is not None):
        return
    c.check(f"H{h}.report_predictions_digest", hrep.get("predictions_sha256") ==
            sha256_file(run_dir / "stage3" / f"h{h:02d}" / "predictions.csv.gz"))
    for period in ("main", "supplementary"):
        entry = hrep.get(period)
        if not c.check(f"H{h}.report_period", entry is not None, period):
            continue
        e_all = pred[pred["period"] == period]
        e_persist = e_all[e_all["persistence_available"] == 1]
        cov = entry["coverage"]
        cal = calendar[(calendar["horizon_months"] == h) & (calendar["period"] == period)]
        c.check(f"H{h}.report_coverage", cov["E_all_keys"] == len(e_all) and cov["E_persist_keys"] == len(e_persist)
                and cov["scheduled_folds"] == len(cal) and cov["countries"] == e_all["country_key"].nunique(), period)
        for cohort, frame, arms in (("E_all", e_all, ("geo", "pool")), ("E_persist", e_persist, ("geo", "persistence"))):
            saved = entry.get(cohort)
            if not c.check(f"H{h}.report_cohort_present", saved is not None, f"{period}/{cohort}"):
                continue
            if len(frame) == 0:
                c.check(f"H{h}.report_empty_cohort", saved.get("status") == "empty_cohort" and saved.get("n") == 0,
                        f"{period}/{cohort}")
                continue
            panels = {arm: _arm_panel(frame, arm) for arm in arms}
            for arm in arms:
                s = saved[arm]
                flat = {"binary.accuracy": s["binary"]["accuracy"], "binary.precision": s["binary"]["precision"],
                        "binary.recall": s["binary"]["recall"], "binary.f1": s["binary"]["f1"],
                        "binary.f2": s["binary"]["f2"], "four_class.accuracy": s["four_class"]["accuracy"],
                        "four_class.macro_f1": s["four_class"]["macro_f1"],
                        "q3_r2_projected": s.get("q3_r2_projected"), "q3_r2_raw": s.get("q3_r2_raw")}
                for name, value in flat.items():
                    c.check(f"H{h}.report_metric", _same(value, panels[arm][name]), f"{period}/{cohort}/{arm}/{name}")
                c.check(f"H{h}.report_counts", s["binary"]["counts"] == panels[arm]["counts"], f"{period}/{cohort}/{arm}")
            a, b = arms
            delta = saved[f"delta_{a}_minus_{b}"]
            for name in delta:
                pa, pb = panels[a][name], panels[b][name]
                c.check(f"H{h}.report_delta", _same(delta[name], None if pa is None or pb is None else pa - pb),
                        f"{period}/{cohort}/{name}")
        if period == "main":
            for name, frame, other in (("geo_vs_pool_E_all", e_all, "pool"),
                                       ("geo_vs_persistence_E_persist", e_persist, "persistence")):
                _replay_bootstrap(c, run_dir, h, name, frame, other, entry.get("bootstrap", {}).get(name))
        routes = entry.get("routes", {})
        c.check(f"H{h}.report_routes", routes.get("rows", {}).get("local") == int((e_all["route"] == "local").sum())
                and routes.get("rows", {}).get("denominator_cohort_rows") == len(e_all)
                and routes.get("map", {}).get("learned_map_areas") == len(frozen["map"]), period)


def _replay_bootstrap(c, run_dir, h, name, frame, other, saved):
    if not c.check(f"H{h}.bootstrap_present", saved is not None, name):
        return
    countries = sorted(frame["country_key"].unique().tolist())
    k = len(countries)
    c.check(f"H{h}.bootstrap_countries", saved["countries"] == countries and saved["K"] == k and
            saved["draws"] == DRAWS and saved["seed"] == SEED, name)
    pa = _f1(_counts(frame["phase_truth"], frame["geo_phase"]))
    pb = _f1(_counts(frame["phase_truth"], frame[_arm_cols(other)[0]]))
    point = None if pa is None or pb is None else float(pa - pb)
    c.check(f"H{h}.bootstrap_point", _same(saved["point_delta"], point), name)
    if k == 0:
        c.check(f"H{h}.bootstrap_empty", saved["interval"] is None, name)
        return
    counts = {}
    for arm in ("geo", other):
        col = _arm_cols(arm)[0]
        counts[arm] = np.array([[_counts(p["phase_truth"], p[col])[x] for x in ("tp", "fp", "fn", "tn")]
                                for p in (frame[frame["country_key"] == cn] for cn in countries)])
    mult = _rng_multiplicities(k)

    def f1(ct):
        den = 2 * ct[:, 0] + ct[:, 1] + ct[:, 2]
        return np.where(den > 0, 2 * ct[:, 0] / np.where(den > 0, den, 1), np.nan)

    delta = f1(mult @ counts["geo"]) - f1(mult @ counts[other])
    eligible = k >= 2 and point is not None and bool(np.isfinite(delta).all())
    c.check(f"H{h}.bootstrap_eligibility", (saved["interval"] is not None) == eligible, name)
    if eligible and saved["interval"] is not None:
        lo, hi = np.percentile(delta, [2.5, 97.5], method="linear")
        c.check(f"H{h}.bootstrap_interval", np.allclose([lo, hi], saved["interval"], rtol=1e-12, atol=1e-12), name)
    draws_path = run_dir / "report" / f"bootstrap_h{h:02d}_{name}.csv.gz"
    if c.check(f"H{h}.bootstrap_draws_present", draws_path.is_file(), name):
        draws = pd.read_csv(draws_path)
        c.check(f"H{h}.bootstrap_draws", np.array_equal(draws[[f"m::{x}" for x in countries]].to_numpy(), mult)
                and np.allclose(draws["delta"].to_numpy(), delta, equal_nan=True), name)


def _replay_requests(c, run_dir, models, horizons, artifacts):
    keys_by_h = {h: pd.read_csv(run_dir / "prepared" / f"keys_h{h:02d}.csv.gz") for h in horizons}
    maps = {h: pd.read_csv(run_dir / "stage1" / f"frozen_map_h{h:02d}.csv", dtype={"node_id": str}) for h in horizons}
    for line in (run_dir / "stage3" / "model_requests.jsonl").read_text().splitlines():
        use = json.loads(line)
        c.check("models.request_succeeded", use["status"] in ("fit", "hit"), use.get("fold", ""))
        digest = use["identity_sha256"]
        if not c.check("models.request_record", digest in models.records, digest):
            continue
        c.guard("models.artifact_valid", models.quartet, digest)
        ident = models.identity(digest)
        h = ident["H"]
        c.check("models.window_ends_at_origin", ident["window"][1] == ident["fitting_origin"] == use["fitting_origin"])
        c.check("models.prepared_artifacts", ident["X_artifact_sha256"] == artifacts[f"X_rich561_h{h:02d}.npy"]
                and ident["keys_artifact_sha256"] == artifacts[f"keys_h{h:02d}.csv.gz"], digest)
        if use.get("use") == "gate":
            c.check("models.gate_origin_is_U_minus_H", use["fitting_origin"] == use["gate_month"] - use["H"])
        keys = keys_by_h[h]
        months = keys["target_ord"].to_numpy()
        lo, hi = ident["window"]
        rows = np.flatnonzero((months >= lo) & (months <= hi))
        if ident["scope"] == "stage3-local":
            areas = maps[h].loc[maps[h]["node_id"] == ident["region_node"], "admin_code"].to_numpy(dtype=np.int64)
            c.check("models.region_members", array_digest(np.sort(areas)) == ident["region_areas"], ident["region_node"])
            rows = rows[np.isin(keys["admin_code"].to_numpy()[rows], areas)]
            glob = models.records.get(ident["global_identity"])
            c.check("models.local_has_same_origin_global", glob is not None and
                    glob["identity"]["fitting_origin"] == ident["fitting_origin"] and
                    ident["global_boosters"] == glob["booster_sha256"], digest)
        c.check("models.fit_rows_rebuilt", array_digest(rows.astype(np.int64)) == ident["fit_rows"]
                and len(rows) == ident["n_rows"], digest)
        c.check("models.target_order",
                target_digests(keys[list(Q)].to_numpy(dtype=np.float64)[rows]) == ident["y_sha256"], digest)
