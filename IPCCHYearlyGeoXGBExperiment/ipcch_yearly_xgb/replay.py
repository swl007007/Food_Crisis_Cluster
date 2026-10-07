"""Zero-fit replay and independent verification (design section 9).

Part A re-executes prediction and reporting against a read-only model store
(any missing model is an error, never a fit) and requires byte-identical keyed
artifacts. Because that re-uses the producer, part B verifies independently
with separately written code: the annual calendar and gate dates, fit pools
and decay-weight digests of every saved model identity, complete validation
keys, count-based support and exact-fraction gate decisions from saved pairs,
fixed within-block routing, projection/decoding by exhaustive enumeration of
contiguous blocks with accurately summed means, and pooled metric counts and
bootstrap intervals recomputed from keyed cohorts.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_yearly_xgb import report, run
from ipcch_yearly_xgb.artifacts import sha256_file, write_json

TARGETS = ("q2", "q3", "q4", "q5")
VOLATILE = {"store_counts", "elapsed_seconds", "predict_summary_sha256", "summary_sha256"}


class Checker:
    def __init__(self):
        self.passed, self.failures = 0, []

    def check(self, name: str, ok, detail: str = "") -> None:
        if bool(ok):
            self.passed += 1
        else:
            self.failures.append(f"{name}: {detail}")


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def compare_trees(c: Checker, a_root: Path, b_root: Path) -> None:
    a = sorted(p.relative_to(a_root) for p in a_root.rglob("*") if p.is_file() and p.name != "model_requests.jsonl")
    b = sorted(p.relative_to(b_root) for p in b_root.rglob("*") if p.is_file() and p.name != "model_requests.jsonl")
    c.check(f"inventory:{a_root.name}", a == b, str(set(a) ^ set(b)))
    for rel in a:
        if not (b_root / rel).is_file():
            continue
        if rel.suffix == ".json":
            ja = json.loads((a_root / rel).read_text(encoding="utf-8"))
            jb = json.loads((b_root / rel).read_text(encoding="utf-8"))
            c.check(f"json:{a_root.name}/{rel}", _strip(ja) == _strip(jb), "content differs")
        else:
            c.check(f"bytes:{a_root.name}/{rel}", sha256_file(a_root / rel) == sha256_file(b_root / rel), "differs")


# ------------------------------------------------------------ independent pieces

def ind_origin(h: int, first_label: str, u: int) -> int:
    fy, fm = map(int, first_label.split("-"))
    first = fy * 12 + fm - 1
    year_start = (u // 12) * 12
    return (first - h) if (u // 12 == first // 12 and u >= first) else (year_start - h)


def ind_project(raw) -> tuple[list, int]:
    """Exhaustive least-squares non-increasing fit over contiguous partitions, fsum means, clip to [0, 1]."""
    vals = [float(v) for v in raw]
    best, best_sse = None, None
    for cuts in itertools.product((0, 1), repeat=3):
        blocks, cur = [], [vals[0]]
        for i, cut in enumerate(cuts):
            if cut:
                blocks.append(cur)
                cur = [vals[i + 1]]
            else:
                cur.append(vals[i + 1])
        blocks.append(cur)
        means = []
        for b in blocks:
            try:
                means.append(math.fsum(b) / len(b))
            except OverflowError:
                means.append(math.fsum(v / len(b) for v in b))
        if any(means[i] < means[i + 1] for i in range(len(means) - 1)):
            continue
        fit = [m for m, b in zip(means, blocks) for _ in b]
        sse = math.fsum((f - v) ** 2 for f, v in zip(fit, vals))
        if best is None or sse < best_sse:
            best, best_sse = fit, sse
    star = [min(1.0, max(0.0, v)) for v in best]
    phase = 1 + sum(1 for v in star if v >= 0.20)
    return star, phase


def ind_counts(truth, pred) -> tuple[int, int, int, int]:
    t = np.asarray(truth) >= 3
    p = np.asarray(pred) >= 3
    return int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum()), int((~t & ~p).sum())


def ind_f1(tp, fp, fn) -> Fraction | None:
    d = 2 * tp + fp + fn
    return Fraction(2 * tp, d) if d else None


def _sha(arr, dtype) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr, dtype=dtype)).tobytes()).hexdigest()


def verify_horizon(c: Checker, hz, calendar, contract: dict, hdir: Path, models_root: Path, ledger_lines: list) -> None:
    h = hz.h
    first = contract["protocol"]["first_main_target"][str(h)]
    floor, vfloor = contract["support"]["local_fit"], contract["support"]["validation"]
    pred = report.read_predictions(hdir / "predictions.csv.gz")
    gates = [json.loads(x) for x in (hdir / "gate_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    blocks = json.loads((hdir / "block_ledger.json").read_text(encoding="utf-8"))
    t_all = hz.keys["target_ord"].to_numpy()
    area_all = hz.keys["admin_code"].to_numpy()
    observed = np.unique(t_all)
    # calendar: anchors/origins/gate dates from the fold calendar, independently
    folds = calendar[calendar["horizon_months"] == h]
    for b in blocks:
        part = folds[(folds["period"] == b["period"]) & (folds["target_ord"] // 12 == b["year"])]
        anchor = int(part["target_ord"].min())
        c.check(f"H{h}/{b['block_id']}:anchor", anchor == b["anchor_ord"], f"{anchor} vs {b['anchor_ord']}")
        o = ind_origin(h, first, anchor)
        c.check(f"H{h}/{b['block_id']}:origin", o == b["fit_origin_ord"], f"{o} vs {b['fit_origin_ord']}")
        if b["status"] == "scored":
            dates = sorted([int(u) for u in observed if u < o])[-6:][::-1]
            c.check(f"H{h}/{b['block_id']}:gate_dates", dates == b["gate_dates"], f"{dates} vs {b['gate_dates']}")
    c.check(f"H{h}:rows_fit_origin", all(ind_origin(h, first, int(a)) == int(o) for a, o in
                                         zip(pred["annual_anchor_ord"], pred["fit_origin_ord"])))
    c.check(f"H{h}:row_origin", np.array_equal(pred["row_origin_ord"].to_numpy(), pred["target_ord"].to_numpy() - h))
    c.check(f"H{h}:fit_origin_before_targets", bool((pred["fit_origin_ord"] < pred["target_ord"]).all()))
    # projection/decoding by enumeration, all saved arms
    for arm in ("pool", "local", "geo"):
        raw = pred[[f"{arm}_{q}_raw" for q in TARGETS]].to_numpy()
        star = pred[[f"{arm}_{q}_star" for q in TARGETS]].to_numpy()
        ph = pred[f"{arm}_phase"].to_numpy()
        ok_rows = ~np.isnan(raw).any(axis=1)
        bad = 0
        for i in np.flatnonzero(ok_rows):
            s, p = ind_project(raw[i])
            if p != ph[i] or not np.array_equal(np.asarray(s), star[i]):
                bad += 1
        c.check(f"H{h}:projection_{arm}", bad == 0, f"{bad} rows")
    # routing fixed within block and consistent with gates
    gmap = {(g["block_id"], g["region"]): g for g in gates}
    for (bid, region), part in pred[pred["region"] != ""].groupby(["block_id", "region"]):
        g = gmap.get((bid, region))
        expect = "local" if (g and g["enabled"] and g.get("current_fit_supported")) else None
        routes = set(part["route"])
        c.check(f"H{h}/{bid}/{region}:route_fixed", len(routes) == 1, str(routes))
        if expect:
            c.check(f"H{h}/{bid}/{region}:route_local", routes == {"local"})
        else:
            c.check(f"H{h}/{bid}/{region}:route_pool", "local" not in routes)
    use_l = (pred["route"] == "local").to_numpy()
    for q in TARGETS:
        geo, pool, loc = (pred[f"{a}_{q}_raw"].to_numpy() for a in ("geo", "pool", "local"))
        c.check(f"H{h}:geo_is_pool_{q}", np.array_equal(geo[~use_l], pool[~use_l]))
        c.check(f"H{h}:geo_is_local_{q}", np.array_equal(geo[use_l], loc[use_l]))
    c.check(f"H{h}:unmapped", bool((pred.loc[pred["region"] == "", "route"] == "unmapped_area_pool").all()))
    # gate decisions from saved pairs and keys
    for b in [b for b in blocks if b["status"] == "scored"]:
        path = hdir / f"pairs_{b['block_id']}.csv.gz"
        pairs = pd.read_csv(path, float_precision="round_trip", dtype={"region": str, "local_provider": str}) \
            if path.is_file() else pd.DataFrame(columns=["region", "row"])
        if len(pairs):
            pairs["local_provider"] = pairs["local_provider"].fillna("")
            for _, r in pairs.sample(n=min(200, len(pairs)), random_state=0).iterrows():
                raw_p = [r[f"pool_{q}_raw"] for q in TARGETS]
                raw_l = [r[f"local_routed_{q}_raw"] for q in TARGETS]
                c.check(f"H{h}/{b['block_id']}:pair_phase", ind_project(raw_p)[1] == r["phase_pool"]
                        and ind_project(raw_l)[1] == r["phase_local_routed"])
            fb = pairs[~pairs["local_fit_ok"].astype(bool)]
            for q in TARGETS:
                c.check(f"H{h}/{b['block_id']}:fallback_equals_pool_{q}",
                        np.array_equal(fb[f"pool_{q}_raw"].to_numpy(), fb[f"local_routed_{q}_raw"].to_numpy()))
        for g in [g for g in gates if g["block_id"] == b["block_id"]]:
            region = g["region"]
            members = set(hz.regions[region].tolist())
            expect_rows = sorted(i for u in b["gate_dates"] for i in np.flatnonzero(t_all == u)
                                 if int(area_all[i]) in members)
            pr = pairs[pairs["region"] == region] if len(pairs) else pairs
            c.check(f"H{h}/{b['block_id']}/{region}:validation_keys_complete",
                    sorted(pr["row"].astype(int).tolist()) == expect_rows)
            n = len(pr)
            truth = hz.keys["phase_truth"].to_numpy()[pr["row"].astype(int).to_numpy()] if n else np.zeros(0)
            crisis = int((truth >= 3).sum())
            ok_dates = set(pr.loc[pr["local_fit_ok"].astype(bool), "validation_month"].astype(int)) if n else set()
            rec = {"keys": n, "areas": int(pr["admin_code"].nunique()) if n else 0,
                   "target_months": int(pr["validation_month"].nunique()) if n else 0,
                   "crisis_keys": crisis, "noncrisis_keys": n - crisis}
            supported = all(rec[k] >= v for k, v in vfloor.items()) and \
                len(ok_dates) >= contract["support"]["min_successful_local_dates"]
            c.check(f"H{h}/{b['block_id']}/{region}:support", supported == g["historical_support"],
                    f"{rec}, dates {len(ok_dates)}")
            for v_month in ok_dates:  # historical regional fitting support, recounted from keys
                vo = ind_origin(h, first, v_month)
                sel = (t_all <= vo) & np.isin(area_all, list(members))
                cnt = (int(sel.sum()), len(set(area_all[sel].tolist())), len(set(t_all[sel].tolist())))
                c.check(f"H{h}/{b['block_id']}/{region}:hist_fit_support",
                        all(x >= f for x, f in zip(cnt, (floor["keys"], floor["areas"], floor["target_months"]))))
            enabled = False
            if supported:
                fp_ = ind_f1(*ind_counts(truth, pr["phase_pool"])[:3])
                fl_ = ind_f1(*ind_counts(truth, pr["phase_local_routed"])[:3])
                enabled = fp_ is not None and fl_ is not None and fl_ - fp_ > Fraction(1, 100)
            c.check(f"H{h}/{b['block_id']}/{region}:enabled", enabled == g["enabled"])
            if g["test_keys"]:
                o = b["fit_origin_ord"]
                sel = (t_all <= o) & np.isin(area_all, list(members))
                cnt = (int(sel.sum()), len(set(area_all[sel].tolist())), len(set(t_all[sel].tolist())))
                cur = all(x >= f for x, f in zip(cnt, (floor["keys"], floor["areas"], floor["target_months"])))
                c.check(f"H{h}/{b['block_id']}/{region}:current_support", cur == g["current_fit_supported"])
    # every fitted model identity: pool rows and decay weights recomputed from the formula
    seen = set()
    for e in ledger_lines:
        if e.get("H") != h or e["status"] not in ("fit", "hit") or e["identity_sha256"] in seen:
            continue
        seen.add(e["identity_sha256"])
        d = e["identity_sha256"]
        rec = json.loads((models_root / d[:2] / d / "record.json").read_text(encoding="utf-8"))
        ident = rec["identity"]
        o = ident["fit_origin"]
        sel = t_all <= o
        if ident["scope"] == "yearly-local":
            sel &= np.isin(area_all, list(hz.regions[ident["region_node"]].tolist()))
        rows = np.flatnonzero(sel).astype(np.int64)
        w = np.array([0.5 ** ((o - int(t)) / 24.0) for t in t_all[rows]], dtype=np.float64)
        c.check(f"model {d[:12]}:n_rows", ident["n_rows"] == len(rows))
        c.check(f"model {d[:12]}:weights_protocol", ident["weights"]["protocol_sha256"] == _sha(w, np.float64))
        c.check(f"model {d[:12]}:weights_effective", ident["weights"]["effective_float32_sha256"] ==
                _sha(w.astype(np.float32), np.float32))
        for q in TARGETS:
            c.check(f"model {d[:12]}:{q}_record_weights",
                    rec["fit_records"][q]["weights"]["protocol_sha256"] == ident["weights"]["protocol_sha256"])
        if ident["scope"] == "yearly-local":
            parent = json.loads((models_root / ident["global_identity"][:2] / ident["global_identity"] /
                                 "record.json").read_text(encoding="utf-8"))
            c.check(f"model {d[:12]}:parent_origin", parent["identity"]["fit_origin"] == o)
            c.check(f"model {d[:12]}:parent_boosters", parent["booster_sha256"] == ident["global_boosters"])


def verify_report(c: Checker, run_dir: Path, rep: dict, horizons: list, p6_loader) -> None:
    for h in horizons:
        pred = report.attach_p6(report.read_predictions(run_dir / "predict" / f"h{h:02d}" / "predictions.csv.gz"),
                                p6_loader(h))
        for period in ("main", "supplementary"):
            e = pred[pred["period"] == period]
            entry = rep["horizons"][str(h)][period]
            if not len(e):
                continue
            for arm in ("pool", "geo", "p6pool", "p6geo"):
                tp, fp, fn, tn = ind_counts(e["phase_truth"], e[f"{arm}_phase"])
                got = entry["E_all"]["panels"][arm]["binary"]["counts"]
                c.check(f"report H{h}/{period}/{arm}:counts", got == {"tp": tp, "fp": fp, "fn": fn, "tn": tn})
                f1 = ind_f1(tp, fp, fn)
                c.check(f"report H{h}/{period}/{arm}:f1", (f1 is None and entry["E_all"]["panels"][arm]["binary"]["f1"]
                                                          is None) or float(f1) == entry["E_all"]["panels"][arm]["binary"]["f1"])
            if period == "main":
                ep = e[e["persistence_available"] == 1]
                for name, frame, other in (("geo_minus_pool_E_all", e, "pool_phase"),
                                           ("geo_minus_persistence_E_persist", ep, "persistence_phase")):
                    b = entry["bootstrap"][name]
                    countries = sorted(frame["country_key"].unique().tolist())
                    ca = np.array([ind_counts(frame.loc[frame.country_key == k, "phase_truth"],
                                              frame.loc[frame.country_key == k, "geo_phase"]) for k in countries])
                    cb = np.array([ind_counts(frame.loc[frame.country_key == k, "phase_truth"],
                                              frame.loc[frame.country_key == k, other]) for k in countries])
                    c.check(f"report H{h}/{name}:country_counts", ca.tolist() == b["country_counts"]["geo"])
                    rng = np.random.default_rng(42)
                    deltas = []
                    for _ in range(2000):
                        m = np.bincount(rng.integers(0, len(countries), size=len(countries)), minlength=len(countries))
                        sa, sb = m @ ca, m @ cb
                        fa, fb = ind_f1(*sa[:3]), ind_f1(*sb[:3])
                        deltas.append(np.nan if fa is None or fb is None else float(fa) - float(fb))
                    deltas = np.array(deltas)
                    if b["interval"] is not None:
                        lo, hi = np.percentile(deltas, (2.5, 97.5), method="linear")
                        c.check(f"report H{h}/{name}:interval", abs(lo - b["interval"][0]) < 1e-12 and
                                abs(hi - b["interval"][1]) < 1e-12, f"{lo},{hi} vs {b['interval']}")
                    else:
                        c.check(f"report H{h}/{name}:interval_na", not np.isfinite(deltas).all() or len(countries) < 2)


def run_replay(run_dir: Path, horizons: dict, calendar, contract: dict, env: dict, source_inventory: dict,
               expect_fits: int | None = None, p6_loader=None) -> dict:
    out = run_dir / "replay"
    out.mkdir()
    c = Checker()
    # the replay store is the original models directory, opened read-only
    run.run_predict(out, horizons, calendar, contract, env, source_inventory, readonly=True,
                    models_root=run_dir / "models")
    compare_trees(c, run_dir / "predict", out / "predict")
    report.run_report(out, contract, p6_loader=p6_loader)
    compare_trees(c, run_dir / "report", out / "report")
    lines = []
    for p in sorted((run_dir / "predict").glob("model_requests*.jsonl")):
        lines += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    for h, hz in horizons.items():
        verify_horizon(c, hz, calendar, contract, run_dir / "predict" / f"h{h:02d}", run_dir / "models", lines)
    rep = json.loads((run_dir / "report" / "report.json").read_text(encoding="utf-8"))
    verify_report(c, run_dir, rep, list(horizons), p6_loader or report.sources.load_p6_predictions)
    fits = {e["identity_sha256"] for e in lines if e["status"] == "fit"}
    stored = {p.name for p in (run_dir / "models").glob("*/*") if p.is_dir() and not p.name.endswith(".tmp")}
    c.check("inventory:store_equals_fits", stored == fits, f"{len(stored)} stored vs {len(fits)} fitted")
    c.check("inventory:no_failed", not any(e["status"] == "failed" for e in lines))
    if expect_fits is not None:
        c.check("inventory:scalar_fits", 4 * len(fits) == expect_fits, f"{4 * len(fits)} vs {expect_fits}")
    replay_lines = (out / "predict" / "model_requests.jsonl").read_text(encoding="utf-8").splitlines()
    c.check("replay:zero_fits", not any(json.loads(x)["status"] == "fit" for x in replay_lines))
    result = {"status": "passed" if not c.failures else "failed", "checks_passed": c.passed,
              "n_failures": len(c.failures), "failures": c.failures[:200], "unique_quartets": len(fits),
              "scalar_fits": 4 * len(fits), "replay_requests": len(replay_lines)}
    write_json(out / "replay.json", result)
    return result
