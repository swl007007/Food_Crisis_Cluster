"""P5 reporting from saved keyed Stage3 predictions only (R8, R25-R27, R49).

Structure adapted from ``IPCCHGeoRFExperiment/report_results.py`` (per-H
paired country-cluster bootstrap with shared multiplicities, percentile
interval with point/cluster/defined-draw checks) under the R49 contract:
2000 draws, ``numpy.random.default_rng(42)`` reset per H x cohort, every draw
kept, interval only when K >= 2, the point delta is defined and all 2000
replicate deltas are finite. Main 2023-2025 and observed 2026 are separate;
2026 and country/month diagnostics are point estimates only.

Arms per row: routed GeoXGB (``geo_*``), matched pooled (``pool_*``, the same
fold's G_H global quartet) and persistence. Cohorts: E_all = all scored keys;
E_persist = keys with lawful persistence (``persistence_available == 1``).
Nothing here fits a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_geoxgb import metrics
from ipcch_geoxgb.artifacts import record_incomplete, sha256_file, write_json
from ipcch_geoxgb.errors import TechnicalError

DRAWS = 2000
SEED = 42
PERCENTILES = (2.5, 97.5)
GZ = {"method": "gzip", "mtime": 0}
SCALARS = (
    ("binary", "accuracy"), ("binary", "precision"), ("binary", "recall"), ("binary", "f1"), ("binary", "f2"),
    ("four_class", "accuracy"), ("four_class", "macro_f1"),
)


def arm_columns(arm: str) -> dict:
    if arm == "persistence":
        return {"phase": "persistence_phase", "q3_star": "persistence_q3", "q3_raw": "persistence_q3"}
    return {"phase": f"{arm}_phase", "q3_star": f"{arm}_q3_star", "q3_raw": f"{arm}_q3_raw"}


def panel(frame: pd.DataFrame, arm: str) -> dict:
    cols = arm_columns(arm)
    return metrics.metric_panel(
        frame["phase_truth"].to_numpy(), frame[cols["phase"]].to_numpy(),
        frame["q3_truth"].to_numpy(), frame[cols["q3_star"]].to_numpy(), frame[cols["q3_raw"]].to_numpy(),
    )


def flat(p: dict) -> dict:
    out = {f"{group}.{name}": p[group][name] for group, name in SCALARS}
    out["q3_r2_projected"] = p.get("q3_r2_projected")
    out["q3_r2_raw"] = p.get("q3_r2_raw")
    return out


def deltas(a: dict, b: dict) -> dict:
    fa, fb = flat(a), flat(b)
    return {k: (None if fa[k] is None or fb[k] is None else fa[k] - fb[k]) for k in fa}


def country_counts(frame: pd.DataFrame, arm: str, countries: list) -> np.ndarray:
    """(K, 4) TP/FP/FN/TN per sorted country for one arm."""
    phase = arm_columns(arm)["phase"]
    rows = []
    for c in countries:
        part = frame[frame["country_key"] == c]
        cnt = metrics.crisis_counts(part["phase_truth"].to_numpy(), part[phase].to_numpy())
        rows.append([cnt["tp"], cnt["fp"], cnt["fn"], cnt["tn"]])
    return np.array(rows, dtype=np.int64).reshape(len(countries), 4)


def _f1_rows(counts: np.ndarray) -> np.ndarray:
    tp, fp, fn = counts[:, 0], counts[:, 1], counts[:, 2]
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)


def bootstrap_delta(frame: pd.DataFrame, arm_a: str, arm_b: str) -> tuple[dict, pd.DataFrame | None]:
    """R49 paired country-cluster bootstrap of crisis-F1(arm_a) - crisis-F1(arm_b)."""
    countries = sorted(frame["country_key"].unique().tolist())
    k = len(countries)
    record = {"countries": countries, "K": k, "draws": DRAWS, "seed": SEED, "rng": "numpy.random.default_rng",
              "percentile_method": "linear", "percentiles": list(PERCENTILES)}
    point_a = metrics.exact_f1(metrics.crisis_counts(frame["phase_truth"], frame[arm_columns(arm_a)["phase"]]))
    point_b = metrics.exact_f1(metrics.crisis_counts(frame["phase_truth"], frame[arm_columns(arm_b)["phase"]]))
    point = None if point_a is None or point_b is None else float(point_a - point_b)
    record["point_delta"] = point
    if k == 0:
        return {**record, "interval": None, "na_reason": "empty cohort"}, None
    ca, cb = country_counts(frame, arm_a, countries), country_counts(frame, arm_b, countries)
    rng = np.random.default_rng(SEED)
    mult = np.zeros((DRAWS, k), dtype=np.int64)
    for d in range(DRAWS):
        mult[d] = np.bincount(rng.integers(0, k, size=k), minlength=k)
    fa, fb = _f1_rows(mult @ ca), _f1_rows(mult @ cb)
    delta = fa - fb
    draws = pd.DataFrame(mult, columns=[f"m::{c}" for c in countries])
    draws.insert(0, "delta", delta)
    draws.insert(0, f"f1_{arm_b}", fb)
    draws.insert(0, f"f1_{arm_a}", fa)
    draws.insert(0, "draw", np.arange(DRAWS))
    defined = int(np.isfinite(delta).sum())
    record.update(defined_draws=defined, undefined_draws=DRAWS - defined,
                  country_counts={arm_a: ca.tolist(), arm_b: cb.tolist()})
    reasons = []
    if k < 2:
        reasons.append("K < 2")
    if point is None:
        reasons.append("point delta undefined")
    if defined < DRAWS:
        reasons.append(f"{DRAWS - defined} undefined replicate deltas")
    if reasons:
        return {**record, "interval": None, "na_reason": "; ".join(reasons)}, draws
    lo, hi = np.percentile(delta, PERCENTILES, method="linear")
    return {**record, "interval": [float(lo), float(hi)], "na_reason": ""}, draws


def _f1_with_reason(frame: pd.DataFrame, arm: str) -> tuple[float | None, str]:
    if len(frame) == 0:
        return None, "no keys"
    b = panel(frame, arm)["binary"]
    return b["f1"], b["na_reasons"].get("f1", "")


def _delta(a: float | None, b: float | None, ra: str, rb: str) -> tuple[float | None, str]:
    if a is None or b is None:
        return None, "; ".join(r for r in (ra, rb) if r) or "undefined arm score"
    return a - b, ""


def diagnostics(frame: pd.DataFrame, by: str, scheduled=None) -> pd.DataFrame:
    """Descriptive per-country / per-month crisis-F1 points, paired deltas, NA reasons, coverage.

    ``scheduled`` (month table) adds every scheduled month, including
    ``no_valid_target`` months as n = 0 rows (R47). No intervals here (R49).
    """
    values = list(frame[by].unique())
    if scheduled is not None:
        values = sorted(set(values) | set(scheduled))
    rows = []
    for value in sorted(values):
        part = frame[frame[by] == value]
        paired = part[part["persistence_available"] == 1]
        row = {by: value, "keys": int(len(part)), "persistence_keys": int(len(paired)),
               "persistence_coverage": (len(paired) / len(part)) if len(part) else None}
        g, rg = _f1_with_reason(part, "geo")
        p, rp = _f1_with_reason(part, "pool")
        d, rd = _delta(g, p, rg, rp)
        gp, rgp = _f1_with_reason(paired, "geo")
        sp, rsp = _f1_with_reason(paired, "persistence")
        dp, rdp = _delta(gp, sp, rgp, rsp)
        row.update({"geo_f1": g, "pool_f1": p, "delta_geo_minus_pool_f1": d, "na_reason_E_all": rd,
                    "geo_f1_paired": gp, "persistence_f1_paired": sp, "delta_geo_minus_persistence_f1": dp,
                    "na_reason_E_persist": rdp})
        if len(part) == 0:
            row["status"] = "no_valid_target"
        rows.append(row)
    return pd.DataFrame(rows)


def cohort_entry(frame: pd.DataFrame, arms: tuple[str, str]) -> dict:
    """Full panels for both arms and every delta; an empty cohort is explicit NA."""
    a, b = arms
    if len(frame) == 0:
        return {"n": 0, "status": "empty_cohort", a: None, b: None,
                f"delta_{a}_minus_{b}": None, "na_reason": "no keys in this cohort"}
    pa, pb = panel(frame, a), panel(frame, b)
    return {"n": int(len(frame)), "status": "scored", a: pa, b: pb, f"delta_{a}_minus_{b}": deltas(pa, pb)}


def route_coverage(e_all: pd.DataFrame, fmap: pd.DataFrame, frozen: dict, gates: list[dict]) -> dict:
    """R36 coverage with explicit denominators (learned map, local adoption, global fallback)."""
    learned = set(fmap["admin_code"].astype(int))
    areas = set(e_all["admin_code"].astype(int))
    route = e_all["route"].astype(str)
    fallback = route[route.str.startswith("global_fallback")]
    return {
        "map": {"accepted_split": bool(frozen["accepted_split"]), "terminal_regions": int(frozen["terminal_regions"]),
                "learned_map_areas": len(learned)},
        "areas": {"denominator_cohort_areas": len(areas), "in_learned_map": len(areas & learned),
                  "unmapped": len(areas - learned),
                  "ever_local_routed": int(e_all.loc[route == "local", "admin_code"].nunique())},
        "rows": {"denominator_cohort_rows": int(len(e_all)), "local": int((route == "local").sum()),
                 "global_fallback": int(len(fallback)),
                 "global_fallback_by_reason": {k: int(v) for k, v in fallback.value_counts().sort_index().items()},
                 "global_only_no_accepted_split": int((route == "global_only_no_accepted_split").sum()),
                 "unmapped_area_global": int((route == "unmapped_area_global").sum())},
        "gate_region_folds": {"denominator_decisions": len(gates),
                              "enabled": sum(1 for g in gates if g.get("enabled")),
                              "adopted_local": sum(1 for g in gates if g.get("route") == "local"),
                              "regions_ever_adopted": len({g["region"] for g in gates if g.get("route") == "local"})},
    }


def run_report(run_dir: Path) -> dict:
    """Report for all H; any exception leaves a durable INCOMPLETE record (R41)."""
    context: dict = {"stage": "report"}
    try:
        return _run_report(run_dir, context)
    except Exception as error:
        record_incomplete(run_dir, "report", context, error)
        raise


def _run_report(run_dir: Path, context: dict) -> dict:
    stage3_dir = run_dir / "stage3"
    s3 = json.loads((stage3_dir / "stage3-summary.json").read_text(encoding="utf-8"))
    out = run_dir / "report"
    out.mkdir()
    report = {"stage": "P5-report", "stage3_summary_sha256": sha256_file(stage3_dir / "stage3-summary.json"),
              "interpretation": ("pointwise descriptive 95% country-cluster bootstrap intervals conditional on the "
                                 "saved predictions and observed cohort; no training/map-selection, shared-shock "
                                 "or future-year uncertainty; no multiplicity adjustment"),
              "horizons": {}}
    for h, info in s3["horizons"].items():
        context.update(H=int(h))
        path = stage3_dir / f"h{int(h):02d}" / "predictions.csv.gz"
        if sha256_file(path) != info["predictions_sha256"]:
            raise TechnicalError(f"H{h} predictions do not match the Stage3 summary digest")
        pred = pd.read_csv(path)
        if pred.duplicated(["admin_code", "target_ord", "horizon_months"]).any():
            raise TechnicalError(f"H{h}: duplicated prediction key")
        hrep = {}
        fmap = pd.read_csv(run_dir / "stage1" / f"frozen_map_h{int(h):02d}.csv", dtype={"node_id": str})
        frozen = json.loads((run_dir / "stage1" / f"frozen_h{int(h):02d}.json").read_text(encoding="utf-8"))
        fold_ledger = pd.read_csv(stage3_dir / f"h{int(h):02d}" / "fold_ledger.csv")
        gate_path = stage3_dir / f"h{int(h):02d}" / "gate_decisions.jsonl"
        gate_all = [json.loads(x) for x in gate_path.read_text(encoding="utf-8").splitlines()] if gate_path.is_file() else []
        for period in ("main", "supplementary"):
            context.update(period=period)
            e_all = pred[pred["period"] == period]
            e_persist = e_all[e_all["persistence_available"] == 1]
            folds = fold_ledger[fold_ledger["period"] == period]
            fold_ids = set(folds["fold_id"])
            gates = [g for g in gate_all if g["fold_id"] in fold_ids]
            entry = {
                "coverage": {"scheduled_folds": int(len(folds)),
                             "scored_folds": int((folds["status"] == "scored").sum()),
                             "no_valid_target_folds": int((folds["status"] == "no_valid_target").sum()),
                             "E_all_keys": int(len(e_all)), "E_persist_keys": int(len(e_persist)),
                             "persistence_coverage": (len(e_persist) / len(e_all)) if len(e_all) else None,
                             "countries": int(e_all["country_key"].nunique())},
                "routes": route_coverage(e_all, fmap, frozen, gates),
                "E_all": cohort_entry(e_all, ("geo", "pool")),
                "E_persist": cohort_entry(e_persist, ("geo", "persistence")),
            }
            if period == "main":
                for name, cohort, other in (("geo_vs_pool_E_all", e_all, "pool"),
                                            ("geo_vs_persistence_E_persist", e_persist, "persistence")):
                    record, draws = bootstrap_delta(cohort, "geo", other)
                    if draws is not None:
                        dpath = out / f"bootstrap_h{int(h):02d}_{name}.csv.gz"
                        draws.to_csv(dpath, index=False, compression=GZ)
                        record["draws_sha256"] = sha256_file(dpath)
                    entry.setdefault("bootstrap", {})[name] = record
            months = [f"{o // 12:04d}-{o % 12 + 1:02d}" for o in sorted(folds["target_ord"].unique())]
            diagnostics(e_all, "country_key").to_csv(out / f"diag_h{int(h):02d}_{period}_country_key.csv", index=False)
            diagnostics(e_all, "target_month", scheduled=months).to_csv(
                out / f"diag_h{int(h):02d}_{period}_target_month.csv", index=False)
            hrep[period] = entry
        hrep["predictions_sha256"] = info["predictions_sha256"]
        report["horizons"][h] = hrep
    digest = write_json(out / "report.json", report)
    return {**report, "report_sha256": digest}
