"""P3 reporting from saved keyed predictions only (PRD R21-R22; design section 8).

Metric panel, cohort and bootstrap helpers are attributed copies of
ipcch_geoxgb/report.py at 6798df2 (``panel``, ``flat``, ``deltas``,
``country_counts``, ``_f1_rows``, ``bootstrap_delta``; 2000 draws,
default_rng(42) reset per cohort, every draw kept, interval only when K >= 2,
the point delta is defined and all draws are finite). Arms: B ``base``,
P ``pool``, G ``geo``, ungated L ``local`` (eligible keys only), original
GeoXGB ``xgbgeo`` and matched pooled XGB ``xgbpool`` on identical keys, and
persistence on E_persist. Nothing is fitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_mlp import metrics, sources
from ipcch_mlp.artifacts import sha256_file, write_json
from ipcch_mlp.errors import TechnicalError

DRAWS, SEED, PERCENTILES = 2000, 42, (2.5, 97.5)
GZ = {"method": "gzip", "mtime": 0}
KEY = ["admin_code", "target_ord", "horizon_months"]
SCALARS = (("binary", "accuracy"), ("binary", "precision"), ("binary", "recall"), ("binary", "f1"), ("binary", "f2"),
           ("four_class", "accuracy"), ("four_class", "macro_f1"))
PREDECLARED_ELIGIBLE = {1: 3211, 3: 4997, 6: 3423, 12: 845}


# ------------------------------------------------------------ copied helpers

def arm_columns(arm: str) -> dict:
    if arm == "persistence":
        return {"phase": "persistence_phase", "q3_star": "persistence_q3", "q3_raw": "persistence_q3"}
    return {"phase": f"{arm}_phase", "q3_star": f"{arm}_q3_star", "q3_raw": f"{arm}_q3_raw"}


def panel(frame: pd.DataFrame, arm: str) -> dict:
    cols = arm_columns(arm)
    return metrics.metric_panel(frame["phase_truth"].to_numpy(), frame[cols["phase"]].to_numpy(),
                                frame["q3_truth"].to_numpy(), frame[cols["q3_star"]].to_numpy(),
                                frame[cols["q3_raw"]].to_numpy())


def flat(p: dict) -> dict:
    out = {f"{g}.{n}": p[g][n] for g, n in SCALARS}
    out["q3_r2_projected"] = p.get("q3_r2_projected")
    out["q3_r2_raw"] = p.get("q3_r2_raw")
    return out


def deltas(a: dict, b: dict) -> dict:
    fa, fb = flat(a), flat(b)
    return {k: (None if fa[k] is None or fb[k] is None else fa[k] - fb[k]) for k in fa}


def country_counts(frame: pd.DataFrame, arm: str, countries: list) -> np.ndarray:
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


def bootstrap_delta(frame: pd.DataFrame, arm_a: str, arm_b: str) -> dict:
    countries = sorted(frame["country_key"].unique().tolist())
    k = len(countries)
    record = {"K": k, "draws": DRAWS, "seed": SEED, "rng": "numpy.random.default_rng", "percentiles": list(PERCENTILES)}
    pa = metrics.exact_f1(metrics.crisis_counts(frame["phase_truth"], frame[arm_columns(arm_a)["phase"]]))
    pb = metrics.exact_f1(metrics.crisis_counts(frame["phase_truth"], frame[arm_columns(arm_b)["phase"]]))
    point = None if pa is None or pb is None else float(pa - pb)
    record["point_delta"] = point
    if k == 0:
        return {**record, "interval": None, "na_reason": "empty cohort"}
    ca, cb = country_counts(frame, arm_a, countries), country_counts(frame, arm_b, countries)
    rng = np.random.default_rng(SEED)
    mult = np.zeros((DRAWS, k), dtype=np.int64)
    for d in range(DRAWS):
        mult[d] = np.bincount(rng.integers(0, k, size=k), minlength=k)
    delta = _f1_rows(mult @ ca) - _f1_rows(mult @ cb)
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
        return {**record, "interval": None, "na_reason": "; ".join(reasons)}
    lo, hi = np.percentile(delta, PERCENTILES, method="linear")
    return {**record, "interval": [float(lo), float(hi)], "na_reason": ""}


def cohort(frame: pd.DataFrame, arms: tuple) -> dict:
    if len(frame) == 0:
        return {"n": 0, "status": "empty_cohort"}
    panels = {a: panel(frame, a) for a in arms}
    return {"n": int(len(frame)), "status": "scored", "panels": panels}


# ------------------------------------------------------------ MLP report

def read_predictions(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path, float_precision="round_trip",
                       dtype={"region": str, "lres_provider": str, "base_provider": str, "pres_provider": str})
    for col in ("region", "lres_provider"):
        pred[col] = pred[col].fillna("")
    return pred


def attach_xgb(pred: pd.DataFrame, p6: pd.DataFrame) -> pd.DataFrame:
    cols = KEY + ["geo_phase", "pool_phase", "geo_q3_star", "geo_q3_raw", "pool_q3_star", "pool_q3_raw", "phase_truth"]
    x = p6[cols].rename(columns={"geo_phase": "xgbgeo_phase", "pool_phase": "xgbpool_phase",
                                 "geo_q3_star": "xgbgeo_q3_star", "geo_q3_raw": "xgbgeo_q3_raw",
                                 "pool_q3_star": "xgbpool_q3_star", "pool_q3_raw": "xgbpool_q3_raw",
                                 "phase_truth": "xgb_phase_truth"})
    m = pred.merge(x, on=KEY, how="left", validate="one_to_one", indicator=True)
    if (m["_merge"] != "both").any() or len(m) != len(pred) or len(pred) != len(p6):
        raise TechnicalError("MLP and original XGB prediction keys differ")
    if not (m["phase_truth"] == m["xgb_phase_truth"]).all():
        raise TechnicalError("truth differs between MLP and original XGB predictions")
    return m.drop(columns=["_merge", "xgb_phase_truth"])


def gate_category(route: pd.Series) -> pd.Series:
    r = route.astype(str)
    return np.select([r == "local", r.str.startswith("pool_fallback:gate_support"),
                      r == "pool_fallback:current_fit_support", r.str.startswith("pool_fallback:"),
                      r == "unmapped_area_pool"],
                     ["adopted", "historical_support_rejected", "current_fit_support", "gain_rejected", "unmapped"],
                     default="other")


def horizon_report(pred: pd.DataFrame, gates: list, h: int, period: str) -> dict:
    e_all = pred[pred["period"] == period]
    e_persist = e_all[e_all["persistence_available"] == 1]
    cat = gate_category(e_all["route"])
    entry = {"coverage": {"E_all_keys": int(len(e_all)), "E_persist_keys": int(len(e_persist)),
                          "countries": int(e_all["country_key"].nunique()),
                          "routes": {k: int(v) for k, v in pd.Series(cat).value_counts().sort_index().items()},
                          "local_eligible_keys": int(e_all["local_eligible"].sum())}}
    if len(e_all) == 0:
        return {**entry, "status": "empty"}
    fold_ids = set(e_all["fold_id"])
    support_ok = {(g["fold_id"], g["region"]) for g in gates if g.get("historical_support") and g["fold_id"] in fold_ids}
    eligible = np.array([(f, r) in support_ok for f, r in zip(e_all["fold_id"], e_all["region"])])
    entry["coverage"]["historical_support_eligible_keys"] = int(eligible.sum())
    entry["coverage"]["eligible_and_current_supported"] = int((eligible & (e_all["local_eligible"] == 1).to_numpy()).sum())
    entry["coverage"]["adopted_keys"] = int((e_all["route"] == "local").sum())
    if period == "main" and int(eligible.sum()) != PREDECLARED_ELIGIBLE[h]:
        raise TechnicalError(f"H{h}: historical-support eligible keys {int(eligible.sum())} != predeclared "
                             f"{PREDECLARED_ELIGIBLE[h]}")
    arms_all = ("base", "pool", "geo", "xgbgeo", "xgbpool")
    entry["E_all"] = cohort(e_all, arms_all)
    pa = entry["E_all"]["panels"]
    entry["E_all"]["deltas"] = {"geo_minus_pool": deltas(pa["geo"], pa["pool"]),
                                "pool_minus_base": deltas(pa["pool"], pa["base"]),
                                "base_minus_xgbpool": deltas(pa["base"], pa["xgbpool"]),
                                "geo_minus_xgbgeo": deltas(pa["geo"], pa["xgbgeo"])}
    entry["E_persist"] = cohort(e_persist, arms_all + ("persistence",))
    if len(e_persist):
        pp = entry["E_persist"]["panels"]
        entry["E_persist"]["deltas"] = {f"{a}_minus_persistence": deltas(pp[a], pp["persistence"])
                                        for a in ("base", "pool", "geo")}
    flips = (e_all["geo_phase"] >= 3) != (e_all["pool_phase"] >= 3)
    entry["label_flips_geo_vs_pool"] = {"crisis": int(flips.sum()),
                                        "phase": int((e_all["geo_phase"] != e_all["pool_phase"]).sum())}
    for arm in ("base", "pool", "geo"):
        for kind in ("raw", "star"):
            entry.setdefault("q3_mse", {})[f"{arm}_{kind}"] = float(np.mean((e_all["q3_truth"] - e_all[f"{arm}_q3_{kind}"]) ** 2))
    # ungated regional diagnostic on exactly the eligible keys (point estimates only)
    diag = e_all[e_all["local_eligible"] == 1]
    dcat = gate_category(diag["route"]) if len(diag) else pd.Series(dtype=str)
    diag_entry = {"keys": int(len(diag)), "by_gate": {}}
    if len(diag):
        dp = cohort(diag, ("local", "pool", "base", "geo", "xgbgeo", "xgbpool"))
        diag_entry["all"] = {**dp, "local_minus_pool": deltas(dp["panels"]["local"], dp["panels"]["pool"])}
        for name in sorted(set(dcat)):
            part = diag[dcat == name]
            cp = cohort(part, ("local", "pool"))
            diag_entry["by_gate"][name] = {**cp, "local_minus_pool": deltas(cp["panels"]["local"], cp["panels"]["pool"])}
    entry["ungated_local_diagnostic"] = diag_entry
    if period == "main":
        entry["bootstrap"] = {"geo_minus_pool_E_all": bootstrap_delta(e_all, "geo", "pool"),
                              "geo_minus_persistence_E_persist": bootstrap_delta(e_persist, "geo", "persistence")}
    return entry


def run_report(run_dir: Path, contract: dict) -> dict:
    out = run_dir / "report"
    out.mkdir()
    s3 = json.loads((run_dir / "stage3" / "stage3-summary.json").read_text(encoding="utf-8"))
    report = {"stage": "P3-report", "stage3_summary_sha256": sha256_file(run_dir / "stage3" / "stage3-summary.json"),
              "interpretation": ("exploratory (evaluation period already viewed); per-seed pointwise country-cluster "
                                 "bootstrap conditional on saved predictions; seeds are replicates, not independent "
                                 "datasets; G-P can change only historical-support eligible keys"),
              "replicates": {}}
    table = []
    for rep, rinfo in s3["replicates"].items():
        report["replicates"][rep] = {}
        for h in contract["horizons_months"]:
            path = run_dir / "stage3" / f"rep{rep}" / f"h{h:02d}" / "predictions.csv.gz"
            if sha256_file(path) != rinfo[str(h)]["predictions_sha256"]:
                raise TechnicalError(f"rep{rep} H{h}: predictions differ from the Stage3 summary digest")
            pred = read_predictions(path)
            pred = attach_xgb(pred, sources.load_p6_predictions(h))
            gates = [json.loads(x) for x in (path.parent / "gate_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
            hrep = {p: horizon_report(pred, gates, h, p) for p in ("main", "supplementary")}
            report["replicates"][rep][str(h)] = hrep
            for period, e in hrep.items():
                if "E_all" not in e:
                    continue
                pa = e["E_all"]["panels"]
                table.append({"replicate": int(rep), "H": h, "period": period, "n": e["E_all"]["n"],
                              **{f"f1_{a}": pa[a]["binary"]["f1"] for a in pa},
                              "geo_minus_pool_f1": e["E_all"]["deltas"]["geo_minus_pool"]["binary.f1"],
                              "adopted_keys": e["coverage"]["adopted_keys"],
                              "eligible_keys": e["coverage"]["historical_support_eligible_keys"],
                              "local_eligible_keys": e["coverage"]["local_eligible_keys"]})
    tdf = pd.DataFrame(table)
    tdf.to_csv(out / "summary_table.csv", index=False)
    seeds_summary = {}
    if len(tdf):
        for (h, period), part in tdf.groupby(["H", "period"]):
            seeds_summary[f"{period}_h{int(h):02d}"] = {
                c: {"mean": float(part[c].mean()), "min": float(part[c].min()), "max": float(part[c].max())}
                for c in part.columns if c.startswith("f1_") or c == "geo_minus_pool_f1"}
    report["seed_mean_range_descriptive"] = seeds_summary
    digest = write_json(out / "report.json", report)
    return {"report_sha256": digest, "rows": len(tdf)}
