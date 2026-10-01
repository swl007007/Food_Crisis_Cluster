"""Final keyed comparisons, the D3 decision and the shared country bootstrap.

python scripts/report_fourclass.py --run-dir RUN

Never fits a model. Accepts every final fold of the four XGB arms through the
acceptance chain, joins each arm 1:1 to the prepared truth keys (a missing prediction
is an incomplete run, never a smaller cohort) and writes ``RUN/report/``.

Cohorts (inherited D13), identical keys within a cohort:
  main_h4 / main_h8   truth + persistence + expert   (D3 cohort and legal expert contrast)
  supp_h4 / supp_h8   truth + persistence
  main_h12            truth + persistence             (no H12 expert proxy)

Arms: main = xgbmap_shared (the single pre-declared main candidate), pooled (P-XGB),
rfmap_shared, rfmap_independent, persistence, expert; historical references
v7_partitioned_rf / v7_pooled_rf (35-month RF, committed v7 predictions) are reported,
never treated as matched-window backend controls.

D3 (per H, on main_h{H}): delta = F1(main) - F1(persistence) > 0 AND the 95% country-
block bootstrap lower bound > 0. Seed 42, 2,000 accepted draws, at most 20,000
attempts, linear 2.5/97.5 percentiles, draws shared by every cohort/arm/H.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from src.experiment import plan  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.utils import acceptance as acc  # noqa: E402
from src.utils.run_identity import file_sha256  # noqa: E402

SEED = 42
DRAWS = 2000
MAX_ATTEMPTS = 20000
HORIZONS = plan.HORIZONS
KEY = ["area", "target_month", "horizon"]
XGB_ARMS = ("xgbmap_shared", "pooled", "rfmap_shared", "rfmap_independent")
ARM_COLUMN = {"main": "y_pred_xgbmap_shared", "pooled": "y_pred_pooled",
              "rfmap_shared": "y_pred_rfmap_shared", "rfmap_independent": "y_pred_rfmap_independent",
              "expert": "expert_code", "persistence": "persistence_code",
              "v7_partitioned_rf": "y_pred_v7_partitioned_rf", "v7_pooled_rf": "y_pred_v7_pooled_rf",
              # inherited names, kept for the shared helpers
              "partitioned": "y_pred_xgbmap_shared"}
#: (arm, baseline) pairs reported as contrasts; the first per cohort is the D3 contrast.
CONTRASTS = (("main", "persistence"), ("main", "pooled"), ("main", "expert"), ("main", "rfmap_shared"),
             ("main", "rfmap_independent"), ("rfmap_shared", "rfmap_independent"), ("rfmap_shared", "pooled"),
             ("pooled", "persistence"), ("main", "v7_partitioned_rf"))


def final_predictions(run: Path) -> pd.DataFrame:
    """Every scheduled final fold of every arm, accepted, stacked per arm."""
    acc.accept_prepared(run)  # schedule and baseline cohorts only through their recorded hashes
    frozen = acc.accept_record(run / "frozen", "frozen.json")
    frozen_sha = file_sha256(run / "frozen" / "frozen.json")
    frames, problems, incomplete = [], [], []
    for fold in acc.schedule(run)["stage3"]:
        if fold["status"] != "scheduled":
            continue
        for arm in XGB_ARMS:
            base = run / "final" / f"h{fold['horizon']}" / fold["target_month"] / arm
            try:
                record = acc.accept_fold(base, {"phase": "final", "horizon": fold["horizon"],
                                                "target_month": fold["target_month"],
                                                "frozen_sha256": frozen_sha,
                                                "g_config": frozen["g_selection"][str(fold["horizon"])]})
            except Exception as exc:  # missing, partial, stale or foreign fold
                problems.append(f"{base}: {exc}")
                continue
            if record.get("status") == "incomplete_coverage_gate":
                incomplete.append(f"h{fold['horizon']} {fold['target_month']} {arm}: coverage gate")
                continue
            if record["arm"] != {"xgbmap_shared": "shared", "rfmap_shared": "shared",
                                 "rfmap_independent": "independent", "pooled": "pooled"}[arm]:
                problems.append(f"{base}: arm {record['arm']!r}")
            preds = pd.read_csv(base / "predictions.csv.gz", float_precision="round_trip")
            frames.append(preds.assign(arm=arm))
    if problems:
        raise RuntimeError(f"final evaluation is not accepted: {problems[:5]}")
    if incomplete:
        raise IncompleteEvaluation(incomplete)
    return pd.concat(frames, ignore_index=True)


class IncompleteEvaluation(RuntimeError):
    """A planned arm-fold was blocked by the inherited coverage gate: report incomplete,
    never a smaller key set."""


def v7_reference() -> pd.DataFrame:
    frames = []
    for h in HORIZONS:
        p = pd.read_csv(acc.V7_RUN / "stage3" / f"h{h}" / "predictions.csv.gz", float_precision="round_trip")
        frames.append(p[["area", "target_month", "horizon", "y_pred_partitioned_code", "y_pred_pooled_code"]])
    return pd.concat(frames).rename(columns={"y_pred_partitioned_code": "y_pred_v7_partitioned_rf",
                                             "y_pred_pooled_code": "y_pred_v7_pooled_rf"})


def load_keyed(run: Path) -> pd.DataFrame:
    acc.accept_prepared(run)
    base = pd.read_csv(run / "prepared" / "ledgers" / "baselines.csv", float_precision="round_trip", low_memory=False)
    base["target_month"] = base["target_label"]
    preds = final_predictions(run)
    wide = base.copy()
    for arm in XGB_ARMS:
        p = preds[preds["arm"] == arm]
        if p.duplicated(KEY).any():
            raise RuntimeError(f"{arm}: duplicate prediction keys")
        p = p[KEY + ["y_true_code", "y_pred_code", "route", "cluster_id"]].rename(
            columns={"y_true_code": f"y_true_{arm}", "y_pred_code": f"y_pred_{arm}", "route": f"route_{arm}",
                     "cluster_id": f"cluster_{arm}"})
        merged = wide.merge(p, on=KEY, how="outer", indicator=True, validate="one_to_one")
        counts = merged["_merge"].value_counts().to_dict()
        if counts.get("left_only", 0) or counts.get("right_only", 0):
            raise RuntimeError(f"{arm}: prediction keys differ from the truth keys {counts}")
        if not (merged["truth_code"] == merged[f"y_true_{arm}"]).all():
            raise RuntimeError(f"{arm}: truth differs from the baseline ledger")
        wide = merged.drop(columns=["_merge", f"y_true_{arm}"])
    v7 = v7_reference()
    wide = wide.merge(v7, on=KEY, how="left", validate="one_to_one")
    v7_truth = []
    for h in HORIZONS:
        p = pd.read_csv(acc.V7_RUN / "stage3" / f"h{h}" / "predictions.csv.gz", float_precision="round_trip")
        v7_truth.append(p[KEY + ["y_true_code"]])
    check = wide.merge(pd.concat(v7_truth).rename(columns={"y_true_code": "v7_truth"}), on=KEY, how="left")
    if wide["y_pred_v7_partitioned_rf"].isna().any() or not (check["v7_truth"] == check["truth_code"]).all():
        raise RuntimeError("the committed v7 reference does not cover every truth key with the same truth")
    return wide


def cohorts(frame: pd.DataFrame) -> dict:
    out = {}
    for horizon in HORIZONS:
        rows = frame[frame["horizon"] == horizon]
        has_p = rows["persistence_code"].notna()
        xgb = ("main", "pooled", "rfmap_shared", "rfmap_independent")
        v7 = ("v7_partitioned_rf", "v7_pooled_rf") if "y_pred_v7_partitioned_rf" in rows else ()
        if horizon in (4, 8):
            has_e = rows["expert_code"].notna()
            out[f"main_h{horizon}"] = (rows[has_p & has_e], xgb + ("expert", "persistence") + v7)
            out[f"supp_h{horizon}"] = (rows[has_p], xgb + ("persistence",) + v7)
        else:
            out[f"main_h{horizon}"] = (rows[has_p], xgb + ("persistence",) + v7)
    return out


def country_matrices(rows: pd.DataFrame, arms, countries) -> np.ndarray:
    """(n_countries, n_arms, 4, 4) confusion counts, countries on the shared axis."""
    index = {c: i for i, c in enumerate(countries)}
    k = fourclass.N_CLASSES
    out = np.zeros((len(countries), len(arms), k, k))
    pos = rows["country"].map(index).to_numpy()
    truth = rows["truth_code"].to_numpy(dtype=int)
    for a, arm in enumerate(arms):
        values = rows[ARM_COLUMN[arm]].to_numpy(dtype=float)
        if np.isnan(values).any():
            raise RuntimeError(f"{arm}: a cohort key has no prediction")
        cells = (pos * k + truth) * k + values.astype(int)
        out[:, a] = np.bincount(cells, minlength=len(countries) * k * k).reshape(len(countries), k, k)
    return out


def macro_from_matrices(mats: np.ndarray) -> np.ndarray:
    """Fixed-four macro F1 over the last two axes (..., 4, 4)."""
    tp = np.diagonal(mats, axis1=-2, axis2=-1)
    fp = mats.sum(axis=-2) - tp
    fn = mats.sum(axis=-1) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return f1.mean(axis=-1)


def bootstrap(cohort_rows: dict, countries: list):
    rng = np.random.default_rng(SEED)
    per = {name: (country_matrices(rows, arms, countries), arms) for name, (rows, arms) in cohort_rows.items()}
    point = {name: macro_from_matrices(m.sum(axis=0)) for name, (m, _) in per.items()}
    accepted, multiplicities, rejected = [], [], []
    attempts = 0
    while len(accepted) < DRAWS and attempts < MAX_ATTEMPTS:
        attempts += 1
        draw = rng.integers(0, len(countries), size=len(countries))
        mult = np.bincount(draw, minlength=len(countries)).astype(float)
        empty = [name for name, (m, _) in per.items() if np.tensordot(mult, m[:, 0], axes=1).sum() == 0]
        if empty:
            rejected.append({"attempt": attempts, "reason": "empty required cohort", "cohorts": empty,
                             "multiplicities": mult.astype(int).tolist()})
            continue
        stats = {name: macro_from_matrices(np.tensordot(mult, m, axes=1)) for name, (m, _) in per.items()}
        accepted.append(stats)
        multiplicities.append(mult.astype(int))
    return per, point, accepted, np.array(multiplicities), rejected, attempts


def contrast_rows(per, point, accepted, metrics, complete):
    rows = []
    for name, (_, arms) in per.items():
        for arm, baseline in CONTRASTS:
            if arm not in arms or baseline not in arms:
                continue
            a, b = arms.index(arm), arms.index(baseline)
            deltas = np.array([s[name][a] - s[name][b] for s in accepted])
            lo, hi = (np.percentile(deltas, [2.5, 97.5], method="linear") if complete else (None, None))
            rows.append({"cohort": name, "contrast": f"{arm} - {baseline}", "n": metrics[name]["n"],
                         "arm_macro_f1": float(point[name][a]), "baseline_macro_f1": float(point[name][b]),
                         "delta": float(point[name][a] - point[name][b]),
                         "ci95_low": float(lo) if complete else None, "ci95_high": float(hi) if complete else None,
                         "interval_excludes_zero": bool(complete and (lo > 0 or hi < 0)),
                         "draw_mean_delta": float(deltas.mean()) if len(deltas) else None})
    return rows


def d3_decision(contrasts: list, complete: bool) -> dict:
    out = {}
    for h in HORIZONS:
        row = next(r for r in contrasts if r["cohort"] == f"main_h{h}" and r["contrast"] == "main - persistence")
        if not complete:
            status = "incomplete"
        else:
            status = "pass" if row["delta"] > 0 and row["ci95_low"] > 0 else "fail"
        out[str(h)] = {"delta": row["delta"], "ci95": [row["ci95_low"], row["ci95_high"]], "status": status}
    statuses = [v["status"] for v in out.values()]
    overall = "incomplete" if "incomplete" in statuses else ("pass" if all(s == "pass" for s in statuses) else "fail")
    return {"per_horizon": out, "overall": overall,
            "rule": "per H on main_h{H}: delta > 0 and 95% country-block CI lower bound > 0 (D2/D3/D17)",
            "interpretation": "three marginal per-H intervals, not a simultaneous 95% statement"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / "report"
    if out.exists():
        raise FileExistsError(f"{out} exists")
    try:
        frame = load_keyed(run)
    except IncompleteEvaluation as exc:
        out.mkdir(parents=True)
        (out / "incomplete.json").write_text(json.dumps({
            "scientific_target_D3": {"overall": "incomplete"}, "blocked_arm_folds": exc.args[0],
            "rule": "a coverage-gate failure makes the planned evaluation incomplete (plan section 6)"},
            indent=2), encoding="utf-8")
        raise SystemExit(f"final evaluation incomplete: {exc.args[0][:5]}")
    groups = cohorts(frame)
    countries = sorted(frame["country"].unique().tolist())
    metrics = {}
    for name, (rows, arms) in groups.items():
        metrics[name] = {"n": int(len(rows)), "countries": int(rows["country"].nunique()),
                         "arms": {arm: fourclass.summary(rows["truth_code"].to_numpy(dtype=int),
                                                         rows[ARM_COLUMN[arm]].to_numpy(dtype=float).astype(int))
                                  for arm in arms}}
    per, point, accepted, mults, rejected, attempts = bootstrap(groups, countries)
    complete = len(accepted) == DRAWS
    contrasts = contrast_rows(per, point, accepted, metrics, complete)
    decision = d3_decision(contrasts, complete)

    out.mkdir(parents=True)
    arm_rows = [{"cohort": n, "arm": a, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                 "category_step_mae": m["category_step_mae"], "n": m["n"],
                 **{f"f1_{c}": m["per_class"][c]["f1"] for c in fourclass.CLASS_LABELS},
                 **{f"recall_{c}": m["per_class"][c]["recall"] for c in fourclass.CLASS_LABELS},
                 **{f"support_{c}": m["per_class"][c]["support"] for c in fourclass.CLASS_LABELS}}
                for n, v in metrics.items() for a, m in v["arms"].items()]
    pd.DataFrame(arm_rows).to_csv(out / "arm_metrics.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "contrasts.csv", index=False)
    descriptive = []
    for name, (rows, arms) in groups.items():
        keyed = rows.assign(year=rows["target_month"].str[:4])
        for by in ("country", "year"):
            for value, part in keyed.groupby(by):
                entry = {"cohort": name, "by": by, "value": value, "n": int(len(part))}
                for arm in arms:
                    entry[f"macro_f1_{arm}"] = fourclass.macro_f1(
                        part["truth_code"].to_numpy(dtype=int), part[ARM_COLUMN[arm]].to_numpy(dtype=float).astype(int))
                descriptive.append(entry)
    pd.DataFrame(descriptive).to_csv(out / "descriptive_by_country_year.csv", index=False)
    draws = pd.DataFrame(mults, columns=countries)
    draws.insert(0, "draw", np.arange(len(draws)))
    for name, (_, arms) in per.items():
        for a, arm in enumerate(arms):
            draws[f"{name}:{arm}"] = [s[name][a] for s in accepted]
    with gzip.open(out / "bootstrap_draws.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        draws.to_csv(handle, index=False, float_format="%.17g")
    keyed_cols = KEY + ["country", "truth_code", "persistence_code", "expert_code"] + \
        [c for arm in XGB_ARMS for c in (f"y_pred_{arm}", f"route_{arm}", f"cluster_{arm}")] + \
        ["y_pred_v7_partitioned_rf", "y_pred_v7_pooled_rf"]
    with gzip.open(out / "keyed_evaluation.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        frame[keyed_cols].sort_values(KEY).to_csv(handle, index=False)
    routes = {f"h{h}": {arm: frame.loc[frame["horizon"] == h, f"route_{arm}"].value_counts().to_dict()
                        for arm in XGB_ARMS} for h in HORIZONS}
    report = {
        "scientific_target_D3": decision,
        "cohort_rule": "inherited D13 cohorts; identical keys within a cohort; failures cannot shrink a cohort",
        "metrics": metrics, "contrasts": contrasts, "routes": routes,
        "frozen_record_sha256": file_sha256(run / "frozen" / "frozen.json"),
        "bootstrap": {"seed": SEED, "draws_requested": DRAWS, "draws_accepted": len(accepted),
                      "attempts": attempts, "rejected": rejected, "complete": complete,
                      "countries": countries, "shared_across": "all cohorts, arms, contrasts and horizons",
                      "method": ("resample the sorted country union with replacement; weight each country's "
                                 "keyed confusion counts by its multiplicity; fixed-four macro F1; linear "
                                 "2.5/97.5 percentiles; conditioned on fitted predictions, no refit")},
        "expert_target": "H4/H8 optimisation target; point gain and interval reported, no hard CI gate (D24)",
        "references": "v7 RF arms use a 35-month window and their own maps: historical reference only",
        "disclosures": ["development G and scheme selection bias (D24)", "conditional historical-map gate bias (D18)",
                        "E1/E2 reuse of random validation rows (D10/D20)",
                        "2021-2024 baselines had been inspected before this study: retrospective evaluation (D16)"],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(decision, indent=1))
    print(pd.DataFrame(contrasts)[["cohort", "contrast", "delta", "ci95_low", "ci95_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
