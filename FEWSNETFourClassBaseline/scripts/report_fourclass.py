"""Keyed comparisons and the shared country bootstrap (PRD R13, R14; design "Reporting").

python scripts/report_fourclass.py --run-dir runs/<id>

Never fits a model. Reads ``prepared/ledgers/baselines.csv`` and each horizon's
Stage 3 ``predictions.csv.gz``; writes ``<run>/report/`` (refuses to overwrite).

Cohorts (D13), all on identical keys within a cohort:
  main_h4 / main_h8   truth + expert + exact-origin persistence; 4 arms
  main_h12            truth + persistence; 3 arms (no fs3 expert)
  supp_h4 / supp_h8   truth + persistence (expert not required); 3 arms
Every truth key must carry both RF predictions; a missing prediction is an
incomplete run, never a smaller cohort.
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
from src.metrics import fourclass  # noqa: E402

SEED = 42
DRAWS = 2000
MAX_ATTEMPTS = 20000
HORIZONS = (4, 8, 12)
KEY = ["area", "target_month", "horizon"]


def verify_stage3(run: Path) -> None:
    """Report only from complete Stage 3 outputs made by the current code/runtime."""
    from src.utils.run_identity import (REQUIRED_STAGE3_FOLD, check_inventory, code_identity, file_sha256,
                                        require_prepared, runtime_identity)
    require_prepared(run)
    for horizon in HORIZONS:
        out = run / "stage3" / f"h{horizon}"
        manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("code") != code_identity() or manifest.get("runtime") != runtime_identity():
            raise RuntimeError(f"h{horizon}: Stage 3 made by different code or runtime")
        if file_sha256(out / "predictions.csv.gz") != manifest["predictions_sha256"]:
            raise RuntimeError(f"h{horizon}: predictions differ from the Stage 3 record")
        if manifest.get("horizon") != horizon or not manifest.get("fold_records") or manifest.get("only_month"):
            raise RuntimeError(f"h{horizon}: Stage 3 record is for another horizon, partial or lists no folds")
        from scripts.compare_partitioned_vs_pooled_rf_k40_nc4 import reconcile_horizon
        reconcile_horizon(run, out, horizon)
        listed = set(manifest["fold_records"])
        on_disk = {p.name for p in (out / "folds").iterdir() if p.is_dir()}
        if listed != on_disk:
            raise RuntimeError(f"h{horizon}: fold records {sorted(listed ^ on_disk)[:5]} differ from fold directories")
        for month, sha in manifest["fold_records"].items():
            fold = out / "folds" / month
            if file_sha256(fold / "fold.json") != sha:
                raise RuntimeError(f"h{horizon} {month}: fold record changed")
            record = json.loads((fold / "fold.json").read_text(encoding="utf-8"))
            if record.get("target_month") != month or record.get("horizon") != horizon:
                raise RuntimeError(f"h{horizon} {month}: fold record describes another fold")
            required = REQUIRED_STAGE3_FOLD if record["status"] == "fitted" else ()
            required = list(required) + [f"models/local_{c}.pkl.xz"
                                         for c in (int(k.split('_')[1]) for k in record.get("estimators", {})
                                                   if k.startswith("local_"))]
            problems = check_inventory(fold, record.get("outputs") or {}, required)
            if problems:
                raise RuntimeError(f"h{horizon} {month}: {problems[:5]}")


def load_keyed(run: Path) -> pd.DataFrame:
    base = pd.read_csv(run / "prepared" / "ledgers" / "baselines.csv")
    base["target_month"] = base["target_label"]
    frames = []
    for horizon in HORIZONS:
        preds = pd.read_csv(run / "stage3" / f"h{horizon}" / "predictions.csv.gz")
        if preds.duplicated(KEY).any():
            raise RuntimeError(f"h{horizon}: duplicate prediction keys")
        frames.append(preds)
    preds = pd.concat(frames, ignore_index=True)
    merged = base.merge(preds, on=KEY, how="outer", indicator=True, validate="one_to_one")
    orphan = merged["_merge"].value_counts().to_dict()
    if orphan.get("left_only", 0) or orphan.get("right_only", 0):
        raise RuntimeError(f"baseline and prediction keys differ: {orphan}")
    if not (merged["truth_code"] == merged["y_true_code"]).all():
        raise RuntimeError("Stage 3 truth disagrees with the baseline ledger")
    return merged.drop(columns="_merge")


def cohorts(frame: pd.DataFrame) -> dict:
    out = {}
    for horizon in HORIZONS:
        rows = frame[frame["horizon"] == horizon]
        has_p = rows["persistence_code"].notna()
        if horizon in (4, 8):
            has_e = rows["expert_code"].notna()
            out[f"main_h{horizon}"] = (rows[has_p & has_e], ("partitioned", "pooled", "expert", "persistence"))
            out[f"supp_h{horizon}"] = (rows[has_p], ("partitioned", "pooled", "persistence"))
        else:
            out[f"main_h{horizon}"] = (rows[has_p], ("partitioned", "pooled", "persistence"))
    return out


ARM_COLUMN = {"partitioned": "y_pred_partitioned_code", "pooled": "y_pred_pooled_code",
              "expert": "expert_code", "persistence": "persistence_code"}


def coverage(frame: pd.DataFrame) -> dict:
    record = {}
    for horizon in HORIZONS:
        rows = frame[frame["horizon"] == horizon]
        has_p, has_e = rows["persistence_code"].notna(), rows["expert_code"].notna()
        record[f"h{horizon}"] = {
            "truth_keys": int(len(rows)), "with_persistence": int(has_p.sum()),
            "with_expert": int(has_e.sum()), "with_both": int((has_p & has_e).sum()),
            "excluded_no_persistence": int((~has_p).sum()),
            "excluded_no_expert_given_persistence": int((has_p & ~has_e).sum()) if horizon != 12 else None,
            "target_months": sorted(rows["target_month"].unique().tolist()),
            "countries": int(rows["country"].nunique()),
            "partitioned_routes": rows["partitioned_route"].value_counts().to_dict(),
        }
    return record


def country_matrices(rows: pd.DataFrame, arms, countries) -> np.ndarray:
    """(n_countries, n_arms, 4, 4) confusion counts, countries on the shared axis."""
    index = {c: i for i, c in enumerate(countries)}
    k = fourclass.N_CLASSES
    out = np.zeros((len(countries), len(arms), k, k))
    pos = rows["country"].map(index).to_numpy()
    truth = rows["truth_code"].to_numpy(dtype=int)
    for a, arm in enumerate(arms):
        pred = rows[ARM_COLUMN[arm]].to_numpy(dtype=float).astype(int)
        cells = (pos * k + truth) * k + pred
        out[:, a] = np.bincount(cells, minlength=len(countries) * k * k).reshape(len(countries), k, k)
    return out


def macro_from_matrices(mats: np.ndarray) -> np.ndarray:
    """Fixed-four macro F1 over the last two axes (…, 4, 4)."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / "report"
    if out.exists():
        raise FileExistsError(f"{out} exists")
    verify_stage3(run)
    frame = load_keyed(run)
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
    contrasts = []
    for name, (_, arms) in per.items():
        for b, baseline in enumerate(arms[1:], start=1):
            deltas = np.array([s[name][0] - s[name][b] for s in accepted])
            lo, hi = (np.percentile(deltas, [2.5, 97.5], method="linear") if complete else (None, None))
            contrasts.append({
                "cohort": name, "contrast": f"partitioned - {baseline}", "n": metrics[name]["n"],
                "partitioned_macro_f1": float(point[name][0]), "baseline_macro_f1": float(point[name][b]),
                "delta": float(point[name][0] - point[name][b]),
                "ci95": [float(lo), float(hi)] if complete else None,
                "interval_excludes_zero": bool(complete and (lo > 0 or hi < 0)),
                "draw_mean_delta": float(deltas.mean()) if len(deltas) else None,
            })

    out.mkdir(parents=True)
    arm_rows = [{"cohort": n, "arm": a, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                 "category_step_mae": m["category_step_mae"], "n": m["n"],
                 **{f"f1_{c}": m["per_class"][c]["f1"] for c in fourclass.CLASS_LABELS},
                 **{f"support_{c}": m["per_class"][c]["support"] for c in fourclass.CLASS_LABELS}}
                for n, v in metrics.items() for a, m in v["arms"].items()]
    pd.DataFrame(arm_rows).to_csv(out / "arm_metrics.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "contrasts.csv", index=False)

    descriptive = []
    for name, (rows, arms) in groups.items():
        for by in ("country", "year"):
            keyed = rows.assign(year=rows["target_month"].str[:4])
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
    keyed_cols = KEY + ["country", "truth_code", "persistence_code", "expert_code",
                        "y_pred_pooled_code", "y_pred_partitioned_code", "partitioned_route"]
    with gzip.open(out / "keyed_evaluation.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        frame[keyed_cols].sort_values(KEY).to_csv(handle, index=False)

    report = {
        "cohort_rule": "D13; identical keys within a cohort; RF failures cannot shrink a cohort",
        "coverage": coverage(frame), "metrics": metrics, "contrasts": contrasts,
        "bootstrap": {"seed": SEED, "draws_requested": DRAWS, "draws_accepted": len(accepted),
                      "attempts": attempts, "rejected": rejected, "complete": complete,
                      "countries": countries, "shared_across": "all cohorts, contrasts and horizons",
                      "method": ("resample the sorted country union with replacement; weight each country's "
                                 "keyed confusion counts by its multiplicity; fixed-four macro F1; linear "
                                 "2.5/97.5 percentiles; conditioned on fitted predictions, no refit"),
                      "interpretation": ("per-horizon marginal/descriptive intervals, not simultaneous; "
                                         "no favorable cell implies overall superiority")},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__ == "__main__":
    main()
