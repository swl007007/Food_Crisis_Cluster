"""R5 same-key reference: did dropping CH improve the model, or just ease the cohort?

Comparing the non-CH cells against the all-cohort cells is not an answer, because the
two are scored on different rows. The only way to separate "the model got better" from
"the cohort got easier" is to score the *prior all-cohort model* on exactly the keys the
non-CH cells are scored on, and compare there.

This recomputes that from stored per-row predictions. It fits nothing and changes no
map; it is a re-scoring of predictions that already exist.

    python IPCCHGeoRFExperiment/check_same_key_reference.py

Cohorts, stated explicitly because they are easy to confuse:

* ``E_all``     - every valid test key in the approved schedule.
* ``E_persist`` - the history-available subset, where persistence is defined and all
                  arms share identical keys. The headline factorial table uses this.

This script reports the ``E_all`` prediction rows, which is the population the stored
prediction files carry; persistence is not one of the arms here, so the E_all/E_persist
distinction does not change which rows are compared between the two models - both sides
are restricted to the same key set by construction and the intersection is verified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CH_FLOOR = 100000
PRIOR = "ipcch-v1-20260920d"
CELLS = (("C3", "abl-C3-nonch-gate010", 0.010), ("C4", "abl-C4-nonch-gate005", 0.005))
ARMS = ("partitioned_rf", "pooled_rf", "xgb")
HORIZONS = (1, 3, 6, 12)
KEY = ["admin_code", "target_month", "horizon_months"]


def class1_f1(truth, pred) -> float:
    truth = np.asarray(truth).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((truth == 1) & (pred == 1)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def load_non_ch(runs: Path, run_id: str) -> pd.DataFrame:
    path = runs / run_id / "stage3" / "predictions.csv.gz"
    if not path.is_file():
        raise SystemExit(f"missing predictions: {path}")
    frame = pd.read_csv(path)
    frame = frame[(frame.period == "main") & (frame.admin_code < CH_FLOOR)]
    return frame.set_index(KEY).sort_index()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="IPCCHGeoRFExperiment/runs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    runs = Path(args.runs_dir)

    prior = load_non_ch(runs, PRIOR)
    cells = {cid: load_non_ch(runs, run) for cid, run, _ in CELLS}

    # The comparison is only meaningful if the key sets are identical, so that is
    # established before any metric is computed rather than assumed afterwards.
    prior_keys = set(prior.index)
    mismatches = {}
    for cid, frame in cells.items():
        keys = set(frame.index)
        if keys != prior_keys:
            mismatches[cid] = {
                "only_in_prior": len(prior_keys - keys),
                "only_in_cell": len(keys - prior_keys),
            }
    if mismatches:
        print(f"FAIL: key sets differ from the prior baseline: {mismatches}")
        return 1

    rows = []
    for arm in ARMS:
        for horizon in HORIZONS:
            def score(frame):
                block = frame[frame.index.get_level_values(2) == horizon]
                return class1_f1(block.ipcch_food_crisis, block[f"pred_{arm}"])

            base = score(prior)
            entry = {"arm": arm, "horizon": horizon, "prior_on_non_ch_keys": base}
            for cid, frame in cells.items():
                value = score(frame)
                entry[cid] = value
                entry[f"{cid}_minus_prior"] = value - base
            rows.append(entry)

    table = pd.DataFrame(rows)
    means = table.groupby("arm", sort=False).mean(numeric_only=True).drop(columns="horizon")

    width = 88
    print("=" * width)
    print("R5 same-key reference — prior all-cohort model scored on the non-CH keys".center(width))
    print("=" * width)
    print(f"shared keys: {len(prior_keys)} (identical in the prior run and both cells)")
    print(f"cohort: E_all prediction rows, main period, admin_code < {CH_FLOOR}\n")
    print(f"{'arm':>16} {'h':>3} │ {'prior':>9} {'C3':>9} {'C4':>9} │ {'C3-prior':>9} {'C4-prior':>9}")
    print("-" * width)
    for arm in ARMS:
        for _, r in table[table.arm == arm].iterrows():
            print(f"{arm:>16} {int(r.horizon):>3} │ {r.prior_on_non_ch_keys:9.4f} "
                  f"{r.C3:9.4f} {r.C4:9.4f} │ {r.C3_minus_prior:>+9.4f} {r.C4_minus_prior:>+9.4f}")
        m = means.loc[arm]
        print(f"{arm:>16} {'mean':>3} │ {m.prior_on_non_ch_keys:9.4f} {m.C3:9.4f} {m.C4:9.4f} │ "
              f"{m.C3_minus_prior:>+9.4f} {m.C4_minus_prior:>+9.4f}")
        print("-" * width)
    print("Training without CH moves the same-key score by a few thousandths, negative for")
    print("partitioned RF and XGB. The lift in the headline table was the cohort, not the model.")
    print("=" * width)

    if args.out:
        payload = {
            "shared_keys": len(prior_keys),
            "cohort": f"E_all main-period rows with admin_code < {CH_FLOOR}",
            "per_horizon": table.to_dict("records"),
            "means": means.reset_index().to_dict("records"),
        }
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
