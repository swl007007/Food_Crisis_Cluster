"""Compare a climate-perturbation run with the original p6-formal-20261004b run (task D8).

Reads only saved keyed Stage3 predictions, frozen-map records and fold ledgers of
both runs. Keys, truth and persistence must be identical; any difference stops.
Per H and period it reports the full metric panel for new/old GeoXGB, new/old
pooled and persistence, new-minus-old deltas, and (main period only) the
original paired country bootstrap (2000 draws, numpy default_rng(42)) for
new-minus-old crisis F1 of GeoXGB and pooled. Nothing is fitted.

Usage (pinned runtime, from IPCCHClimateGeoXGBExperiment/):
    PYTHONPATH=. python scripts/compare_with_p6.py --new-run DIR --old-run DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_climate_geoxgb import report
from ipcch_climate_geoxgb.artifacts import sha256_file, write_json
from ipcch_climate_geoxgb.errors import TechnicalError

HORIZONS = (1, 3, 6, 12)
KEY = ["admin_code", "target_ord", "horizon_months"]
SHARED = ["target_month", "origin_ord", "fold_id", "period", "country_key", "phase_truth", "crisis_truth", "q3_truth",
          "persistence_available", "persistence_phase", "persistence_q3", "persistence_source_month",
          "persistence_age_months"]
ARM_SOURCES = {"geo": "geo", "pool": "pool"}


def load(run: Path, h: int) -> pd.DataFrame:
    summary = json.loads((run / "stage3" / "stage3-summary.json").read_text(encoding="utf-8"))
    path = run / "stage3" / f"h{h:02d}" / "predictions.csv.gz"
    if sha256_file(path) != summary["horizons"][str(h)]["predictions_sha256"]:
        raise TechnicalError(f"{run.name} H{h}: predictions differ from the Stage3 summary digest")
    frame = pd.read_csv(path)
    if frame.duplicated(KEY).any():
        raise TechnicalError(f"{run.name} H{h}: duplicated keys")
    return frame


def merge(new: pd.DataFrame, old: pd.DataFrame, h: int) -> pd.DataFrame:
    if len(new) != len(old):
        raise TechnicalError(f"H{h}: {len(new)} new rows vs {len(old)} old rows")
    m = new.merge(old, on=KEY, how="outer", suffixes=("_n", "_o"), indicator=True, validate="one_to_one")
    if (m["_merge"] != "both").any():
        raise TechnicalError(f"H{h}: key sets differ ({m['_merge'].value_counts().to_dict()})")
    out = m[KEY].copy()
    for col in SHARED:
        a, b = m[f"{col}_n"], m[f"{col}_o"]
        same = (a == b) | (a.isna() & b.isna())
        if not same.all():
            raise TechnicalError(f"H{h}: column {col} differs between runs on {int((~same).sum())} keys")
        out[col] = a
    for tag, suffix in (("new", "_n"), ("old", "_o")):
        for arm in ARM_SOURCES:
            out[f"{tag}_{arm}_phase"] = m[f"{arm}_phase{suffix}"]
            out[f"{tag}_{arm}_q3_star"] = m[f"{arm}_q3_star{suffix}"]
            out[f"{tag}_{arm}_q3_raw"] = m[f"{arm}_q3_raw{suffix}"]
        out[f"{tag}_route"] = m[f"route{suffix}"]
    return out.sort_values(KEY, kind="mergesort").reset_index(drop=True)


ARMS = ("new_geo", "old_geo", "new_pool", "old_pool")


def panels(frame: pd.DataFrame, with_persistence: bool) -> dict:
    out = {arm: report.flat(report.panel(frame, arm)) for arm in ARMS}
    if with_persistence:
        out["persistence"] = report.flat(report.panel(frame, "persistence"))
    return out


def delta(p: dict, a: str, b: str) -> dict:
    return {k: (None if p[a][k] is None or p[b][k] is None else p[a][k] - p[b][k]) for k in p[a]}


def label_flips(frame: pd.DataFrame, arm: str) -> dict:
    n, o = frame[f"new_{arm}_phase"].to_numpy(), frame[f"old_{arm}_phase"].to_numpy()
    return {"phase_changed": int((n != o).sum()), "crisis_changed": int(((n >= 3) != (o >= 3)).sum())}


def frozen_summary(run: Path, h: int) -> dict:
    rec = json.loads((run / "stage1" / f"frozen_h{h:02d}.json").read_text(encoding="utf-8"))
    fmap = pd.read_csv(run / "stage1" / f"frozen_map_h{h:02d}.csv", dtype={"node_id": str})
    return {"candidate": rec.get("candidate"), "accepted_split": rec.get("accepted_split"),
            "terminal_regions": rec.get("terminal_regions"), "mapped_areas": int(len(fmap)),
            "region_sizes": fmap["node_id"].value_counts().sort_index().to_dict()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-run", type=Path, required=True)
    ap.add_argument("--old-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"new_run": str(args.new_run), "old_run": str(args.old_run),
              "interpretation":
              "pointwise descriptive country-cluster bootstrap conditional on saved predictions; exploratory, "
              "evaluation period already viewed; GeoXGB difference mixes feature and re-learned map effects, "
              "pooled difference isolates the feature recipe under the matched global pipeline",
              "horizons": {}}
    rows = []
    for h in HORIZONS:
        frame = merge(load(args.new_run, h), load(args.old_run, h), h)
        hres = {"maps": {"new": frozen_summary(args.new_run, h), "old": frozen_summary(args.old_run, h)}}
        for period in ("main", "supplementary"):
            e_all = frame[frame["period"] == period]
            e_persist = e_all[e_all["persistence_available"] == 1]
            entry = {"E_all_keys": int(len(e_all)), "E_persist_keys": int(len(e_persist)),
                     "routes": {"new_local_rows": int((e_all["new_route"] == "local").sum()),
                                "old_local_rows": int((e_all["old_route"] == "local").sum())},
                     "label_flips": {arm: label_flips(e_all, arm) for arm in ARM_SOURCES}}
            if len(e_all):
                p_all = panels(e_all, False)
                entry["E_all"] = {"panels": p_all,
                                  "delta_new_minus_old_geo": delta(p_all, "new_geo", "old_geo"),
                                  "delta_new_minus_old_pool": delta(p_all, "new_pool", "old_pool"),
                                  "delta_new_geo_minus_new_pool": delta(p_all, "new_geo", "new_pool"),
                                  "delta_old_geo_minus_old_pool": delta(p_all, "old_geo", "old_pool")}
                p_per = panels(e_persist, True)
                entry["E_persist"] = {"panels": p_per,
                                      "delta_new_geo_minus_persistence": delta(p_per, "new_geo", "persistence"),
                                      "delta_old_geo_minus_persistence": delta(p_per, "old_geo", "persistence")}
                for arm in ("geo", "pool"):
                    rows.append({"H": h, "period": period, "arm": arm, "n": len(e_all),
                                 "f1_new": p_all[f"new_{arm}"]["binary.f1"], "f1_old": p_all[f"old_{arm}"]["binary.f1"],
                                 "macro_f1_new": p_all[f"new_{arm}"]["four_class.macro_f1"],
                                 "macro_f1_old": p_all[f"old_{arm}"]["four_class.macro_f1"],
                                 "q3_r2_new": p_all[f"new_{arm}"]["q3_r2_projected"],
                                 "q3_r2_old": p_all[f"old_{arm}"]["q3_r2_projected"]})
            if period == "main":
                boot = {}
                for arm in ("geo", "pool"):
                    record, draws = report.bootstrap_delta(e_all, f"new_{arm}", f"old_{arm}")
                    if draws is not None:
                        dpath = args.out / f"bootstrap_h{h:02d}_new_minus_old_{arm}.csv.gz"
                        draws.to_csv(dpath, index=False, compression=report.GZ)
                        record["draws_sha256"] = sha256_file(dpath)
                    record.pop("country_counts", None)
                    boot[f"new_minus_old_{arm}_E_all"] = record
                for tag in ("new", "old"):
                    record, _ = report.bootstrap_delta(e_persist, f"{tag}_geo", "persistence")
                    record.pop("country_counts", None)
                    boot[f"{tag}_geo_minus_persistence_E_persist"] = record
                entry["bootstrap"] = boot
            hres[period] = entry
        result["horizons"][str(h)] = hres
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "summary_table.csv", index=False)
    digest = write_json(args.out / "comparison.json", result)
    print(json.dumps({"comparison_sha256": digest, "rows": len(table)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
