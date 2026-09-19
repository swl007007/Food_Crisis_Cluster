#!/usr/bin/env python3
"""Phase 4 entrypoint: select tau on 2020 and freeze it.

Reads ONLY the 2020 calibrated rows. It does not open the 2021-2024 file.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from persistencecorrection.selection import select_threshold, freeze, sha256_file

PHASE3 = Path(__file__).resolve().parent / "outputs" / "phase3_20260918"
OUT = Path(__file__).resolve().parent / "outputs" / "phase4_20260919"
OUT.mkdir(parents=True, exist_ok=False)

selections, hashes, traces = {}, {}, {}
for scope in (1, 2):
    src = PHASE3 / f"calibrated_selection_2020_fs{scope}.csv"
    d = pd.read_csv(src)
    hashes[f"fs{scope}"] = sha256_file(src)
    res = select_threshold(d["persistence"], d["p_cal"], d["y_true"])
    traces[f"fs{scope}"] = res.pop("trace")
    res["n_rows"] = int(len(d))
    selections[f"fs{scope}"] = res
    print(f"fs{scope}: n={len(d)} candidates={res['n_candidates']} "
          f"persistence_F1={res['baseline_persistence_f1']:.6f} -> "
          f"tau={res['tau']} F1={res['selected_f1']:.6f}")

digest = freeze(selections, hashes, OUT / "frozen_thresholds.json")
pd.DataFrame(
    [{"scope": s, "tau": t, "f1": f} for s, tr in traces.items() for t, f in tr]
).to_csv(OUT / "selection_trace_2020.csv", index=False)
print(f"\nFROZEN sha256={digest}")
print(f"artifact={OUT / 'frozen_thresholds.json'}")
