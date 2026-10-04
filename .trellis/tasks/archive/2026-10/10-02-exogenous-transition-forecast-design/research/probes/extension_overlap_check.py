"""Covariate-only pre-actual comparison: combined 2025 panel versus the pinned panel (read-only).

Reads keys plus the 69 schema covariate sources only (explicit usecols; no IPC/outcome/expert
columns). (1) On ALL shared (area, month) keys: per-source mismatch counts under the
load_extension rule (np.isclose equal_nan, rtol 0, atol 1e-9; equal signed infinities agree), split admitted (51) / excluded (18), with
mismatch month span. (2) Missingness of the combined panel at the actual-case months (static at
2025-02/06, monthly at 2025-01/05) against the pinned panel at the same calendar months of 2024.
Writes only a compact summary beside this script; no raw values. Usage: python extension_overlap_check.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PKG = HERE.parents[5] / "FEWSNETGeoXGBExperiment"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SRC = HERE.parents[8] / "1.Source Data"
PINNED = SRC / "FEWSNET_forecast_unadjusted_bm.csv"
COMBINED = SRC / "assembled_FEWSNET" / "FEWSNET_forecast_unadjusted_bm_2025_combined.csv"
schema = json.loads((PKG / "feature-schema.json").read_text(encoding="utf-8"))
alignment = json.loads((HERE.parents[1] / "launch" / "alignment.json").read_text(encoding="utf-8"))
sources = list(schema["static_sources"] + schema["dynamic_sources_at_origin"])
assert len(sources) == 69 and set(sources) == set(alignment), "schema/alignment source sets differ"
kind = {s: alignment[s]["kind"] for s in sources}
KEYS = ["FEWSNET_admin_code", "date"]


def load(path, batch):
    f = pd.read_csv(path, usecols=KEYS + batch)
    f["ym"] = f["date"].astype(str).str[:7]
    return f.drop(columns="date")


out = {"pinned": str(PINNED), "combined": str(COMBINED), "rule": "np.isclose(equal_nan=True, rtol=0, atol=1e-9), as load_extension np.allclose", "sources": {}}
batches = [sources[i:i + 24] for i in range(0, len(sources), 24)]
for bi, batch in enumerate(batches):
    a, b = load(PINNED, batch), load(COMBINED, batch)
    if bi == 0:
        dup_b = b.duplicated(["FEWSNET_admin_code", "ym"], keep=False)
        out["combined_duplicate_key_months"] = sorted(b.loc[dup_b, "ym"].unique().tolist())
        out["pinned_months"] = [a["ym"].min(), a["ym"].max(), int(a["ym"].nunique())]
        out["combined_months"] = [b["ym"].min(), b["ym"].max(), int(b["ym"].nunique())]
    b = b[~b.duplicated(["FEWSNET_admin_code", "ym"], keep=False)]
    m = a.merge(b, on=["FEWSNET_admin_code", "ym"], suffixes=("_p", "_c"), validate="one_to_one")
    if bi == 0:
        out["shared_keys"] = int(len(m))
        out["pinned_keys"] = int(len(a))
        out["shared_months"] = [m["ym"].min(), m["ym"].max(), int(m["ym"].nunique())]
    for s in batch:
        x, y = m[f"{s}_p"].to_numpy(float), m[f"{s}_c"].to_numpy(float)
        nx, ny = np.isnan(x), np.isnan(y)
        ok = np.isclose(x, y, equal_nan=True, rtol=0, atol=1e-9)   # the load_extension (np.allclose) rule
        bad = ~ok
        finite = np.isfinite(x) & np.isfinite(y)
        same_inf = np.isinf(x) & np.isinf(y) & (np.sign(x) == np.sign(y))
        months = m.loc[bad, "ym"]
        out["sources"][s] = {"kind": kind[s], "admitted": kind[s] != "excluded", "mismatch": int(bad.sum()),
                             "nan_pattern_mismatch": int((nx != ny).sum()),
                             "both_same_signed_inf": int(same_inf.sum()),
                             "finite_value_mismatch": int((finite & bad).sum()),
                             "inf_involved_mismatch": int((bad & (np.isinf(x) | np.isinf(y))).sum()),
                             "max_abs_finite_diff": float(np.abs(x[finite] - y[finite]).max()) if finite.any() else None,
                             "mismatch_months": int(months.nunique()),
                             "first_last": [months.min(), months.max()] if len(months) else None}
    # (2) missingness at the actual-case months vs pinned at the same 2024 months
    for s in batch:
        months = ("2025-02", "2025-06") if kind[s] == "static" else ("2025-01", "2025-05") \
            if kind[s] == "monthly" else ()
        miss = {}
        for mo in months:
            base = mo.replace("2025", "2024")
            cb = b.loc[b["ym"] == mo, s]
            pb = a.loc[a["ym"] == base, s]
            miss[mo] = {"rows": int(len(cb)), "missing": int(cb.isna().sum()),
                        f"pinned_{base}_rows": int(len(pb)), f"pinned_{base}_missing": int(pb.isna().sum())}
        out["sources"][s]["actual_month_missingness"] = miss
    del a, b, m
    print(f"batch {bi + 1}/{len(batches)} done", flush=True)
src = out["sources"]
for grp, flag in (("admitted", True), ("excluded", False)):
    g = {s: v for s, v in src.items() if v["admitted"] == flag}
    out[f"{grp}_summary"] = {"sources": len(g), "sources_with_mismatch": sum(v["mismatch"] > 0 for v in g.values()),
                             "total_mismatch_cells": sum(v["mismatch"] for v in g.values())}
(HERE.with_name("extension_overlap_check_summary.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: out[k] for k in out if k != "sources"}, indent=1))
