"""Preparation for the four-class baseline: pinned sources, ledgers, snapshots, schedules.

python scripts/prepare_fourclass.py --run-dir runs/<id> [--source-root PATH]

Writes into a fresh ``<run>/prepared`` directory and refuses to overwrite one:

* manifests/sources.json    source, release and package identities (A1, A8)
* manifests/preflight.json  key, phase, expert and geography checks (A1)
* manifests/runtime.json    interpreter and package versions (A1)
* manifests/features.json   ordered schema, provenance rules, raw missingness (A2)
* manifests/schedule.json   Stage 1 and Stage 3 folds with support (A3)
* ledgers/observations.csv  every valid area-month assessment, raw and merged
* ledgers/baselines.csv     Stage 3 truth, exact-origin persistence and expert (R3)
* snapshot_h{4,8,12}.parquet origin-aligned 162-column predictors per key (R6-R8)
* geometry/                  coordinates, adjacency cache, polygon contiguity info
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from src.feature import fourclass_features as ff  # noqa: E402
from src.metrics.fourclass import merge_phase  # noqa: E402

#: Package copy of the approved schema (byte-identical to the archived task's
#: feature-schema.json; SHA-256 recorded in every run).
SCHEMA_PATH = PACKAGE / "feature-schema.json"
APPROVED_SCHEMA_SHA256 = "51b6f8b21b76a78510522c34e2d1f2a648b7aec768bbcac3dd2318669fa13349"
RELEASE_ZIP = PACKAGE.parent / "GeoRFBaseline" / "releases" / "georf-baseline-v0.1.0.zip"
RELEASE_SHA256 = "39a26138e3fafb0be2bbd22e9760095d6cdefa7d79b98b4f798cb3aa79b500a0"
#: The repository sits at Analysis/2.source_code/Step5_Geo_RF_trial/<repo>; the pinned
#: sources are Analysis/1.Source Data. A relocated checkout must pass --source-root.
_PARENTS = Path(__file__).resolve().parents
DEFAULT_SOURCE_ROOT = _PARENTS[5] / "1.Source Data" if len(_PARENTS) > 5 else None

#: research/release-and-integration.md, "Inspected input identities".
PINNED_SOURCES = {
    "panel": ("FEWSNET_forecast_unadjusted_bm.csv",
              "611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651"),
    "fewsnet": ("Outcome/FEWSNET_IPC/FEWSNET.csv",
                "8fdd4cca6f6ba26b84efc209c8eb36492e1257d51e24edd2c2ad4962df7b38d0"),
    "coordinates": ("FEWSNET_admin_code_lat_lon.csv",
                    "a06be85849bb726a4505ed284bed14b100b61f998fb4586a6e14439aca8a4bcb"),
    "shapefile": ("Outcome/FEWSNET_IPC/FEWS NET Admin Boundaries/FEWS_Admin_LZ_v3.shp",
                  "3aba66a6fbf6b2a8beb153df76a67662ce4e0a898fcb11e292a17cc174f5f742"),
}
SHAPEFILE_SIDECARS = (".shx", ".dbf", ".prj", ".cpg")
PINNED_RUNTIME = {"python": "3.12.10", "numpy": "2.2.6", "pandas": "2.2.3",
                  "scikit-learn": "1.6.1", "scipy": "1.15.2", "geopandas": "1.0.1",
                  "shapely": "2.1.0", "polars": "1.27.1", "xgboost": "3.0.0"}

from src.experiment import plan  # noqa: E402

HORIZONS = plan.HORIZONS
SCOPE_OF = plan.SCOPE_OF
#: Snapshot keys start with the first label month of the source (2010-01). The 59-month
#: windows of the actual schedule reach back to 2010-03 (Stage 3 internal gate origins);
#: build_schedule checks that every scheduled window lower bound is covered. Features of
#: early keys keep the NaN that the frozen formulas produce before the 2010-01 scaffold.
SNAPSHOT_FIRST_MONTH = "2010-01"
STAGE1_TARGETS = ("2018-01", "2020-12")
STAGE3_TARGETS = plan.FINAL_TARGETS
TRAIN_WINDOW = plan.WINDOW + 1  # labels in [O-59, O), i.e. 59 months excluding the origin (D9)
PARTITION_INFO_CUTOFF = plan.PARTITION_INFO_CUTOFF
EXPERT_FIELD = {4: "fews_proj_near", 8: "fews_proj_med"}
ADMIN_UNIVERSE = (0, 5717)


class PreflightError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def mi(label: str) -> int:
    year, month = label.split("-")
    return int(year) * 12 + int(month) - 1


# --------------------------------------------------------------------------------------
# identities
# --------------------------------------------------------------------------------------

def pin_sources(root: Path) -> dict:
    record = {}
    for key, (relative, expected) in PINNED_SOURCES.items():
        path = root / relative
        if not path.is_file():
            raise PreflightError(f"missing pinned source {path}")
        actual = sha256(path)
        if actual != expected:
            raise PreflightError(f"{key} hash drift: {actual} != {expected}")
        record[key] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    shp = root / PINNED_SOURCES["shapefile"][0]
    sidecars = {}
    for suffix in SHAPEFILE_SIDECARS:
        side = shp.with_suffix(suffix)
        if not side.is_file():
            raise PreflightError(f"missing shapefile sidecar {side}")
        sidecars[suffix] = {"sha256": sha256(side), "bytes": side.stat().st_size}
    record["shapefile"]["sidecars"] = sidecars
    return record


def package_identity() -> dict:
    """Release hash plus every payload file that differs from it (R15, A8)."""
    import zipfile
    if sha256(RELEASE_ZIP) != RELEASE_SHA256:
        raise PreflightError("release archive hash drift")
    released = {}
    with zipfile.ZipFile(RELEASE_ZIP) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            released[name.split("/", 1)[1]] = hashlib.sha256(archive.read(name)).hexdigest()
    current = {}
    for path in sorted(PACKAGE.rglob("*")):
        rel = path.relative_to(PACKAGE).as_posix()
        if (path.is_file() and not rel.startswith("runs/") and "__pycache__" not in rel
                and not rel.endswith(".pyc")):
            current[rel] = sha256(path)
    return {
        "release_zip": str(RELEASE_ZIP), "release_sha256": RELEASE_SHA256,
        "unchanged": sorted(p for p in released if current.get(p) == released[p]),
        "modified": {p: {"release": released[p], "package": current[p]}
                     for p in sorted(released) if p in current and current[p] != released[p]},
        "removed": sorted(p for p in released if p not in current),
        "added": {p: current[p] for p in sorted(current) if p not in released},
    }


def runtime_identity() -> dict:
    from importlib.metadata import version
    record = {"python": platform.python_version(), "executable": sys.executable,
              "platform": platform.platform()}
    for name in PINNED_RUNTIME:
        if name != "python":
            record[name] = version(name)
    import xgboost
    build = xgboost.build_info()
    record["xgboost_build"] = {k: build[k] for k in sorted(build) if k != "libxgboost"}
    record["xgboost_library_sha256"] = sha256(Path(build["libxgboost"]))
    drift = {k: (record[k], v) for k, v in PINNED_RUNTIME.items() if record[k] != v}
    record["matches_tested_environment"] = not drift
    if drift:
        raise PreflightError(f"runtime differs from the pinned release environment: {drift}")
    return record


# --------------------------------------------------------------------------------------
# source validation
# --------------------------------------------------------------------------------------

def load_panel(path: Path) -> pd.DataFrame:
    import polars as pl
    panel = pl.read_csv(path, infer_schema_length=100000).to_pandas()
    panel["date"] = pd.to_datetime(panel["date"], format="%Y-%m")  # source stamps are YYYY-MM
    for col in panel.columns:
        if col.startswith("AEZ_"):
            panel[col] = panel[col].astype(float)
    return panel.sort_values(["FEWSNET_admin_code", "date"]).reset_index(drop=True)


def check_phase(values: pd.Series, name: str) -> dict:
    finite = values.dropna()
    bad = ~finite.isin([1.0, 2.0, 3.0, 4.0, 5.0])
    if bad.any():
        raise PreflightError(f"{name}: {int(bad.sum())} non-integral or out-of-range phases")
    return {"valid": int(finite.size), "missing": int(values.isna().sum()),
            "raw_counts": {str(int(k)): int(v) for k, v in finite.value_counts().sort_index().items()}}


def preflight(panel: pd.DataFrame, fewsnet: pd.DataFrame, coords: pd.DataFrame,
              schema: dict) -> dict:
    report = {}
    keys = panel[["FEWSNET_admin_code", "date"]]
    if keys.isna().any().any() or keys.duplicated().any():
        raise PreflightError("null or duplicate panel keys")
    if (panel["date"].dt.day != 1).any() or (panel["month"] != panel["date"].dt.month).any():
        raise PreflightError("panel dates are not first-of-month or disagree with month")
    areas = panel["FEWSNET_admin_code"].unique()
    months = panel["date"].nunique()
    if len(panel) != areas.size * months:
        raise PreflightError("panel is not a complete area x month scaffold")
    lo, hi = ADMIN_UNIVERSE
    if areas.min() < lo or areas.max() > hi:
        raise PreflightError("admin codes outside the release universe 0..5717")
    report["scaffold"] = {"rows": int(len(panel)), "areas": int(areas.size), "months": int(months),
                          "first": str(panel["date"].min().date()), "last": str(panel["date"].max().date()),
                          "admin_code_range": [int(areas.min()), int(areas.max())]}

    report["truth"] = check_phase(panel["fews_ipc"], "fews_ipc")
    labelled = panel["fews_ipc"].notna()
    crisis_consistent = panel.loc[labelled, "fews_ipc_crisis"].eq((panel.loc[labelled, "fews_ipc"] >= 3).astype(int))
    orphan_binary = int((~labelled & panel["fews_ipc_crisis"].notna()).sum())
    if not crisis_consistent.all() or orphan_binary:
        raise PreflightError("fews_ipc_crisis disagrees with fews_ipc >= 3")
    report["truth"]["binary_matches_phase_ge3"] = int(crisis_consistent.sum())
    report["truth"]["binary_without_phase"] = orphan_binary
    report["truth"]["label_months"] = sorted(panel.loc[labelled, "date"].dt.strftime("%Y-%m").unique().tolist())
    for field in ("fews_proj_near", "fews_proj_med"):
        report[field] = check_phase(panel[field], field)

    iso = panel.groupby("FEWSNET_admin_code")["ISO"].nunique(dropna=False)
    if (iso != 1).any() or panel["ISO"].isna().any():
        raise PreflightError("an area maps to zero or several countries")
    report["countries"] = {k: int(v) for k, v in panel.groupby("ISO")["FEWSNET_admin_code"].nunique().items()}

    first = panel.groupby("FEWSNET_admin_code")[["lat", "lon"]].first()
    coords = coords.set_index("FEWSNET_admin_code")
    if set(coords.index) != set(first.index):
        raise PreflightError("coordinate CSV and panel disagree on the area universe")
    diff = float((first - coords.loc[first.index, ["lat", "lon"]]).abs().to_numpy().max())
    if diff != 0.0:
        raise PreflightError(f"panel and coordinate CSV disagree (max |diff| {diff})")
    report["coordinates"] = {"areas": int(len(coords)), "max_abs_diff_vs_panel": diff}

    static = {}
    for name in schema["static_sources"]:
        values = panel[["FEWSNET_admin_code", name]].copy()
        values[name] = values[name].astype(float)
        finite = values[np.isfinite(values[name])]
        conflicts = int((finite.groupby("FEWSNET_admin_code")[name].nunique() > 1).sum())
        if conflicts:
            raise PreflightError(f"static source {name} varies within {conflicts} areas")
        static[name] = {"areas_with_conflict": 0, "missing_cells": int(values[name].isna().sum())}
    report["static_invariance"] = static

    # Source agreement with the independent FEWS NET outcome file (A1).
    fx = fewsnet.dropna(subset=["admin_code", "year", "month"]).copy()
    fx["date"] = pd.to_datetime(dict(year=fx["year"].astype(int), month=fx["month"].astype(int), day=1))
    fx["FEWSNET_admin_code"] = fx["admin_code"].astype(int)
    if fx.duplicated(["FEWSNET_admin_code", "date"]).any():
        raise PreflightError("duplicate keys in FEWSNET.csv")
    merged = panel[["FEWSNET_admin_code", "date", "fews_ipc", "fews_proj_near", "fews_proj_med"]].merge(
        fx[["FEWSNET_admin_code", "date", "fews_ipc", "fews_proj_near", "fews_proj_med"]],
        on=["FEWSNET_admin_code", "date"], how="outer", suffixes=("_panel", "_fewsnet"))
    agreement = {"fewsnet_rows_dropped_null_key": int(len(fewsnet) - len(fx))}
    for field in ("fews_ipc", "fews_proj_near", "fews_proj_med"):
        a, b = merged[field + "_panel"], merged[field + "_fewsnet"]
        both = a.notna() & b.notna()
        extra = merged.loc[a.isna() & b.notna(), "date"].dt.strftime("%Y-%m").value_counts().sort_index()
        agreement[field] = {"both": int(both.sum()), "disagree": int((both & (a != b)).sum()),
                            "panel_only": int((a.notna() & b.isna()).sum()),
                            "fewsnet_only": int((a.isna() & b.notna()).sum()),
                            "fewsnet_only_by_month": {k: int(v) for k, v in extra.items()}}
        if agreement[field]["disagree"] or agreement[field]["panel_only"]:
            raise PreflightError(f"{field}: panel disagrees with FEWSNET.csv")
    agreement["resolution"] = ("The panel is the model input and the sole truth/expert/history "
                               "source. FEWSNET.csv agrees on every shared key; its extra truth "
                               "rows predate the panel scaffold (2009) and are not used.")
    report["fewsnet_agreement"] = agreement

    inf = {c: int(np.isinf(panel[c].to_numpy(dtype=float)).sum())
           for c in schema["static_sources"] + schema["dynamic_sources_at_origin"]}
    report["infinite_source_cells"] = {k: v for k, v in inf.items() if v}
    report["infinite_policy"] = ("Every XGBoost input converts +/-inf to NaN and passes NaN natively as "
                                 "missing (D14); no imputer is fitted.")
    return report


# --------------------------------------------------------------------------------------
# ledgers, snapshots and schedules
# --------------------------------------------------------------------------------------

def build_observations(panel: pd.DataFrame) -> pd.DataFrame:
    obs = panel.loc[panel["fews_ipc"].notna(), ["FEWSNET_admin_code", "date", "fews_ipc", "ISO"]].copy()
    obs["month"] = ff.month_index(obs["date"])
    obs["merged_class"] = merge_phase(obs["fews_ipc"]).astype(int)
    obs["class_code"] = obs["merged_class"] - 1
    return obs.rename(columns={"FEWSNET_admin_code": "area", "fews_ipc": "raw_phase", "ISO": "country"})


def build_snapshot(scaffold, schema, observations, horizon) -> pd.DataFrame:
    keys = observations.loc[observations["month"] >= mi(SNAPSHOT_FIRST_MONTH),
                            ["area", "month", "country", "raw_phase", "class_code"]].copy()
    keys = keys.rename(columns={"month": "target_month"}).sort_values(["area", "target_month"])
    keys["horizon"] = horizon
    keys["origin_month"] = keys["target_month"] - horizon
    areas = keys["area"].to_numpy()
    covariates = ff.covariate_features(scaffold, schema, areas,
                                       keys["target_month"].to_numpy(), keys["origin_month"].to_numpy())
    history = ff.history_features(
        observations.rename(columns={"merged_class": "phase"})[["area", "month", "phase"]],
        areas, keys["origin_month"].to_numpy())
    features = pd.concat([covariates, history], axis=1)[schema["ordered_features"]]
    keys = keys.reset_index(drop=True)
    snapshot = pd.concat([keys, features], axis=1)
    if snapshot["origin_month"].ge(snapshot["target_month"]).any():
        raise PreflightError("an origin is not strictly before its target")
    return snapshot


def build_baselines(panel: pd.DataFrame, observations: pd.DataFrame, targets=None) -> pd.DataFrame:
    """Truth plus exact-origin persistence and calendar-aligned expert (D3, D5).

    ``targets`` maps horizon -> (first, last) target month; default = the final Stage 3
    schedule. The development ledger uses the six development targets per horizon."""
    frames = []
    obs = observations.set_index(["area", "month"])
    panel = panel.assign(month=ff.month_index(panel["date"])).set_index(["FEWSNET_admin_code", "month"])
    for horizon, months in (targets or {h: STAGE3_TARGETS[h] for h in HORIZONS}).items():
        if isinstance(months, tuple):
            first, last = months
            rows = observations[(observations["month"] >= mi(first)) & (observations["month"] <= mi(last))]
        else:
            rows = observations[observations["month"].isin([mi(t) for t in months])]
        frame = rows[["area", "month", "country", "raw_phase", "class_code"]].rename(
            columns={"month": "target_month", "raw_phase": "truth_raw_phase", "class_code": "truth_code"})
        frame = frame.assign(horizon=horizon, origin_month=frame["target_month"] - horizon)
        index = pd.MultiIndex.from_arrays([frame["area"], frame["origin_month"]])
        persist = obs["raw_phase"].reindex(index).to_numpy()
        frame["persistence_source_month"] = ff.month_label(frame["origin_month"])
        frame["persistence_raw_phase"] = persist
        frame["persistence_code"] = merge_phase(persist) - 1
        if horizon in EXPERT_FIELD:
            raw = panel[EXPERT_FIELD[horizon]].reindex(index).to_numpy()
            frame["expert_field"] = EXPERT_FIELD[horizon]
            frame["expert_source_month"] = ff.month_label(frame["origin_month"])
            frame["expert_raw_phase"] = raw
            frame["expert_code"] = merge_phase(raw) - 1
        else:
            frame["expert_field"] = "none (no fs3 expert proxy, D4)"
            frame["expert_source_month"] = ""
            frame["expert_raw_phase"] = np.nan
            frame["expert_code"] = np.nan
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out.insert(2, "target_label", ff.month_label(out["target_month"]))
    return out


def gate_dates(label_months, origin: int) -> list:
    """The six most recent globally observed label months U < O (plan section 5)."""
    earlier = sorted(m for m in label_months if m < origin)
    return earlier[-plan.GATE_DATES:]


def _fold(horizon, target, labelled, label_months, with_gate):
    origin = target - horizon
    entry = {"scope": SCOPE_OF[horizon], "horizon": horizon,
             "target_month": ff.month_label([target])[0], "origin_month": ff.month_label([origin])[0],
             "train_label_months": [ff.month_label([origin - plan.WINDOW])[0], ff.month_label([origin - 1])[0]],
             "target_rows": int(labelled.get(target, 0)),
             "status": "scheduled" if labelled.get(target, 0) else "skipped_empty_target"}
    if with_gate:
        gates = gate_dates(label_months, origin)
        entry["gate"] = [{"validation_month": ff.month_label([u])[0],
                          "internal_origin": ff.month_label([u - horizon])[0],
                          "fit_label_months": [ff.month_label([u - horizon - plan.WINDOW])[0],
                                               ff.month_label([u - horizon - 1])[0]]} for u in gates]
    return entry


def build_schedule(observations: pd.DataFrame) -> dict:
    labelled = observations.groupby("month").size()
    label_months = sorted(int(m) for m in labelled.index)
    schedule = {"stage1": [], "stage1_roots": [], "stage1_candidates": [], "development": [], "stage3": [],
                "stage1_tb3_roots": [], "stage1_tb3_candidates": [],
                "stage1_rootinc_roots": [], "stage1_rootinc_candidates": [],
                "stage1_rootconf_roots": [], "stage1_rootconf_candidates": []}
    for horizon in HORIZONS:
        for target in range(mi(STAGE1_TARGETS[0]), mi(STAGE1_TARGETS[1]) + 1):
            entry = _fold(horizon, target, labelled, label_months, with_gate=False)
            label = entry["target_month"]
            if entry["status"] == "scheduled" and label not in plan.STAGE1_TARGETS:
                raise PreflightError(f"labelled 2018-2020 month {label} is not a frozen Stage 1 target")
            if label in plan.STAGE1_TARGETS and entry["status"] != "scheduled":
                raise PreflightError(f"frozen Stage 1 target {label} has no labels")
            schedule["stage1"].append(entry)
            if entry["status"] != "scheduled":
                continue
            for ratio in plan.SPLIT_RATIOS:
                for seed in plan.SPLIT_SEEDS:
                    schedule["stage1_roots"].append({"horizon": horizon, "target_month": label,
                                                     "origin_month": entry["origin_month"],
                                                     "ratio": ratio, "split_seed": seed})
                    for local in plan.L_CONFIGS:
                        for family in plan.THRESHOLD_FAMILIES:
                            schedule["stage1_candidates"].append({
                                "horizon": horizon, "target_month": label, "origin_month": entry["origin_month"],
                                "ratio": ratio, "split_seed": seed, "local_config": local,
                                "threshold_family": family})
            if label in plan.TB3_TARGETS:
                # D27 (A2): separate six-root time-block contrast; the 648 lists are unchanged.
                schedule["stage1_tb3_roots"].append({"horizon": horizon, "target_month": label,
                                                     "origin_month": entry["origin_month"],
                                                     "ratio": plan.TIME_BLOCK, "split_seed": plan.TB3_SEED})
                schedule["stage1_tb3_candidates"].append({
                    "horizon": horizon, "target_month": label, "origin_month": entry["origin_month"],
                    "ratio": plan.TIME_BLOCK, "split_seed": plan.TB3_SEED, "local_config": plan.TB3_LOCAL,
                    "threshold_family": plan.TB3_FAMILY})
            if label in plan.ROOTINC_TARGETS:
                # D28 (A3): six shared-root increment roots; the 648 and tb3 lists are unchanged.
                common = {"horizon": horizon, "target_month": label, "origin_month": entry["origin_month"],
                          "ratio": plan.ROOTINC_RATIO, "split_seed": plan.ROOTINC_SEED, "increment_source": "root"}
                schedule["stage1_rootinc_roots"].append(dict(common))
                schedule["stage1_rootinc_candidates"].append({**common, "local_config": plan.ROOTINC_LOCAL,
                                                              "threshold_family": plan.ROOTINC_FAMILY})
                # D29 (A4): the same six roots with the label-blind S/C confirmation split.
                conf = {**common, "confirmation_seed": plan.CONFIRMATION_SEED}
                schedule["stage1_rootconf_roots"].append(dict(conf))
                schedule["stage1_rootconf_candidates"].append({**conf, "local_config": plan.ROOTINC_LOCAL,
                                                               "threshold_family": plan.ROOTINC_FAMILY})
        for target in (mi(t) for t in plan.DEV_TARGETS):
            entry = _fold(horizon, target, labelled, label_months, with_gate=True)
            if entry["status"] != "scheduled":
                raise PreflightError(f"development target {entry['target_month']} has no labels")
            schedule["development"].append(entry)
        first, last = STAGE3_TARGETS[horizon]
        for target in range(mi(first), mi(last) + 1):
            origin = target - horizon
            if origin <= mi(PARTITION_INFO_CUTOFF):
                raise PreflightError("a Stage 3 origin is not after the partition cutoff")
            schedule["stage3"].append(_fold(horizon, target, labelled, label_months, with_gate=True))
    for stage in ("stage1", "stage3"):
        rows = schedule[stage]
        schedule[f"{stage}_counts"] = {
            "scheduled_folds": sum(r["status"] == "scheduled" for r in rows),
            "skipped_empty": sum(r["status"] != "scheduled" for r in rows),
            "first_supported_target": {
                str(h): min((r["target_month"] for r in rows if r["horizon"] == h and r["status"] == "scheduled"),
                            default=None) for h in HORIZONS},
        }
    schedule["stage1_counts"].update(roots=len(schedule["stage1_roots"]),
                                     candidates=len(schedule["stage1_candidates"]))
    if len(schedule["stage1_candidates"]) != 648 or len(schedule["development"]) != 18:
        raise PreflightError("the frozen 648 candidate tasks / 18 development folds are not reproduced")
    tb3_expected = len(plan.TB3_TARGETS) * len(HORIZONS)
    if len(schedule["stage1_tb3_roots"]) != tb3_expected or len(schedule["stage1_tb3_candidates"]) != tb3_expected \
            or tb3_expected != 6:
        raise PreflightError("the D27 time-block contrast must schedule exactly 6 roots / 6 candidates")
    schedule["stage1_tb3_counts"] = {"roots": len(schedule["stage1_tb3_roots"]),
                                     "candidates": len(schedule["stage1_tb3_candidates"]),
                                     "rule": ("D27: latest 3 observed label months of each root pool = common "
                                              "E1/E2 validation, earlier rows fitting; L1/gt0, seed 42 only")}
    if len(schedule["stage1_rootinc_roots"]) != 6 or len(schedule["stage1_rootinc_candidates"]) != 6:
        raise PreflightError("the D28 shared-root contrast must schedule exactly 6 roots / 6 candidates")
    schedule["stage1_rootinc_counts"] = {"roots": 6, "candidates": 6,
                                         "rule": "D28/A3: r80, seed 42, L1, gt0; children continue the shared root once"}
    if len(schedule["stage1_rootconf_roots"]) != 6 or len(schedule["stage1_rootconf_candidates"]) != 6:
        raise PreflightError("the D29 confirmation contrast must schedule exactly 6 roots / 6 candidates")
    schedule["stage1_rootconf_counts"] = {"roots": 6, "candidates": 6,
                                          "rule": ("D29/A4: the D28 rootinc roots; original r80 validation split "
                                                   "label-blind into search S and diagnostic confirmation C (seed 42)")}
    lower = []
    for stage in ("stage1", "development", "stage3"):
        for r in schedule[stage]:
            if r["status"] == "scheduled":
                lower.append(r["train_label_months"][0])
                lower += [g["fit_label_months"][0] for g in r.get("gate", [])]
    schedule["earliest_window_lower_bound"] = min(lower)
    schedule["first_label_month"] = ff.month_label([label_months[0]])[0]
    if mi(SNAPSHOT_FIRST_MONTH) > label_months[0]:
        raise PreflightError("snapshot keys start after the first label month a window can reach")
    schedule["training_window_rule"] = ("labels in [O-59, O): 59 calendar months, origin excluded (D9 revised); "
                                        "applies to every global, parent, child and local fit at its own origin")
    schedule["gate_rule"] = ("Stage 3 / development: the six most recent globally observed label months "
                             "U < O; internal origin V = U - H; internal fits use [V-59, V)")
    return schedule


def build_geometry(out: Path, coords: pd.DataFrame, shapefile: Path, areas: np.ndarray) -> dict:
    """Release polygon grouping over the labelled-area universe (setup_spatial_groups)."""
    from src.adjacency.adjacency_utils import load_or_create_adjacency_matrix
    from src.customize.customize import PolygonGroupGenerator
    import geopandas as gpd

    out.mkdir(parents=True, exist_ok=True)
    coords.to_csv(out / "FEWSNET_admin_code_lat_lon.csv", index=False)
    shapes = gpd.read_file(shapefile)
    adj_raw, id_mapping, _ = load_or_create_adjacency_matrix(
        shapefile_path=str(shapefile), polygon_id_column="admin_code",
        cache_dir=str(out), force_regenerate=True)
    code_to_adj = {int(code): idx for idx, code in id_mapping.items()}
    area_to_poly = {int(code): i for i, code in enumerate(areas)}
    adjacency = {}
    for i, code in enumerate(areas):
        neighbours = []
        for adj_idx in adj_raw.get(code_to_adj.get(int(code), -1), []):
            mapped = area_to_poly.get(int(id_mapping[adj_idx]))
            if mapped is not None:
                neighbours.append(mapped)
        adjacency[i] = np.array(neighbours, dtype=int)
    lookup = coords.set_index("FEWSNET_admin_code").loc[areas]
    generator = PolygonGroupGenerator(
        polygon_centroids=lookup[["lat", "lon"]].to_numpy(),
        polygon_group_mapping={i: [int(areas[i])] for i in range(len(areas))},
        neighbor_distance_threshold=0.8, adjacency_dict=adjacency)
    with open(out / "polygon_contiguity_info.pkl", "wb") as handle:
        pickle.dump(generator.get_contiguity_info(), handle)
    degrees = np.array([len(v) for v in adjacency.values()])
    return {
        "shapefile_polygons": int(len(shapes)), "crs": str(shapes.crs),
        "invalid_geometries": int((~shapes.is_valid).sum()),
        "invalid_geometry_note": "used as released; adjacency uses touches() with a positive shared length",
        "area_universe": "areas with at least one valid label (release setup_spatial_groups filters to labelled rows)",
        "areas": int(len(areas)),
        "areas_in_shapefile": int(sum(int(a) in code_to_adj for a in areas)),
        "adjacency_degree": {"min": int(degrees.min()), "max": int(degrees.max()),
                             "mean": float(degrees.mean()), "isolated": int((degrees == 0).sum())},
    }


def feature_manifest(schema: dict, snapshots: dict) -> dict:
    provenance = {}
    for name in schema["static_sources"]:
        provenance[name] = {"role": "static", "source": name, "month": "O (invariant; checked)"}
    for name in schema["dynamic_sources_at_origin"]:
        provenance[name] = {"role": "dynamic", "source": name, "month": "exactly O, no earlier substitution"}
    for name, source, width in (("WFP_Price_m4", "WFP_Price", 4), ("WFP_Price_m12", "WFP_Price", 12),
                                ("nightlight_m12", "nightlight", 12)):
        provenance[name] = {"role": "covariate_derived", "source": source,
                            "month": f"sum over [O-{width}, O-1], all {width} values required"}
    for k in range(1, 13):
        provenance[f"EVI_l{k}"] = {"role": "covariate_derived", "source": "EVI", "month": f"exactly O-{k}"}
    for name in schema["known_calendar"]:
        provenance[name] = {"role": "calendar", "source": "target month T", "month": "T (known in advance)"}
    for block, names in schema["history_blocks"].items():
        for name in names:
            provenance[name] = {"role": f"history:{block}", "source": "fews_ipc merged 1..4",
                                "month": "same-area observations at months <= O"}
    missing = {}
    for horizon, snap in snapshots.items():
        values = snap[schema["ordered_features"]].to_numpy(dtype=float)
        missing[str(horizon)] = {
            "keys": int(len(snap)),
            "nan_cells": {n: int(c) for n, c in zip(schema["ordered_features"], np.isnan(values).sum(axis=0)) if c},
            "inf_cells": {n: int(c) for n, c in zip(schema["ordered_features"], np.isinf(values).sum(axis=0)) if c},
        }
    return {"ordered_features": schema["ordered_features"], "count": len(schema["ordered_features"]),
            "schema_sha256": sha256(SCHEMA_PATH), "provenance": provenance,
            "excluded_from_X": schema["metadata_not_predictors_proposed"] + schema["excluded_predictors"]
            + ["fews_ipc (current outcome)", "fews_ipc_crisis (current outcome)"],
            "raw_missingness_before_imputation": missing,
            "snapshot_key_rule": (f"every valid labelled area-month from {SNAPSHOT_FIRST_MONTH}; features "
                                  "depend only on the complete scaffold and on observations <= O, never on "
                                  "which keys are selected")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT,
                        required=DEFAULT_SOURCE_ROOT is None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    out = args.run_dir / "prepared"
    if out.exists():
        raise FileExistsError(f"{out} exists; preparation outputs are never overwritten")
    out.mkdir(parents=True)
    manifests = out / "manifests"

    from src.utils.run_identity import code_identity as _code, code_identity_at as _code_at
    from src.utils.run_identity import verifier_identity as _ver, verifier_identity_at as _ver_at
    if not args.preflight_only and (_code() != _code_at("HEAD") or _ver() != _ver_at("HEAD")):
        raise PreflightError("package code differs from the committed HEAD; commit before an "
                             "authoritative run so the run is bound to committed code")
    runtime = runtime_identity()
    sources = pin_sources(args.source_root)
    if sha256(SCHEMA_PATH) != APPROVED_SCHEMA_SHA256:
        raise PreflightError("feature-schema.json differs from the approved schema")
    schema = ff.load_schema(SCHEMA_PATH)
    write_json(manifests / "runtime.json", runtime)
    write_json(manifests / "sources.json", {"sources": sources, "package": package_identity(),
                                            "feature_schema": {"path": str(SCHEMA_PATH), "sha256": sha256(SCHEMA_PATH)}})

    print("loading panel", flush=True)
    panel = load_panel(Path(sources["panel"]["path"]))
    fewsnet = pd.read_csv(sources["fewsnet"]["path"])
    coords = pd.read_csv(sources["coordinates"]["path"])
    report = preflight(panel, fewsnet, coords, schema)
    write_json(manifests / "preflight.json", report)
    print("preflight passed", flush=True)
    if args.preflight_only:
        return

    observations = build_observations(panel)
    ledgers = out / "ledgers"
    ledgers.mkdir()
    observations.assign(month_label=ff.month_label(observations["month"])).drop(columns=["date"]).to_csv(
        ledgers / "observations.csv", index=False)
    baselines = build_baselines(panel, observations)
    baselines.to_csv(ledgers / "baselines.csv", index=False)
    build_baselines(panel, observations, {h: list(plan.DEV_TARGETS) for h in HORIZONS}).to_csv(
        ledgers / "dev_baselines.csv", index=False)
    schedule = build_schedule(observations)
    write_json(manifests / "schedule.json", schedule)

    print("building snapshots", flush=True)
    scaffold = ff.Scaffold(panel, schema["static_sources"] + schema["dynamic_sources_at_origin"])
    snapshots = {}
    for horizon in HORIZONS:
        snapshots[horizon] = build_snapshot(scaffold, schema, observations, horizon)
        snapshots[horizon].to_parquet(out / f"snapshot_h{horizon}.parquet", index=False)
        print(f"  h={horizon}: {len(snapshots[horizon])} keys", flush=True)
    write_json(manifests / "features.json", feature_manifest(schema, snapshots))

    print("building geometry", flush=True)
    areas = np.sort(observations["area"].unique())
    geometry = build_geometry(out / "geometry", coords, Path(sources["shapefile"]["path"]), areas)
    write_json(manifests / "geometry.json", geometry)
    hashes = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*"))
              if p.is_file() and p.name != "outputs.json"}
    from src.utils.run_identity import code_identity, code_identity_at, git_head, runtime_identity as runtime_digest
    write_json(manifests / "outputs.json", hashes)
    # Completion marker, written last: binds these outputs to code and runtime. The
    # working-tree code must equal the committed code at git_head (audit A01).
    code = code_identity()
    head = git_head()
    write_json(manifests / "identity.json", {
        "stage": "prepare", "code": code, "runtime": runtime_digest(),
        "git_head": head, "code_equals_git_head": code == code_identity_at(head),
        "outputs_sha256": sha256(manifests / "outputs.json")})
    print("preparation complete", flush=True)


if __name__ == "__main__":
    main()
