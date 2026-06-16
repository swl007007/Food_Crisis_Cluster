# GeoRF m2 Cluster-Level Profile Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a representative GeoRF m2 local-cluster diagnostic pack: a 13-row descriptive profile table, inter-cluster similarity heatmaps, intra-cluster cohesion bars, long-format diagnostic CSVs, and a reviewer-facing note.

**Architecture:** Add one focused script, `scripts/analyze_georf_m2_cluster_profiles.py`, with small testable data functions and a CLI that writes artifacts to `final_artifacts_in_paper_updated/`. The script reads the refined GeoRF m2 cluster mapping, February Stage 3 predictions, FEWSNET shapefile attributes, and raw panel descriptors, then computes descriptive profiles and similarity/cohesion diagnostics without changing model training or predictions.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn cosine similarity, matplotlib/seaborn, geopandas only for shapefile attribute lookup. Unit tests use `unittest` and small in-memory DataFrames.

---

## File Structure

- Create `scripts/analyze_georf_m2_cluster_profiles.py`
  - CLI entrypoint and all reusable helper functions for this reviewer artifact.
  - Default inputs:
    - `result_partition_k40_compare_GF_fs1/refined/cluster_mapping_k40_nc13_m2_refined_contig3.csv`
    - `result_partition_k40_compare_GF_fs1/predictions_monthly.csv`
    - `C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm_phase_change.csv`
    - FEWSNET Admin Boundaries shapefile.
  - Default outputs:
    - `georf_m2_cluster_profile_table.csv`
    - `georf_m2_cluster_profile_similarity.png`
    - `georf_m2_cluster_profile_similarity_matrices.csv`
    - `georf_m2_cluster_profile_cohesion.csv`
    - `georf_m2_cluster_profile_note.md`

- Create `src/tests/test_georf_m2_cluster_profiles.py`
  - Tests pure functions only; no shapefile dependency.

- Modify `final_artifacts_in_paper_updated/README.md`
  - Register new profile table, figure, matrices, cohesion CSV, and note.

No commits should be made during execution unless the user explicitly asks, because the current worktree contains unrelated modified and untracked files.

---

### Task 1: Core Joins and February m2 Filtering

**Files:**
- Create: `src/tests/test_georf_m2_cluster_profiles.py`
- Create: `scripts/analyze_georf_m2_cluster_profiles.py`

- [ ] **Step 1: Write the failing import and join tests**

Add this test file:

```python
import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_georf_m2_cluster_profiles.py"
spec = importlib.util.spec_from_file_location("analyze_georf_m2_cluster_profiles", SCRIPT_PATH)
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)


class GeoRFM2ClusterProfilesTests(unittest.TestCase):
    def test_attach_clusters_normalizes_admin_codes(self):
        mapping = pd.DataFrame({
            "FEWSNET_admin_code": ["1", "2"],
            "cluster_id": [10, 11],
        })
        records = pd.DataFrame({
            "FEWSNET_admin_code": [1.0, 2.0, 3.0],
            "value": [4, 5, 6],
        })

        joined = profiles.attach_clusters(records, mapping)

        self.assertEqual(joined["cluster_id"].tolist(), [10, 11])
        self.assertEqual(joined["FEWSNET_admin_code"].tolist(), ["1", "2"])

    def test_filter_february_predictions_keeps_only_m2_target_months(self):
        predictions = pd.DataFrame({
            "FEWSNET_admin_code": ["1", "1", "1"],
            "month_start": ["2021-02-01", "2021-06-01", "2022-02-01"],
            "y_true": [0, 1, 1],
            "y_pred_partitioned": [0, 0, 1],
        })

        filtered = profiles.filter_february_predictions(predictions)

        self.assertEqual(filtered["target_month"].tolist(), ["2021-02", "2022-02"])
        self.assertEqual(filtered["y_true"].tolist(), [0, 1])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest src.tests.test_georf_m2_cluster_profiles
```

Expected: FAIL because `scripts/analyze_georf_m2_cluster_profiles.py` does not exist.

- [ ] **Step 3: Implement minimal core helpers**

Create `scripts/analyze_georf_m2_cluster_profiles.py` with:

```python
#!/usr/bin/env python3
"""Build GeoRF m2 cluster-level descriptive profiles and similarity diagnostics."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = REPO_ROOT / "result_partition_k40_compare_GF_fs1" / "refined" / "cluster_mapping_k40_nc13_m2_refined_contig3.csv"
DEFAULT_PREDICTIONS = REPO_ROOT / "result_partition_k40_compare_GF_fs1" / "predictions_monthly.csv"
DEFAULT_PANEL = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data"
    r"\FEWSNET_forecast_unadjusted_bm_phase_change.csv"
)
DEFAULT_SHAPEFILE = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome"
    r"\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_artifacts_in_paper_updated"

MARKET_COLUMNS = ["market_access", "market_distance"]
CONFLICT_COLUMNS = [
    "distance_to_nearest_acled",
    "event_count_battles",
    "event_count_explosions",
    "event_count_violence",
    "sum_fatalities_battles",
    "sum_fatalities_explosions",
    "sum_fatalities_violence",
    "event_count_battles_w5",
    "event_count_explosions_w5",
    "event_count_violence_w5",
    "sum_fatalities_battles_w5",
    "sum_fatalities_explosions_w5",
    "sum_fatalities_violence_w5",
    "event_count_battles_w10",
    "event_count_explosions_w10",
    "event_count_violence_w10",
    "sum_fatalities_battles_w10",
    "sum_fatalities_explosions_w10",
    "sum_fatalities_violence_w10",
]


def resolve_path(path: Path) -> Path:
    raw = str(path)
    if os.name == "nt":
        wsl_match = re.match(r"^[\\/]+mnt[\\/]+([A-Za-z])[\\/]+(.*)$", raw)
        if wsl_match:
            drive, rest = wsl_match.groups()
            windows_rest = rest.replace("/", "\\")
            return Path(f"{drive.upper()}:\\{windows_rest}")
    win_match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if win_match and os.name != "nt":
        drive, rest = win_match.groups()
        return Path("/mnt") / drive.lower() / rest.replace("\\", "/")
    return path.expanduser()


def normalize_admin_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_cluster_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(resolve_path(path), usecols=["FEWSNET_admin_code", "cluster_id"])
    df["FEWSNET_admin_code"] = normalize_admin_code(df["FEWSNET_admin_code"])
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="raise").astype(int)
    return df.drop_duplicates("FEWSNET_admin_code").reset_index(drop=True)


def attach_clusters(records: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    left = records.copy()
    right = mapping[["FEWSNET_admin_code", "cluster_id"]].copy()
    left["FEWSNET_admin_code"] = normalize_admin_code(left["FEWSNET_admin_code"])
    right["FEWSNET_admin_code"] = normalize_admin_code(right["FEWSNET_admin_code"])
    joined = left.merge(right, on="FEWSNET_admin_code", how="inner")
    return joined.reset_index(drop=True)


def filter_february_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    df["target_month"] = pd.to_datetime(df["month_start"], errors="coerce").dt.strftime("%Y-%m")
    df = df[df["target_month"].str.endswith("-02", na=False)].copy()
    df["y_true"] = pd.to_numeric(df["y_true"], errors="raise").astype(int)
    df["y_pred_partitioned"] = pd.to_numeric(df["y_pred_partitioned"], errors="raise").astype(int)
    return df.reset_index(drop=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m unittest src.tests.test_georf_m2_cluster_profiles
```

Expected: PASS.

---

### Task 2: Error Modes, Crisis Prevalence, and Cluster Profile Table

**Files:**
- Modify: `src/tests/test_georf_m2_cluster_profiles.py`
- Modify: `scripts/analyze_georf_m2_cluster_profiles.py`

- [ ] **Step 1: Add failing tests for error modes and profile table**

Append tests:

```python
    def test_add_error_modes_labels_tp_fp_fn_tn(self):
        df = pd.DataFrame({
            "y_true": [1, 0, 1, 0],
            "y_pred_partitioned": [1, 1, 0, 0],
        })

        labeled = profiles.add_error_modes(df)

        self.assertEqual(labeled["error_mode"].tolist(), ["TP", "FP", "FN", "TN"])

    def test_build_error_summary_returns_cluster_counts_and_shares(self):
        predictions = pd.DataFrame({
            "cluster_id": [1, 1, 1, 2],
            "y_true": [1, 0, 1, 0],
            "y_pred_partitioned": [1, 1, 0, 0],
        })

        summary = profiles.build_error_summary(predictions)
        row = summary[summary["cluster_id"].eq(1)].iloc[0]

        self.assertEqual(row["n_observations"], 3)
        self.assertEqual(row["tp_count"], 1)
        self.assertEqual(row["fp_count"], 1)
        self.assertEqual(row["fn_count"], 1)
        self.assertAlmostEqual(row["crisis_prevalence"], 2 / 3)
        self.assertEqual(row["main_error_mode"], "FN/FP")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest src.tests.test_georf_m2_cluster_profiles
```

Expected: FAIL because `add_error_modes` and `build_error_summary` are missing.

- [ ] **Step 3: Implement error-mode helpers**

Add:

```python
def add_error_modes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    y_true = pd.to_numeric(out["y_true"], errors="raise").astype(int)
    y_pred = pd.to_numeric(out["y_pred_partitioned"], errors="raise").astype(int)
    conditions = [
        y_true.eq(1) & y_pred.eq(1),
        y_true.eq(0) & y_pred.eq(1),
        y_true.eq(1) & y_pred.eq(0),
        y_true.eq(0) & y_pred.eq(0),
    ]
    out["error_mode"] = np.select(conditions, ["TP", "FP", "FN", "TN"], default="invalid")
    return out


def dominant_non_tn_mode(counts: dict[str, int]) -> str:
    non_tn = {key: counts.get(key, 0) for key in ("FN", "FP", "TP")}
    max_count = max(non_tn.values()) if non_tn else 0
    if max_count == 0:
        return "TN-dominant"
    winners = sorted([key for key, value in non_tn.items() if value == max_count])
    return "/".join(winners)


def build_error_summary(predictions_with_clusters: pd.DataFrame) -> pd.DataFrame:
    labeled = add_error_modes(predictions_with_clusters)
    rows = []
    for cluster_id, sub in labeled.groupby("cluster_id"):
        counts = sub["error_mode"].value_counts().to_dict()
        n = int(len(sub))
        row = {
            "cluster_id": int(cluster_id),
            "n_observations": n,
            "crisis_prevalence": float(sub["y_true"].mean()) if n else 0.0,
            "tp_count": int(counts.get("TP", 0)),
            "fp_count": int(counts.get("FP", 0)),
            "fn_count": int(counts.get("FN", 0)),
            "tn_count": int(counts.get("TN", 0)),
            "main_error_mode": dominant_non_tn_mode(counts),
        }
        for mode in ("TP", "FP", "FN", "TN"):
            row[f"{mode.lower()}_share"] = row[f"{mode.lower()}_count"] / n if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m unittest src.tests.test_georf_m2_cluster_profiles
```

Expected: PASS.

---

### Task 3: Market, Conflict, AEZ, Countries, and Regions Profiles

**Files:**
- Modify: `src/tests/test_georf_m2_cluster_profiles.py`
- Modify: `scripts/analyze_georf_m2_cluster_profiles.py`

- [ ] **Step 1: Add failing tests for descriptive profile helpers**

Append tests:

```python
    def test_dominant_aez_uses_largest_mean_one_hot_share(self):
        df = pd.DataFrame({
            "cluster_id": [1, 1, 2],
            "AEZ_10000": [1, 1, 0],
            "AEZ_20000": [0, 0, 1],
        })

        result = profiles.build_dominant_aez(df, ["AEZ_10000", "AEZ_20000"])

        self.assertEqual(result.loc[result["cluster_id"].eq(1), "dominant_AEZ"].iloc[0], "AEZ_10000")
        self.assertEqual(result.loc[result["cluster_id"].eq(1), "dominant_AEZ_share"].iloc[0], 1.0)

    def test_profile_label_uses_tertiles(self):
        self.assertEqual(profiles.tertile_label(0.8), "high")
        self.assertEqual(profiles.tertile_label(0.5), "moderate")
        self.assertEqual(profiles.tertile_label(0.1), "low")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest src.tests.test_georf_m2_cluster_profiles
```

Expected: FAIL for missing helpers.

- [ ] **Step 3: Implement profile helper functions**

Add:

```python
def available_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in df.columns]


def aez_columns(df: pd.DataFrame) -> list[str]:
    return sorted([column for column in df.columns if re.match(r"^AEZ_\\d+", str(column))])


def build_dominant_aez(panel_with_clusters: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(columns=["cluster_id", "dominant_AEZ", "dominant_AEZ_share"])
    means = panel_with_clusters.groupby("cluster_id")[columns].mean()
    rows = []
    for cluster_id, row in means.iterrows():
        dominant = row.astype(float).idxmax()
        rows.append({
            "cluster_id": int(cluster_id),
            "dominant_AEZ": str(dominant),
            "dominant_AEZ_share": float(row[dominant]),
        })
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def tertile_label(value: float) -> str:
    if value >= 2 / 3:
        return "high"
    if value >= 1 / 3:
        return "moderate"
    return "low"


def cluster_percentile(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series([0.5] * len(series), index=series.index)
    return series.rank(pct=True)


def build_market_profile(panel_with_clusters: pd.DataFrame) -> pd.DataFrame:
    cols = available_columns(panel_with_clusters, MARKET_COLUMNS)
    grouped = panel_with_clusters.groupby("cluster_id")[cols].mean().reset_index()
    if "market_access" in grouped:
        grouped["market_access_rank"] = cluster_percentile(grouped["market_access"])
    else:
        grouped["market_access_rank"] = 0.5
    if "market_distance" in grouped:
        grouped["market_distance_rank"] = cluster_percentile(grouped["market_distance"])
    else:
        grouped["market_distance_rank"] = 0.5
    grouped["market_access_profile"] = grouped.apply(
        lambda row: f"{tertile_label(row['market_access_rank'])} access / {tertile_label(row['market_distance_rank'])} distance",
        axis=1,
    )
    return grouped


def build_conflict_profile(panel_with_clusters: pd.DataFrame) -> pd.DataFrame:
    cols = available_columns(panel_with_clusters, CONFLICT_COLUMNS)
    grouped = panel_with_clusters.groupby("cluster_id")[cols].mean().reset_index()
    if cols:
        exposure = grouped[cols].fillna(0).sum(axis=1)
        grouped["conflict_exposure_rank"] = cluster_percentile(exposure)
    else:
        grouped["conflict_exposure_rank"] = 0.5
    grouped["conflict_exposure_profile"] = grouped["conflict_exposure_rank"].map(tertile_label)
    return grouped
```

- [ ] **Step 4: Add shapefile attribute loader**

Add:

```python
def load_region_lookup(shapefile_path: Path) -> pd.DataFrame:
    import geopandas as gpd
    from scripts.plot_region_class_prevalence import REGION_MAP

    gdf = gpd.read_file(resolve_path(shapefile_path))
    candidates = ("FEWSNET_admin_code", "admin_code", "adm_code", "FNID", "uid")
    found = next((column for column in candidates if column in gdf.columns), None)
    if found is None:
        raise ValueError(f"Admin-code column not found in shapefile. Tried: {candidates}")
    if found != "FEWSNET_admin_code":
        gdf = gdf.rename(columns={found: "FEWSNET_admin_code"})
    if "ADMIN0" not in gdf.columns:
        raise ValueError("ADMIN0 column not found in FEWSNET shapefile")
    lookup = gdf[["FEWSNET_admin_code", "ADMIN0"]].copy()
    lookup["FEWSNET_admin_code"] = normalize_admin_code(lookup["FEWSNET_admin_code"])
    lookup["region"] = lookup["ADMIN0"].map(REGION_MAP).fillna("Other")
    return lookup.drop_duplicates("FEWSNET_admin_code").reset_index(drop=True)


def build_country_region_summary(mapping: pd.DataFrame, region_lookup: pd.DataFrame) -> pd.DataFrame:
    merged = attach_clusters(region_lookup, mapping)
    rows = []
    for cluster_id, sub in merged.groupby("cluster_id"):
        countries = sorted(sub["ADMIN0"].dropna().unique().tolist())
        regions = sub["region"].value_counts()
        dominant_region = str(regions.index[0]) if not regions.empty else "Unknown"
        if len(countries) <= 8:
            country_text = "; ".join(countries)
        else:
            country_text = "; ".join(f"{region} ({count})" for region, count in regions.items())
        rows.append({
            "cluster_id": int(cluster_id),
            "n_polygons": int(sub["FEWSNET_admin_code"].nunique()),
            "countries_or_regions_included": country_text,
            "dominant_region": dominant_region,
        })
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
```

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -m unittest src.tests.test_georf_m2_cluster_profiles
```

Expected: PASS.

---

### Task 4: Similarity Matrices and Intra-Cluster Cohesion

**Files:**
- Modify: `src/tests/test_georf_m2_cluster_profiles.py`
- Modify: `scripts/analyze_georf_m2_cluster_profiles.py`

- [ ] **Step 1: Add failing tests for similarity and cohesion**

Append tests:

```python
    def test_similarity_matrix_returns_long_format_and_diagonal_one(self):
        df = pd.DataFrame({
            "cluster_id": [1, 1, 2, 2],
            "x": [1.0, 1.0, 0.0, 0.0],
            "y": [0.0, 0.0, 1.0, 1.0],
        })

        matrix = profiles.build_intercluster_similarity(df, ["x", "y"], "market")

        diag = matrix[matrix["cluster_i"].eq(matrix["cluster_j"])]
        self.assertTrue((diag["similarity"].round(6) == 1.0).all())
        self.assertEqual(set(matrix["profile_type"]), {"market"})

    def test_feature_cohesion_is_between_zero_and_one(self):
        df = pd.DataFrame({
            "cluster_id": [1, 1, 2, 2],
            "x": [1.0, 1.1, 0.0, 0.1],
            "y": [0.0, 0.1, 1.0, 1.1],
        })

        cohesion = profiles.build_feature_cohesion(df, ["x", "y"], "market")

        self.assertTrue(cohesion["cohesion"].between(0, 1).all())
        self.assertEqual(set(cohesion["profile_type"]), {"market"})

    def test_error_cohesion_uses_dominant_error_mode_share(self):
        df = pd.DataFrame({
            "cluster_id": [1, 1, 1, 2],
            "error_mode": ["TN", "TN", "FP", "FN"],
        })

        cohesion = profiles.build_error_cohesion(df)

        self.assertAlmostEqual(cohesion.loc[cohesion["cluster_id"].eq(1), "cohesion"].iloc[0], 2 / 3)
        self.assertAlmostEqual(cohesion.loc[cohesion["cluster_id"].eq(2), "cohesion"].iloc[0], 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest src.tests.test_georf_m2_cluster_profiles
```

Expected: FAIL for missing similarity/cohesion helpers.

- [ ] **Step 3: Implement standardization and cosine helpers**

Add:

```python
def standardized_feature_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median(numeric_only=True)).fillna(0.0)
    std = values.std(ddof=0).replace(0, 1.0)
    return (values - values.mean()) / std


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0 if np.linalg.norm(a - b) == 0 else 0.0
    return float(np.dot(a, b) / denom)


def build_intercluster_similarity(df: pd.DataFrame, columns: list[str], profile_type: str) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(columns=["profile_type", "cluster_i", "cluster_j", "similarity"])
    features = standardized_feature_frame(df, columns)
    tmp = pd.concat([df[["cluster_id"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    centroids = tmp.groupby("cluster_id")[columns].mean().sort_index()
    rows = []
    for cluster_i, vec_i in centroids.iterrows():
        for cluster_j, vec_j in centroids.iterrows():
            rows.append({
                "profile_type": profile_type,
                "cluster_i": int(cluster_i),
                "cluster_j": int(cluster_j),
                "similarity": cosine(vec_i.to_numpy(dtype=float), vec_j.to_numpy(dtype=float)),
            })
    return pd.DataFrame(rows)


def build_feature_cohesion(df: pd.DataFrame, columns: list[str], profile_type: str) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(columns=["profile_type", "cluster_id", "cohesion"])
    features = standardized_feature_frame(df, columns)
    tmp = pd.concat([df[["cluster_id"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    centroids = tmp.groupby("cluster_id")[columns].mean()
    rows = []
    for cluster_id, sub in tmp.groupby("cluster_id"):
        centroid = centroids.loc[cluster_id].to_numpy(dtype=float)
        similarities = [
            max(0.0, min(1.0, (cosine(row.to_numpy(dtype=float), centroid) + 1.0) / 2.0))
            for _, row in sub[columns].iterrows()
        ]
        rows.append({
            "profile_type": profile_type,
            "cluster_id": int(cluster_id),
            "cohesion": float(np.mean(similarities)) if similarities else 0.0,
        })
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def build_error_similarity(error_summary: pd.DataFrame) -> pd.DataFrame:
    columns = ["tp_share", "fp_share", "fn_share", "tn_share"]
    rows = []
    data = error_summary.set_index("cluster_id")[columns].sort_index()
    for cluster_i, vec_i in data.iterrows():
        for cluster_j, vec_j in data.iterrows():
            rows.append({
                "profile_type": "error_mode",
                "cluster_i": int(cluster_i),
                "cluster_j": int(cluster_j),
                "similarity": cosine(vec_i.to_numpy(dtype=float), vec_j.to_numpy(dtype=float)),
            })
    return pd.DataFrame(rows)


def build_error_cohesion(labeled_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, sub in labeled_predictions.groupby("cluster_id"):
        shares = sub["error_mode"].value_counts(normalize=True)
        rows.append({
            "profile_type": "error_mode",
            "cluster_id": int(cluster_id),
            "cohesion": float(shares.max()) if not shares.empty else 0.0,
        })
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m unittest src.tests.test_georf_m2_cluster_profiles
```

Expected: PASS.

---

### Task 5: Figure, Note, CLI, and Artifact Writing

**Files:**
- Modify: `scripts/analyze_georf_m2_cluster_profiles.py`
- Modify: `final_artifacts_in_paper_updated/README.md`

- [ ] **Step 1: Implement profile table assembly**

Add:

```python
def build_cluster_profile_table(
    mapping: pd.DataFrame,
    region_lookup: pd.DataFrame,
    panel_with_clusters: pd.DataFrame,
    predictions_with_clusters: pd.DataFrame,
) -> pd.DataFrame:
    country_region = build_country_region_summary(mapping, region_lookup)
    error_summary = build_error_summary(predictions_with_clusters)
    dominant_aez = build_dominant_aez(panel_with_clusters, aez_columns(panel_with_clusters))
    market = build_market_profile(panel_with_clusters)
    conflict = build_conflict_profile(panel_with_clusters)
    table = country_region.merge(error_summary, on="cluster_id", how="left")
    table = table.merge(dominant_aez, on="cluster_id", how="left")
    table = table.merge(market[["cluster_id", "market_access_profile"]], on="cluster_id", how="left")
    table = table.merge(conflict[["cluster_id", "conflict_exposure_profile"]], on="cluster_id", how="left")
    table["crisis_prevalence"] = table["crisis_prevalence"].fillna(0.0)
    return table.sort_values("cluster_id").reset_index(drop=True)
```

- [ ] **Step 2: Implement plotting**

Add:

```python
def plot_similarity_figure(similarities: pd.DataFrame, cohesion: pd.DataFrame, output_path: Path, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    profile_order = [
        ("market_access", "Market access"),
        ("conflict_exposure", "Conflict exposure"),
        ("error_mode", "Error mode"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 14), gridspec_kw={"width_ratios": [3.2, 1.2]})
    for row_idx, (profile_type, title) in enumerate(profile_order):
        sub = similarities[similarities["profile_type"].eq(profile_type)]
        matrix = sub.pivot(index="cluster_i", columns="cluster_j", values="similarity").sort_index().sort_index(axis=1)
        sns.heatmap(matrix, ax=axes[row_idx, 0], vmin=-1, vmax=1, cmap="vlag", square=True, cbar=row_idx == 0)
        axes[row_idx, 0].set_title(f"{title}: inter-cluster similarity", fontweight="bold")
        axes[row_idx, 0].set_xlabel("Cluster")
        axes[row_idx, 0].set_ylabel("Cluster")

        coh = cohesion[cohesion["profile_type"].eq(profile_type)].sort_values("cluster_id")
        axes[row_idx, 1].barh(coh["cluster_id"].astype(str), coh["cohesion"], color="#4c78a8")
        axes[row_idx, 1].set_xlim(0, 1)
        axes[row_idx, 1].set_title("Within-cluster\ncohesion", fontweight="bold")
        axes[row_idx, 1].set_xlabel("Cohesion")
        axes[row_idx, 1].invert_yaxis()
    fig.suptitle("GeoRF m2 Cluster Profiles: Similarity and Cohesion", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 3: Implement note writer**

Add:

```python
def write_note(output_dir: Path, profile_table: pd.DataFrame) -> Path:
    note_path = output_dir / "georf_m2_cluster_profile_note.md"
    note_path.write_text(
        "\n".join([
            "# GeoRF m2 Cluster-Level Profile Diagnostics",
            "",
            "中文审查说明：",
            "",
            "该 appendix 只汇报一个代表性 GeoRF m2 refined local-model partition 的描述性 cluster profiles。",
            "每个 cluster 的 market access、conflict exposure、AEZ、region/country composition、crisis prevalence 和 error mode 均从现有数据与 Stage 3 prediction outputs 汇总。",
            "Similarity heatmaps 使用标准化描述性 profile 或 observed error composition 计算，并不来自 model internals。",
            "因此这些结果用于刻画 local model domains，不解释为 feature importance 或 causal drivers。",
            "",
            f"该表共包含 {len(profile_table)} 个 GeoRF m2 clusters。",
            "",
            "Appendix text (English):",
            "",
            "We summarize cluster-level descriptive profiles for one representative GeoRF m2 local-model partition. ",
            "The profiles include country/region composition, crisis prevalence, dominant AEZ, market-access characteristics, conflict exposure, and observed error modes. ",
            "Inter-cluster similarities are computed from standardized descriptive profiles or observed error-mode composition, while the paired cohesion bars report within-cluster consistency. ",
            "These summaries characterize the local-model domains and should not be interpreted as feature importance or causal attribution.",
            "",
        ]),
        encoding="utf-8",
    )
    return note_path
```

- [ ] **Step 4: Implement `main()` orchestration**

Add below `parse_args`:

```python
def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_cluster_mapping(args.mapping)
    predictions = pd.read_csv(resolve_path(args.predictions))
    feb_predictions = filter_february_predictions(predictions)
    predictions_with_clusters = attach_clusters(feb_predictions, mapping)
    labeled_predictions = add_error_modes(predictions_with_clusters)

    panel = pd.read_csv(resolve_path(args.panel))
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel[panel["date"].dt.month.eq(2)].copy()
    panel_with_clusters = attach_clusters(panel, mapping)

    region_lookup = load_region_lookup(args.shapefile)
    profile_table = build_cluster_profile_table(mapping, region_lookup, panel_with_clusters, predictions_with_clusters)

    market_cols = available_columns(panel_with_clusters, MARKET_COLUMNS)
    conflict_cols = available_columns(panel_with_clusters, CONFLICT_COLUMNS)
    error_summary = build_error_summary(predictions_with_clusters)

    similarities = pd.concat([
        build_intercluster_similarity(panel_with_clusters, market_cols, "market_access"),
        build_intercluster_similarity(panel_with_clusters, conflict_cols, "conflict_exposure"),
        build_error_similarity(error_summary),
    ], ignore_index=True)
    cohesion = pd.concat([
        build_feature_cohesion(panel_with_clusters, market_cols, "market_access"),
        build_feature_cohesion(panel_with_clusters, conflict_cols, "conflict_exposure"),
        build_error_cohesion(labeled_predictions),
    ], ignore_index=True)

    profile_table.to_csv(output_dir / "georf_m2_cluster_profile_table.csv", index=False)
    similarities.to_csv(output_dir / "georf_m2_cluster_profile_similarity_matrices.csv", index=False)
    cohesion.to_csv(output_dir / "georf_m2_cluster_profile_cohesion.csv", index=False)
    plot_similarity_figure(similarities, cohesion, output_dir / "georf_m2_cluster_profile_similarity.png", args.dpi)
    write_note(output_dir, profile_table)

    print(f"Wrote GeoRF m2 cluster profile artifacts to {output_dir}")
    print(f"Clusters: {profile_table['cluster_id'].nunique()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Update README**

Add under Diagnostic Data in `final_artifacts_in_paper_updated/README.md`:

```markdown
- `georf_m2_cluster_profile_table.csv`: descriptive cluster-level profiles for
  the representative GeoRF m2 refined local-model partition.
- `georf_m2_cluster_profile_similarity_matrices.csv`: long-format
  inter-cluster similarity matrices for market access, conflict exposure, and
  error-mode composition.
- `georf_m2_cluster_profile_cohesion.csv`: within-cluster cohesion values
  paired with the similarity heatmaps.
- `georf_m2_cluster_profile_note.md`: Chinese reviewer-facing note and English
  appendix text for the cluster profile diagnostics.
```

Add under Core Figures:

```markdown
- `georf_m2_cluster_profile_similarity.png`: market-access, conflict-exposure,
  and error-mode inter-cluster similarity heatmaps with within-cluster cohesion
  bars for the representative GeoRF m2 local-model partition.
```

---

### Task 6: Verification and Smoke Run

**Files:**
- Verify all created/modified files.

- [ ] **Step 1: Run unit tests**

Run:

```bash
python3 -m unittest src.tests.test_georf_m2_cluster_profiles
```

Expected: all tests pass.

- [ ] **Step 2: Compile script**

Run:

```bash
python3 -m py_compile scripts/analyze_georf_m2_cluster_profiles.py
```

Expected: exit code 0.

- [ ] **Step 3: Run full artifact generation with Windows Python**

Use Windows Python because WSL `python3` may not have geopandas:

```bash
/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe scripts/analyze_georf_m2_cluster_profiles.py
```

Expected:

```text
Wrote GeoRF m2 cluster profile artifacts to .../final_artifacts_in_paper_updated
Clusters: 13
```

- [ ] **Step 4: Inspect generated artifacts**

Run:

```bash
ls -lh final_artifacts_in_paper_updated/georf_m2_cluster_profile_*
python3 - <<'PY'
import pandas as pd
base = 'final_artifacts_in_paper_updated'
print(pd.read_csv(f'{base}/georf_m2_cluster_profile_table.csv').shape)
print(pd.read_csv(f'{base}/georf_m2_cluster_profile_similarity_matrices.csv').groupby('profile_type').size())
print(pd.read_csv(f'{base}/georf_m2_cluster_profile_cohesion.csv').groupby('profile_type').size())
PY
```

Expected:
- profile table has 13 rows;
- similarity matrices have 169 rows per profile type;
- cohesion has 13 rows per profile type;
- PNG exists and is non-empty.

- [ ] **Step 5: Visual sanity check**

Open:

```text
final_artifacts_in_paper_updated/georf_m2_cluster_profile_similarity.png
```

Expected:
- three heatmap rows are visible;
- cohesion bars are visible;
- cluster labels do not overlap badly;
- title states GeoRF m2 cluster profiles.

- [ ] **Step 6: Check whitespace and status**

Run:

```bash
git diff --check
git status --short -- scripts/analyze_georf_m2_cluster_profiles.py src/tests/test_georf_m2_cluster_profiles.py final_artifacts_in_paper_updated/README.md final_artifacts_in_paper_updated/georf_m2_cluster_profile_*
```

Expected:
- `git diff --check` exits 0;
- status shows only the intended new script, test, README edit, and generated artifacts.

---

## Self-Review Against Spec

- Scope is GeoRF m2 refined mapping only: covered in defaults and output names.
- 13 cluster profile rows: covered by `build_cluster_profile_table` and smoke check.
- Countries/regions, crisis prevalence, dominant AEZ, market access, conflict exposure, error modes: covered in Tasks 2 and 3.
- Inter-cluster differences: covered by 13x13 similarity matrices in Task 4.
- Intra-cluster similarity: covered by cohesion bars and cohesion CSV in Task 4.
- Reviewer-safe framing: covered by `write_note`.
- No model retraining or Stage 3 changes: plan only reads existing artifacts.
