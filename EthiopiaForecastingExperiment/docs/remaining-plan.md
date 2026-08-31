# Ethiopia Baseline Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a forecast-time-safe, temporally validated, reproducible Ethiopia baseline and the formal contract required before a separate CDS-weather implementation plan may begin.

**Architecture:** Keep all experimental code, tests, configurations, manifests, and outputs under `EthiopiaForecastingExperiment/`; treat production code and released paper artifacts as read-only reference providers. First characterize the current baseline, then evaluate pre-registered corrections on matched temporal folds, freeze the selected Ethiopia baseline, and finally approve a provider-neutral CDS forecast-cube contract.

**Tech Stack:** Python 3.12, pandas, NumPy, scikit-learn, standard-library `unittest`, JSON/CSV/Markdown manifests, and existing read-only GeoRF artifacts where compatibility is proven.

**Spec:** `EthiopiaForecastingExperiment/docs/contract-draft.md`

**Status:** Recorded for handoff on 2026-08-31. No step is authorized for execution by this document alone.

## Global Constraints

- Geography is Ethiopia only, derived from `ISO3 == "ETH"` in the authoritative FEWS NET panel.
- `FEWSNET_admin_code` is the canonical spatial key; reference `area_id` values are forbidden as join keys.
- Production horizons remain exactly 4, 8, and 12 months.
- Existing source data, production entry points, release archives, paper artifacts, and reproducibility manifests are read-only.
- Every feature must be evaluated against a declared forecast-origin availability rule.
- Model selection and threshold selection use training/validation data only; target test labels never select a candidate.
- Forward prediction differences are not performance differences.
- CDS acquisition remains blocked until the Ethiopia baseline is reviewed and frozen.
- Before modifying any production symbol in a later task, run GitNexus upstream impact analysis and stop on HIGH or CRITICAL risk.
- Use experiment-specific paths and names; never overwrite `result_GeoRF*`, `GeoRFExperiment/`, archived releases, or paper artifacts.

---

## Planned File Structure

```text
EthiopiaForecastingExperiment/
├── __init__.py
├── configs/
│   ├── baseline_candidates.json
│   └── frozen_baseline.json
├── contracts/
│   └── cds_forecast_cube.schema.json
├── docs/
│   ├── decisions/
│   │   ├── 0001-baseline-audit-scope.md
│   │   └── 0002-baseline-freeze.md
│   └── cds-product-decision.md
├── manifests/
│   ├── cohort_manifest.json
│   └── baseline_run_manifest.json
├── outputs/
│   └── baseline_audit/
├── src/
│   ├── __init__.py
│   ├── cohort.py
│   ├── temporal_contract.py
│   ├── baseline_audit.py
│   ├── freeze_baseline.py
│   └── cds_contract.py
└── tests/
    ├── __init__.py
    ├── test_cohort.py
    ├── test_temporal_contract.py
    ├── test_baseline_audit.py
    ├── test_freeze_baseline.py
    └── test_cds_contract.py
```

Generated manifests and outputs remain uncommitted until their provenance and storage policy are approved. Source, tests, contracts, configs, and reviewed decision documents are eligible for version control.

### Task 1: Freeze the Ethiopia Cohort and Source Snapshot

**Files:**
- Create: `EthiopiaForecastingExperiment/__init__.py`
- Create: `EthiopiaForecastingExperiment/src/__init__.py`
- Create: `EthiopiaForecastingExperiment/tests/__init__.py`
- Create: `EthiopiaForecastingExperiment/src/cohort.py`
- Create: `EthiopiaForecastingExperiment/tests/test_cohort.py`
- Generate: `EthiopiaForecastingExperiment/manifests/cohort_manifest.json`

**Interfaces:**
- Consumes: authoritative FEWS NET CSV path.
- Produces: `load_ethiopia_panel(source_csv: Path) -> pandas.DataFrame` and `build_cohort_manifest(panel: pandas.DataFrame, source_csv: Path) -> dict[str, object]`.

- [ ] **Step 1: Write the failing cohort tests**

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from EthiopiaForecastingExperiment.src.cohort import (
    build_cohort_manifest,
    load_ethiopia_panel,
)


class CohortTests(unittest.TestCase):
    def test_filters_eth_and_preserves_fewsnet_key(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.csv"
            pd.DataFrame({
                "ISO3": ["ETH", "KEN", "ETH"],
                "FEWSNET_admin_code": [11, 22, 33],
                "date": ["2024-01", "2024-01", "2024-02"],
            }).to_csv(path, index=False)
            result = load_ethiopia_panel(path)
            self.assertEqual(result["ISO3"].unique().tolist(), ["ETH"])
            self.assertEqual(set(result["FEWSNET_admin_code"]), {11, 33})

    def test_manifest_records_key_counts_and_date_bounds(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.csv"
            panel = pd.DataFrame({
                "ISO3": ["ETH", "ETH"],
                "FEWSNET_admin_code": [11, 33],
                "date": ["2024-01", "2024-02"],
            })
            panel.to_csv(path, index=False)
            manifest = build_cohort_manifest(panel, path)
            self.assertEqual(manifest["iso3"], "ETH")
            self.assertEqual(manifest["admin_code_count"], 2)
            self.assertEqual(manifest["date_min"], "2024-01")
            self.assertEqual(manifest["date_max"], "2024-02")
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
python3 -m unittest EthiopiaForecastingExperiment.tests.test_cohort -v
```

Expected: FAIL because `EthiopiaForecastingExperiment.src.cohort` does not exist.

- [ ] **Step 3: Implement strict cohort loading and manifest construction**

The implementation must require `ISO3`, `FEWSNET_admin_code`, and `date`; reject null admin codes; filter only exact `ETH`; normalize dates to monthly `YYYY-MM`; reject duplicate `(FEWSNET_admin_code, date)` keys; calculate the source SHA-256; and sort by admin code and month.

```python
from hashlib import sha256
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"ISO3", "FEWSNET_admin_code", "date"}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ethiopia_panel(source_csv: Path) -> pd.DataFrame:
    panel = pd.read_csv(source_csv)
    missing = REQUIRED_COLUMNS.difference(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    panel = panel.loc[panel["ISO3"].eq("ETH")].copy()
    if panel["FEWSNET_admin_code"].isna().any():
        raise ValueError("Null FEWSNET_admin_code in Ethiopia cohort")
    panel["date"] = pd.PeriodIndex(panel["date"], freq="M").astype(str)
    keys = ["FEWSNET_admin_code", "date"]
    if panel.duplicated(keys).any():
        raise ValueError("Duplicate Ethiopia admin-month keys")
    return panel.sort_values(keys).reset_index(drop=True)


def build_cohort_manifest(panel: pd.DataFrame, source_csv: Path) -> dict[str, object]:
    return {
        "iso3": "ETH",
        "source_path": str(source_csv),
        "source_sha256": _file_sha256(source_csv),
        "row_count": int(len(panel)),
        "admin_code_count": int(panel["FEWSNET_admin_code"].nunique()),
        "date_min": str(panel["date"].min()),
        "date_max": str(panel["date"].max()),
        "month_count": int(panel["date"].nunique()),
    }
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
python3 -m unittest EthiopiaForecastingExperiment.tests.test_cohort -v
```

Expected: both tests PASS.

- [ ] **Step 5: Run the authoritative read-only cohort snapshot**

Expected invariant from the 2026-08-31 inspection: 187,200 rows, 1,040 admin codes, 180 months, 2010-01 through 2024-12. Stop if any value differs; do not silently update the contract.

- [ ] **Step 6: Commit the reviewed cohort utility and test**

```bash
git add EthiopiaForecastingExperiment/__init__.py EthiopiaForecastingExperiment/src/__init__.py EthiopiaForecastingExperiment/tests/__init__.py EthiopiaForecastingExperiment/src/cohort.py EthiopiaForecastingExperiment/tests/test_cohort.py
git commit -m "add Ethiopia cohort contract"
```

### Task 2: Characterize Forecast-Time Availability and Temporal Boundaries

**Files:**
- Create: `EthiopiaForecastingExperiment/src/temporal_contract.py`
- Create: `EthiopiaForecastingExperiment/tests/test_temporal_contract.py`
- Generate: `EthiopiaForecastingExperiment/outputs/baseline_audit/availability_audit.csv`

**Interfaces:**
- Consumes: target month, horizon, feature names, and declared availability classes.
- Produces: `forecast_origin(target_month: str, horizon: int) -> pandas.Period`, `target_relative_lag(horizon: int, lead: int) -> int`, and `audit_feature_availability(feature_names: list[str], horizon: int, policy: dict[str, str]) -> pandas.DataFrame`.

- [ ] **Step 1: Write failing tests for origin, lead, lag, and forbidden contemporaneous features**

```python
import unittest

from EthiopiaForecastingExperiment.src.temporal_contract import (
    audit_feature_availability,
    forecast_origin,
    target_relative_lag,
    validate_lead_path,
)


class TemporalContractTests(unittest.TestCase):
    def test_horizon_8_leads_map_to_target_lags_8_through_2(self):
        self.assertEqual(
            [target_relative_lag(8, lead) for lead in range(7)],
            [8, 7, 6, 5, 4, 3, 2],
        )

    def test_origin_is_calendar_month_subtraction(self):
        self.assertEqual(str(forecast_origin("2027-01", 8)), "2026-05")

    def test_lead_zero_through_six_is_seven_values(self):
        self.assertEqual(validate_lead_path(range(7)), tuple(range(7)))

    def test_observed_target_month_weather_is_not_forecast_safe(self):
        audit = audit_feature_availability(
            ["Rainf_f_tavg_mean"],
            horizon=4,
            policy={"Rainf_f_tavg_mean": "observed_only"},
        )
        self.assertFalse(bool(audit.loc[0, "eligible"]))
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
python3 -m unittest EthiopiaForecastingExperiment.tests.test_temporal_contract -v
```

Expected: FAIL because the temporal-contract module does not exist.

- [ ] **Step 3: Implement calendar-aware month arithmetic**

Use `pandas.Period(freq="M")`; never implement a month lag as an unvalidated row shift. Encode the agreed windows: horizon 4 permits leads 0-4, while horizons 8 and 12 permit leads 0-6.

```python
import pandas as pd


ALLOWED_LEADS = {
    4: tuple(range(5)),
    8: tuple(range(7)),
    12: tuple(range(7)),
}


def forecast_origin(target_month: str, horizon: int) -> pd.Period:
    if horizon not in ALLOWED_LEADS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    return pd.Period(target_month, freq="M") - horizon


def target_relative_lag(horizon: int, lead: int) -> int:
    if lead not in ALLOWED_LEADS.get(horizon, ()):
        raise ValueError(f"Lead {lead} is not allowed for horizon {horizon}")
    return horizon - lead


def validate_lead_path(leads) -> tuple[int, ...]:
    resolved = tuple(int(value) for value in leads)
    if resolved != tuple(range(7)):
        raise ValueError("Expected inclusive lead path 0 through 6")
    return resolved


def audit_feature_availability(
    feature_names: list[str],
    horizon: int,
    policy: dict[str, str],
) -> pd.DataFrame:
    allowed_classes = {"static", "historical_at_origin", "forecast_at_origin"}
    rows = []
    for name in feature_names:
        availability_class = policy.get(name, "undeclared")
        rows.append({
            "feature_name": name,
            "horizon": horizon,
            "availability_class": availability_class,
            "eligible": availability_class in allowed_classes,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Add a read-only characterization of current production behavior**

Record, without repairing production code, that the current feature matrix retains contemporaneous time-varying columns, the configured 36-month split currently yields an effective 35-month interval, and `feature_engineering_3()` returns inside its outer feature loop. The audit output must label each item `observed_code_behavior`, not `confirmed_overfitting`.

- [ ] **Step 5: Run the temporal tests**

Run:

```bash
python3 -m unittest EthiopiaForecastingExperiment.tests.test_temporal_contract -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit the temporal contract**

```bash
git add EthiopiaForecastingExperiment/src/temporal_contract.py EthiopiaForecastingExperiment/tests/test_temporal_contract.py
git commit -m "define Ethiopia forecast-time contract"
```

### Task 3: Approve the Baseline Audit Scope

**Files:**
- Create: `EthiopiaForecastingExperiment/docs/decisions/0001-baseline-audit-scope.md`
- Create: `EthiopiaForecastingExperiment/configs/baseline_candidates.json`

**Interfaces:**
- Consumes: advisor decision and Tasks 1-2 evidence.
- Produces: an accepted model-comparison matrix and pre-registered candidate definitions.

- [ ] **Step 1: Present the recommended paired audit for approval**

Recommended comparison on identical Ethiopia rows and folds:

1. pooled Random Forest;
2. fixed-partition GeoRF using the current frozen global partitions where Ethiopia mappings are complete;
3. each of the above with forecast-time-safe feature availability;
4. regularized candidates selected on temporal validation only.

This pairing distinguishes base-learner overfitting from partition-induced overfitting. If the user rejects paired scope, record the chosen narrower scope and its inferential limitation.

- [ ] **Step 2: Record the accepted decision**

The decision document must state model families, partition source and hashes, evaluation months, metrics, seed policy, candidate-selection rule, and forbidden uses of test labels. Do not continue without explicit acceptance.

- [ ] **Step 3: Write the candidate configuration**

The initial recommended candidate IDs are:

- `legacy_reproduction`: characterize current behavior without claiming operational validity;
- `availability_safe`: remove target-month covariates unavailable at the forecast origin;
- `availability_safe_window36`: additionally use a true 36-month calendar window;
- `regularized_rf`: select `max_depth`, `min_samples_leaf`, and `max_features` only through temporal validation;
- `partition_support_guard`: use a pooled fallback when a branch or cluster fails the accepted minimum-support rule.

Numeric regularization grids and minimum-support thresholds must be written into the accepted decision before execution; they may not be chosen after test results are viewed.

Use this pre-registration template as the recommended starting point; Decision 0001 must either accept it verbatim or record the approved changes before any test-fold result is inspected:

```json
{
  "horizons": [4, 8, 12],
  "evaluation_months": [
    "2021-02", "2021-06", "2021-10",
    "2022-02", "2022-06", "2022-10",
    "2023-02", "2023-06", "2023-10",
    "2024-02", "2024-06", "2024-10"
  ],
  "validation_months": 6,
  "random_seeds": [5, 17, 29],
  "selection_metric": "validation_average_precision",
  "regularized_rf_grid": {
    "max_depth": [4, 8, 12, null],
    "min_samples_leaf": [1, 5, 10, 20],
    "max_features": ["sqrt", 0.5, 1.0]
  },
  "partition_support_guard": {
    "minimum_training_rows": 100,
    "minimum_positive_rows": 10,
    "fallback": "pooled_rf"
  }
}
```

- [ ] **Step 4: Commit the accepted decision and configuration**

```bash
git add EthiopiaForecastingExperiment/docs/decisions/0001-baseline-audit-scope.md EthiopiaForecastingExperiment/configs/baseline_candidates.json
git commit -m "record Ethiopia baseline audit scope"
```

### Task 4: Implement the Approved Baseline Audit Runner

**Files:**
- Create: `EthiopiaForecastingExperiment/src/baseline_audit.py`
- Create: `EthiopiaForecastingExperiment/tests/test_baseline_audit.py`
- Generate: `EthiopiaForecastingExperiment/manifests/baseline_run_manifest.json`
- Generate: `EthiopiaForecastingExperiment/outputs/baseline_audit/metrics_by_fold.csv`
- Generate: `EthiopiaForecastingExperiment/outputs/baseline_audit/predictions_by_fold.csv`

**Interfaces:**
- Consumes: frozen cohort, candidate configuration, approved partition providers, and monthly target rows.
- Produces: `run_candidate(candidate_id: str, panel: pandas.DataFrame, folds: list[dict[str, str]]) -> tuple[pandas.DataFrame, pandas.DataFrame]` and `summarize_generalization(predictions: pandas.DataFrame) -> pandas.DataFrame`.

- [ ] **Step 1: Write synthetic tests proving test-label isolation**

Tests must monkey-patch the selector so access to test labels raises an exception, then verify candidate and threshold selection complete using training/validation labels alone. Add deterministic synthetic checks for class-1 precision, recall, F1, average precision, balanced accuracy, Brier score, calibration bins, support, and generalization gap.

```python
import unittest

import numpy as np

from EthiopiaForecastingExperiment.src.baseline_audit import (
    compute_binary_metrics,
    select_threshold,
)


class BaselineAuditTests(unittest.TestCase):
    def test_threshold_selection_uses_validation_only(self):
        y_val = np.array([0, 0, 1, 1])
        p_val = np.array([0.1, 0.4, 0.6, 0.9])
        threshold = select_threshold(y_val, p_val, candidates=(0.3, 0.5, 0.7))
        self.assertEqual(threshold, 0.5)

    def test_metrics_are_deterministic(self):
        y_true = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = compute_binary_metrics(y_true, probability, threshold=0.5)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["support"], 4)
        self.assertAlmostEqual(metrics["f1_class1"], 1.0)
```

- [ ] **Step 2: Run the tests and verify failure before implementation**

Run:

```bash
python3 -m unittest EthiopiaForecastingExperiment.tests.test_baseline_audit -v
```

Expected: FAIL because the audit runner does not exist.

- [ ] **Step 3: Implement deterministic fold execution**

Use the accepted 2021-2024 release-month evaluation set unless Decision 0001 explicitly changes it. Record training months, validation months, test month, horizon, random seed, feature list hash, cohort hash, partition hash, threshold, model parameters, package versions, and row counts for every fold.

Implement `select_threshold(y_validation, probability_validation, candidates)` with no test-label argument. Implement `compute_binary_metrics()` with scikit-learn metric functions and explicit confusion-matrix counts. `run_candidate()` must receive already separated training, validation, and test frames so candidate selection cannot query target-test labels through the source panel.

- [ ] **Step 4: Implement overfitting evidence outputs**

Report train/validation/test metric gaps, calibration drift, fold-to-fold dispersion, admin and class support, cluster support, and candidate-selection frequency. Label overfitting as supported only when the accepted Decision 0001 threshold is met; otherwise report the diagnostics without the label.

- [ ] **Step 5: Run focused and full experiment tests**

```bash
python3 -m unittest discover -s EthiopiaForecastingExperiment/tests -v
```

Expected: all experiment tests PASS.

- [ ] **Step 6: Run one approved smoke fold**

Run a single Ethiopia target month and horizon into a temporary output directory. Verify unique `(FEWSNET_admin_code, target_month, model, candidate_id)` prediction keys and confirm that no production path changed.

- [ ] **Step 7: Commit the runner and tests**

```bash
git add EthiopiaForecastingExperiment/src/baseline_audit.py EthiopiaForecastingExperiment/tests/test_baseline_audit.py
git commit -m "add Ethiopia baseline audit runner"
```

### Task 5: Evaluate Corrections and Freeze the Baseline

**Files:**
- Create: `EthiopiaForecastingExperiment/src/freeze_baseline.py`
- Create: `EthiopiaForecastingExperiment/tests/test_freeze_baseline.py`
- Create: `EthiopiaForecastingExperiment/docs/decisions/0002-baseline-freeze.md`
- Generate: `EthiopiaForecastingExperiment/configs/frozen_baseline.json`

**Interfaces:**
- Consumes: completed audit metrics, predictions, manifests, and accepted selection rule.
- Produces: `select_baseline(metrics: pandas.DataFrame, rule: dict[str, object]) -> str` and an immutable frozen-baseline configuration.

- [ ] **Step 1: Write a failing test for deterministic selection**

The fixture must contain one candidate with higher test F1 but worse validation performance and one candidate with the best accepted validation score. Assert that selection chooses the validation winner and never reads the test metric column.

```python
import unittest

import pandas as pd

from EthiopiaForecastingExperiment.src.freeze_baseline import select_baseline


class FreezeBaselineTests(unittest.TestCase):
    def test_validation_winner_is_selected_despite_lower_test_f1(self):
        metrics = pd.DataFrame([
            {
                "candidate_id": "unsafe_test_winner",
                "contract_valid": False,
                "validation_average_precision": 0.70,
                "test_f1": 0.95,
                "complexity_rank": 1,
            },
            {
                "candidate_id": "safe_validation_winner",
                "contract_valid": True,
                "validation_average_precision": 0.80,
                "test_f1": 0.75,
                "complexity_rank": 2,
            },
        ])
        selected = select_baseline(
            metrics,
            rule={
                "metric": "validation_average_precision",
                "tie_break": "lowest_complexity_rank",
            },
        )
        self.assertEqual(selected, "safe_validation_winner")
```

- [ ] **Step 2: Implement the accepted selection rule**

Selection must enforce availability safety first, then apply the pre-registered validation metric and complexity tie-break. A candidate that violates the feature-availability contract is ineligible regardless of apparent test performance.

```python
def select_baseline(metrics, rule: dict[str, object]) -> str:
    eligible = metrics.loc[metrics["contract_valid"].eq(True)].copy()
    if eligible.empty:
        raise ValueError("No contract-valid baseline candidate")
    metric = str(rule["metric"])
    best_score = eligible[metric].max()
    finalists = eligible.loc[eligible[metric].eq(best_score)]
    selected = finalists.sort_values("complexity_rank").iloc[0]
    return str(selected["candidate_id"])
```

- [ ] **Step 3: Run all experiment tests**

```bash
python3 -m unittest discover -s EthiopiaForecastingExperiment/tests -v
```

Expected: all tests PASS.

- [ ] **Step 4: Write the baseline-freeze decision**

Record the selected candidate, rejected candidates and reasons, cohort/source hashes, exact features, temporal folds, partitions, model parameters, threshold policy, seeds, package versions, metrics, limitations, and rollback path. Keep the original paper result lineage unchanged.

- [ ] **Step 5: Validate the frozen bundle**

Reload `frozen_baseline.json`, recompute hashes, verify prediction-key uniqueness and metric counts, and reproduce at least one fold into a temporary directory with byte-identical predictions where deterministic behavior is expected.

- [ ] **Step 6: Commit the freeze utility, tests, configuration, and decision**

```bash
git add EthiopiaForecastingExperiment/src/freeze_baseline.py EthiopiaForecastingExperiment/tests/test_freeze_baseline.py EthiopiaForecastingExperiment/configs/frozen_baseline.json EthiopiaForecastingExperiment/docs/decisions/0002-baseline-freeze.md
git commit -m "freeze Ethiopia forecasting baseline"
```

### Task 6: Approve the Provider-Neutral CDS Forecast-Cube Contract

**Files:**
- Create: `EthiopiaForecastingExperiment/contracts/cds_forecast_cube.schema.json`
- Create: `EthiopiaForecastingExperiment/docs/cds-product-decision.md`
- Create: `EthiopiaForecastingExperiment/src/cds_contract.py`
- Create: `EthiopiaForecastingExperiment/tests/test_cds_contract.py`

**Interfaces:**
- Consumes: frozen Ethiopia cohort and a separately reviewed CDS product choice.
- Produces: a schema requiring `FEWSNET_admin_code`, `forecast_origin_month`, `issue_time_utc`, `valid_month`, `lead_month`, `variable`, `statistic`, `value`, `units`, `provider`, `forecast_system_version`, `source_asset`, and `source_checksum`; and `validate_forecast_cube(cube: pandas.DataFrame, cohort_codes: set[int]) -> None` for cross-field and cohort checks.

- [ ] **Step 1: Select the formal CDS product before writing acquisition code**

The product decision must record provider dataset ID, forecast-system version, initialization schedule, maximum lead, ensemble members/statistics, units, spatial resolution, release latency, license, API request fields, and whether hindcasts are available. The reference assembled IPCCH CSV is not acceptable provenance.

- [ ] **Step 2: Write a failing validator test**

The test fixture must include one valid row and separate invalid fixtures for an out-of-cohort admin code, lead 7, a valid month inconsistent with origin plus lead, a duplicated cube key, a non-finite value, blank units, and a blank checksum. Assert that each invalid fixture raises `ValueError` with a field-specific message.

```python
import unittest

import pandas as pd

from EthiopiaForecastingExperiment.src.cds_contract import validate_forecast_cube


class CDSContractTests(unittest.TestCase):
    def valid_cube(self):
        return pd.DataFrame([{
            "FEWSNET_admin_code": 11,
            "forecast_origin_month": "2026-08",
            "issue_time_utc": "2026-08-01T00:00:00Z",
            "valid_month": "2027-02",
            "lead_month": 6,
            "variable": "precipitation",
            "statistic": "ensemble_mean",
            "value": 1.25,
            "units": "mm_month-1",
            "provider": "CDS",
            "forecast_system_version": "accepted-system-version",
            "source_asset": "accepted-asset-id",
            "source_checksum": "abc123",
        }])

    def test_valid_cube_passes(self):
        validate_forecast_cube(self.valid_cube(), cohort_codes={11})

    def test_calendar_mismatch_fails(self):
        cube = self.valid_cube()
        cube.loc[0, "valid_month"] = "2027-01"
        with self.assertRaisesRegex(ValueError, "valid_month"):
            validate_forecast_cube(cube, cohort_codes={11})

    def test_out_of_cohort_code_fails(self):
        with self.assertRaisesRegex(ValueError, "cohort"):
            validate_forecast_cube(self.valid_cube(), cohort_codes={99})

    def test_lead_above_six_fails(self):
        cube = self.valid_cube()
        cube.loc[0, "lead_month"] = 7
        with self.assertRaisesRegex(ValueError, "lead_month"):
            validate_forecast_cube(cube, cohort_codes={11})

    def test_duplicate_key_fails(self):
        cube = pd.concat([self.valid_cube(), self.valid_cube()], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_forecast_cube(cube, cohort_codes={11})

    def test_nonfinite_value_fails(self):
        cube = self.valid_cube()
        cube.loc[0, "value"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_forecast_cube(cube, cohort_codes={11})

    def test_blank_units_fails(self):
        cube = self.valid_cube()
        cube.loc[0, "units"] = ""
        with self.assertRaisesRegex(ValueError, "units"):
            validate_forecast_cube(cube, cohort_codes={11})

    def test_blank_checksum_fails(self):
        cube = self.valid_cube()
        cube.loc[0, "source_checksum"] = " "
        with self.assertRaisesRegex(ValueError, "source_checksum"):
            validate_forecast_cube(cube, cohort_codes={11})
```

- [ ] **Step 3: Run the validator test and verify failure before implementation**

```bash
python3 -m unittest EthiopiaForecastingExperiment.tests.test_cds_contract -v
```

Expected: FAIL because the CDS contract module does not exist.

- [ ] **Step 4: Write the schema and cross-field validator**

The JSON schema enforces required fields and primitive types. The Python validator enforces calendar identity `valid_month = forecast_origin_month + lead_month`, allowed leads 0-6, Ethiopia cohort membership, unique forecast-cube keys, finite numeric values, explicit units, and non-empty source checksums.

```python
import numpy as np
import pandas as pd


CUBE_KEY = [
    "FEWSNET_admin_code",
    "forecast_origin_month",
    "valid_month",
    "variable",
    "statistic",
]


def validate_forecast_cube(cube: pd.DataFrame, cohort_codes: set[int]) -> None:
    required = set(CUBE_KEY).union({
        "issue_time_utc", "lead_month", "value", "units", "provider",
        "forecast_system_version", "source_asset", "source_checksum",
    })
    missing = required.difference(cube.columns)
    if missing:
        raise ValueError(f"Missing forecast-cube fields: {sorted(missing)}")
    if not set(cube["FEWSNET_admin_code"]).issubset(cohort_codes):
        raise ValueError("Forecast cube contains out-of-cohort admin code")
    if not cube["lead_month"].between(0, 6).all():
        raise ValueError("lead_month must be between 0 and 6")
    origin = pd.PeriodIndex(cube["forecast_origin_month"], freq="M")
    valid = pd.PeriodIndex(cube["valid_month"], freq="M")
    expected = pd.PeriodIndex(
        [period + int(lead) for period, lead in zip(origin, cube["lead_month"])],
        freq="M",
    )
    if not valid.equals(expected):
        raise ValueError("valid_month must equal origin plus lead_month")
    if cube.duplicated(CUBE_KEY).any():
        raise ValueError("Duplicate forecast-cube key")
    if not np.isfinite(cube["value"].astype(float)).all():
        raise ValueError("value must be finite")
    for field in ("units", "source_checksum"):
        if cube[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"{field} must be non-empty")
```

- [ ] **Step 5: Run the validator test**

```bash
python3 -m unittest EthiopiaForecastingExperiment.tests.test_cds_contract -v
```

Expected: all validator tests PASS.

- [ ] **Step 6: Record harmonization requirements**

The product decision must define how CDS precipitation and temperature relate to historical `Rainf_f_tavg_mean` and `Tair_f_tavg_mean`, including unit conversion, monthly aggregation, polygon weighting, ensemble statistic, missingness, bias adjustment, and measurement-domain labels. Forecast values may not silently overwrite historical observed/reanalysis values.

- [ ] **Step 7: Stop and create a separate CDS implementation plan**

Do not download CDS data or implement harmonization under this baseline plan. After the product decision and schema are approved, run a new brainstorming and writing-plans cycle for acquisition, polygon aggregation, harmonized feature construction, matched forward prediction, registration of prediction vintages, and delayed outcome evaluation.

- [ ] **Step 8: Commit the approved contract documents and validator**

```bash
git add EthiopiaForecastingExperiment/contracts/cds_forecast_cube.schema.json EthiopiaForecastingExperiment/docs/cds-product-decision.md EthiopiaForecastingExperiment/src/cds_contract.py EthiopiaForecastingExperiment/tests/test_cds_contract.py
git commit -m "define Ethiopia CDS forecast contract"
```

## Final Verification Before Handoff

- [ ] Run `python3 -m unittest discover -s EthiopiaForecastingExperiment/tests -v` and record the exact pass count.
- [ ] Run `git diff --check` and resolve all whitespace errors.
- [ ] Confirm `git status --short` contains no generated data or model artifacts intended to remain local.
- [ ] Confirm production code and frozen release hashes are unchanged.
- [ ] If any production symbol changed, run GitNexus `detect_changes()` against `main` and verify only approved execution flows are affected.
- [ ] Re-read `docs/contract-draft.md` and map every locked contract to a test, decision, configuration, or manifest in this plan.
- [ ] Report unavailable checks honestly; do not call an unexecuted or label-unavailable comparison a performance result.

## Handoff Boundary

This plan ends after the baseline is frozen and the provider-neutral CDS contract is approved. Formal CDS acquisition and the CDS-enhanced forward run are a separate project increment because their exact implementation depends on a product and vintage decision that has not yet been made.
