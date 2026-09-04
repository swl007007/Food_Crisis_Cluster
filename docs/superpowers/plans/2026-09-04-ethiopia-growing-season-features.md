# Ethiopia Previous-Growing-Season Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen Ethiopia admin-month lookup of previous-growing-season statistics and join its seven model features at each forecast origin for fs0-fs3.

**Architecture:** The spec-authorized single-purpose script reads the frozen ETH baseline and frozen monthly SPI table, assigns each admin to one of three fixed FAO Country Brief calendar profiles, aggregates completed seasons, and writes the fixed 2010-2024 lookup CSV. The existing horizon aligner loads that table once inside `build_aligned_panels()` and joins only its seven model columns on `FEWSNET_admin_code + forecast_origin_month`; `align_horizon()` and the working-panel builder remain unchanged.

**Tech Stack:** Python 3.12, pandas, NumPy, the existing standard-library stack, and `unittest`. Add no dependency.

**Spec:** `EthiopiaForecastingExperiment/docs/growing-season-feature-contract.md`

**Execution status (2026-09-04):** Complete. The user explicitly skipped TDD;
the two red-step runs were omitted, while post-implementation focused tests and
independent artifact checks were retained.

## Global Constraints

- Preserve the frozen 1,040-admin cohort and exact `FEWSNET_admin_code` identity.
- Baseline SHA-256 must equal `25e458ac6fbdb27c9b264ada111bfe8fc38c6f2c376bd023dc8fb1c4d15eb855`.
- SPI table SHA-256 must equal `d18e311aba521c680975a39ea8193c3b8f29d47f4ed5bb18fd0a6a0d6e40b547`.
- Calendar profiles and exact ADMIN1/ADMIN2 sets come verbatim from the approved spec; unmatched admins use `meher_only`.
- Mean features require `ceil(2/3 * expected_months)` observations; `sum(EVI)` requires the full season.
- A lookup month `F` may use only the most recent season with `season_end < F`.
- For target `M`, join at `F=M-H`; never select a season relative to `M`.
- Keep audit fields in the lookup only. Add only the seven approved feature columns to aligned model tables.
- Keep fs0/fs1/fs2/fs3 at 1/4/8/12 months and preserve current aligned row counts and key hashes.
- Use the spec-authorized generator and fixed output path; add no alternate path option.
- Preserve all pre-existing uncommitted conflict-intensity changes. Do not edit `prepare_working_panel.py` for this feature.
- Do not commit unless the user separately authorizes it.

## Files

- Create: `EthiopiaForecastingExperiment/prepare_growing_season_lookup.py` — sole lookup generator and writer.
- Create: `EthiopiaForecastingExperiment/tests/test_growing_season_features.py` — artifact-contract and exact-origin checks.
- Modify: `EthiopiaForecastingExperiment/prepare_horizon_aligned_data.py` — load and join the lookup at forecast origin.
- Modify: `EthiopiaForecastingExperiment/data_lineage.jsonl` — append lookup step 22 and alignment steps 23-26 after artifacts pass.
- Generated, ignored: `EthiopiaForecastingExperiment/data/interim/growing_season/ethiopia_previous_growing_season_monthly.csv`.
- Do not modify: `EthiopiaForecastingExperiment/prepare_working_panel.py` or existing ledger rows 1-24.

---

### Task 1: Build the independent growing-season lookup

**Files:**

- Create: `EthiopiaForecastingExperiment/prepare_growing_season_lookup.py`
- Create: `EthiopiaForecastingExperiment/tests/test_growing_season_features.py`

**Interfaces:**

- Consumes: frozen baseline columns `FEWSNET_admin_code`, `date`, `ADMIN1`, `ADMIN2`, `gpp_mean`, `Tair_f_tavg_mean`, `EVI`; frozen SPI columns `FEWSNET_admin_code`, `date`, `SPI_1`, `SPI_3`, `SPI_6`, `SPI_12`.
- Produces: only the fixed lookup artifact authorized by the spec; no reusable production API.

- [x] **Step 1: Record the pre-existing dirty-worktree boundary**

Run:

```bash
git status --short
git diff -- EthiopiaForecastingExperiment/prepare_working_panel.py EthiopiaForecastingExperiment/prepare_horizon_aligned_data.py EthiopiaForecastingExperiment/data_lineage.jsonl EthiopiaForecastingExperiment/tests/test_conflict_intensity.py
```

Expected: the current conflict-intensity changes remain visible and are not reverted or overwritten during this plan.

- [x] **Step 2: Run pre-edit GitNexus impact checks**

Run impact analysis for `align_horizon`, `build_aligned_panels`, and `main` in `prepare_horizon_aligned_data.py`. Report `UNKNOWN` if the ETH module is still absent from the index; do not report it as zero impact. New generator symbols have no pre-existing callers.

- [x] **Step 3: Add one artifact-contract test file**

Create `test_growing_season_features.py` using the existing `sys.path` + stdlib
`unittest` pattern. The test reads the fixed generated lookup and asserts the
spec's acceptance contract: exact columns, unique admin-month keys, 1,040
admins, 180 months, 187,200 rows, group counts `650/301/89`, strict
`previous_season_end < lookup_month`, 87,360 complete rows in 2018-2024, and
10,352 full-period all-feature-null rows. Keep the seven feature names local to
the test; do not create a production import seam for tests.

- [x] **Step 4: Skip the TDD red step by user direction**

Execution note: intentionally not run; the user directed implementation without
TDD.

- [x] **Step 5: Implement the single-purpose generator**

Define only fixed constants and one `main()` entry point; do not create a public
builder, calendar-assignment helper, writer helper, calendar classes, provider
adapters, configuration files, or a generic seasonal framework. Reuse
`file_sha256` from `era5_drought_spi.py`.

```python
KEY = "FEWSNET_admin_code"
DATE = "date"
DEFAULT_BASELINE = EXPERIMENT_DIR / "outputs" / "baseline_audit" / "fewsnet_eth_pre_georf_20260831" / "fewsnet_eth_pre_georf.csv.gz"
DEFAULT_SPI = EXPERIMENT_DIR / "data" / "interim" / "era5_drought_spi" / "ethiopia_spi_monthly.csv"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "data" / "interim" / "growing_season" / "ethiopia_previous_growing_season_monthly.csv"
MODEL_FEATURES = (
    "previous_season_avg_SPI_1",
    "previous_season_avg_SPI_3",
    "previous_season_avg_SPI_6",
    "previous_season_avg_SPI_12",
    "previous_season_avg_gpp_mean",
    "previous_season_avg_Tair_f_tavg_mean",
    "previous_season_sum_EVI",
)
SEASONS = {
    "meher_only": (("meher", 6, 12),),
    "belg_meher_bimodal": (("belg", 2, 7), ("meher", 6, 12)),
    "pastoral_bimodal": (("gu_genna", 3, 5), ("deyr_hageya", 10, 12)),
}
PASTORAL_ADMIN2 = frozenset({
    ("Oromia", "Borena"), ("Oromia", "Guji"), ("Oromia", "West Guji"),
    ("Somali", "Afder"), ("Somali", "Daawa"), ("Somali", "Doolo"),
    ("Somali", "Korahe"), ("Somali", "Liban"), ("Somali", "Shabelle"),
})
BELG_MEHER_ADMIN2 = frozenset({
    ("Tigray", "Southern"), ("Tigray", "South Eastern"),
    ("Amhara", "North Wello"), ("Amhara", "South Wello"),
    ("Amhara", "North Shewa (AM)"), ("Amhara", "Oromia"),
    ("Oromia", "East Hararge"), ("Oromia", "West Hararge"),
    ("Oromia", "East Shewa"), ("Oromia", "Arsi"),
    ("Oromia", "Bale"), ("Oromia", "East Bale"),
    ("SNNP", "Guraghe"), ("SNNP", "Hadiya"), ("SNNP", "Halaba"),
    ("SNNP", "Kembata Tibaro"), ("SNNP", "Siltie"),
    ("SNNP", "Yem Special"),
})
```

Import the existing hash helper and perform calendar assignment directly inside
`main()`:

```python
from era5_drought_spi import file_sha256

pairs = list(zip(admins["ADMIN1"], admins["ADMIN2"]))
admins["calendar_group"] = np.select(
    [
        [pair in PASTORAL_ADMIN2 for pair in pairs],
        [pair in BELG_MEHER_ADMIN2 or pair[0] == "Sidama" for pair in pairs],
    ],
    ["pastoral_bimodal", "belg_meher_bimodal"],
    default="meher_only",
)
```

The `main()` body must perform these concrete operations in order: validate and
one-to-one merge source keys;
merge one calendar group per admin; generate each `SEASONS` interval for
2010-2024; aggregate the seven source columns by admin and interval; apply the
coverage formulas below; construct all 187,200 lookup keys; select the maximum
season end strictly below each lookup month; and construct the ordered model and
audit columns. Write the reproducible generated table directly to the fixed
spec-authorized path with `DataFrame.to_csv(index=False)`; add no alternate path,
overwrite switch, temporary-file protocol, or writer abstraction.

Use these exact aggregation expressions for each admin-season group:

```python
observed = values.notna().sum()
mean_value = values.mean() if observed >= math.ceil(2 * expected_months / 3) else np.nan
evi_sum = values.sum() if observed == expected_months else np.nan
```

`main()` must:

1. verify both fixed input hashes before reading;
2. read only required columns and parse month-start dates;
3. reject duplicate keys, unequal key sets, non-ETH baseline content, or dates outside 2010-01 through 2024-12;
4. build the lookup directly;
5. assert group counts `650/301/89`, 187,200 output rows, 180 months, and no duplicate admin/year/month keys;
6. assert 87,360 rows in 2018-2024 with no null among `MODEL_FEATURES` and 10,352 all-feature-null rows across the full output;
7. write the fixed generated artifact and report its path and output hash.

Build seasons directly with small fixed loops over `SEASONS` and years, aggregate by admin, and map each lookup month to the latest strict earlier season. Use `math.ceil(2 * expected_months / 3)` for mean validity and exact equality for EVI coverage. Keep one `expected_months` column and these per-variable audit columns:

```text
observed_SPI_1_months, observed_SPI_3_months,
observed_SPI_6_months, observed_SPI_12_months,
observed_gpp_mean_months, observed_Tair_f_tavg_mean_months,
observed_EVI_months
```

Do not retain a season name/year or an assignment-rule field. `calendar_group`, `previous_season_start`, and `previous_season_end` are sufficient audit identity.

- [x] **Step 6: Run the focused test and generator smoke check**

Run:

```bash
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' EthiopiaForecastingExperiment/prepare_growing_season_lookup.py
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' -m unittest EthiopiaForecastingExperiment.tests.test_growing_season_features -v
```

Expected: tests pass; generator reports 187,200 rows, group counts `650/301/89`, 87,360 complete 2018-2024 rows, and the fixed output path.

---

### Task 2: Join seasonal values at the forecast origin

**Files:**

- Modify: `EthiopiaForecastingExperiment/prepare_horizon_aligned_data.py`
- Modify: `EthiopiaForecastingExperiment/tests/test_growing_season_features.py`

**Interfaces:**

- Consumes: lookup rows unique on `FEWSNET_admin_code + year + month`.
- Produces: fs0-fs3 aligned panels containing seven seasonal model columns joined at `forecast_origin_month`.

- [x] **Step 1: Add exact-origin assertions**

Keep `align_horizon()` unchanged. In the test, create a temporary working panel,
a temporary lookup where every admin-month has visibly different seasonal
values, and a temporary output directory. Use `unittest.mock.patch.object` only
in test code to point `prepare_horizon_aligned_data.DEFAULT_SEASON_LOOKUP` at the
temporary lookup, then call the existing `build_aligned_panels()` interface.
For each scope, assert `forecast_origin_month = target_month - H`, compare all
seven values to the temporary lookup row keyed by
`FEWSNET_admin_code + forecast_origin_month`, and assert no audit-only lookup
field appears in output. The existing five-argument `align_horizon()` call in
`test_conflict_intensity.py` remains untouched.

- [x] **Step 2: Skip the TDD red step by user direction**

Execution note: intentionally not run; the user directed implementation without
TDD.

- [x] **Step 3: Add the fixed lookup read and origin merge**

In `prepare_horizon_aligned_data.py`, declare the seven spec-fixed names locally
to avoid a new dependency on the generator module:

```python
SEASON_FEATURES = (
    "previous_season_avg_SPI_1",
    "previous_season_avg_SPI_3",
    "previous_season_avg_SPI_6",
    "previous_season_avg_SPI_12",
    "previous_season_avg_gpp_mean",
    "previous_season_avg_Tair_f_tavg_mean",
    "previous_season_sum_EVI",
)
DEFAULT_SEASON_LOOKUP = EXPERIMENT_DIR / "data" / "interim" / "growing_season" / "ethiopia_previous_growing_season_monthly.csv"
```

Leave `align_horizon()` unchanged. At the start of `build_aligned_panels()`, read
the fixed lookup once and directly prepare the join key:

```python
seasonal_lookup = pd.read_csv(
    DEFAULT_SEASON_LOOKUP,
    usecols=[KEY, "year", "month", *SEASON_FEATURES],
)
seasonal_lookup["forecast_origin_month"] = pd.to_datetime(
    {
        "year": seasonal_lookup.pop("year"),
        "month": seasonal_lookup.pop("month"),
        "day": 1,
    }
)
if seasonal_lookup.duplicated([KEY, "forecast_origin_month"]).any():
    raise ValueError("Growing-season lookup has duplicate keys")
```

After each existing `align_horizon()` call and before `write_and_verify()`, add
the required merge unconditionally:

```python
aligned = aligned.merge(
    seasonal_lookup,
    on=[KEY, "forecast_origin_month"],
    how="left",
    validate="many_to_one",
)
```

Do not add the names to `REFERENCE_DYNAMIC_FEATURES`, because they do not live
in the working panel. Do not add a loader function, optional lookup parameter,
cross-module feature-name import, or CLI path option. Update the console summary
to report 47 working-panel dynamic features plus seven seasonal features.

- [x] **Step 4: Run focused tests**

Run:

```bash
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' -m unittest EthiopiaForecastingExperiment.tests.test_growing_season_features EthiopiaForecastingExperiment.tests.test_conflict_intensity EthiopiaForecastingExperiment.tests.test_local_partition_experiment -v
```

Expected: all focused tests pass, including the unchanged conflict test that calls `align_horizon()` without a lookup.

- [x] **Step 5: Rebuild all four aligned artifacts**

Run:

```bash
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' EthiopiaForecastingExperiment/prepare_horizon_aligned_data.py --overwrite
```

Expected rows remain fs0 `50,868`, fs1 `49,836`, fs2 `48,801`, fs3 `47,766`; each output grows from 84 to 91 columns.

---

### Task 3: Independently verify artifacts and append lineage

**Files:**

- Modify: `EthiopiaForecastingExperiment/data_lineage.jsonl`
- Verify only: generated lookup and four aligned CSVs.

**Interfaces:**

- Consumes: final code and generated artifacts from Tasks 1-2.
- Produces: append-only provenance records and completion evidence.

- [x] **Step 1: Independently recompute the lookup contract**

Run a separate one-off Python check, without importing the generator, that reads the two frozen inputs and output and asserts:

```python
assert len(lookup) == 187_200
assert lookup["FEWSNET_admin_code"].nunique() == 1_040
assert not lookup.duplicated(["FEWSNET_admin_code", "year", "month"]).any()
assert lookup.groupby("calendar_group")["FEWSNET_admin_code"].nunique().to_dict() == {
    "meher_only": 650,
    "belg_meher_bimodal": 301,
    "pastoral_bimodal": 89,
}
lookup_month = pd.to_datetime(
    {"year": lookup["year"], "month": lookup["month"], "day": 1}
)
valid = lookup["previous_season_end"].notna()
assert (lookup.loc[valid, "previous_season_end"] < lookup_month.loc[valid]).all()
evaluation = lookup[lookup["year"].between(2018, 2024)]
assert len(evaluation) == 87_360
assert not evaluation[list(MODEL_FEATURES)].isna().any().any()
assert lookup[list(MODEL_FEATURES)].isna().all(axis=1).sum() == 10_352
```

Recompute representative season means/sums directly from monthly source rows for at least one admin in each calendar group. Verify every mean's observed count meets `ceil(2/3 * expected_months)` and every published EVI sum has `observed_EVI_months == expected_months`.

- [x] **Step 2: Independently verify all four origin joins**

For each aligned CSV, reconstruct lookup month from `forecast_origin_month`,
merge the seven lookup values on exact admin/month, and compare every seasonal
column with `np.allclose(aligned[column], expected[column], equal_nan=True)`.
Assert:

- horizons are exactly `1/4/8/12`;
- row counts remain `50,868/49,836/48,801/47,766`;
- output columns are exactly 91;
- existing key hashes remain:
  - fs0 `d179fe461bc06f4a64663bf840e7d32051740352b14f8975cbac5865778495f5`
  - fs1 `bd7ad1953f7be4b1b99590eb18f3f1c0e5bf46f633443479b058e84589921edb`
  - fs2 `057264816155d1e92b5d7cea778f27f99bf733a2abb00d97ea6725920fbf55eb`
  - fs3 `9eafa7875674a98a5fe7ff9e634d1f39729bda205f4cc7878401f903093fd443`.

- [x] **Step 3: Append five lineage records**

Append, never rewrite:

1. `22_add_previous_growing_season_lookup`, recording both input hashes, FAO Country Brief URL/reference date, fixed calendar sets, group counts, coverage rules, lookup row/null accounting, generator hash, and lookup file/schema/key hashes.
2. `23_align_fs0_previous_growing_season` through
   `26_align_fs3_previous_growing_season`, each recording both upstreams in
   `input_step_ids=["17_add_conflict_intensity_ma12",
   "22_add_previous_growing_season_lookup"]`, plus horizon, row accounting, 47
   working-panel dynamic features, seven seasonal features, aligner hash, and
   output file/schema/key hashes.

Compute hashes only after final code and CSV bytes are stable. Follow ledger line 1 exactly:

```text
file_sha256   = SHA-256 of artifact bytes
schema_sha256 = SHA-256 of ordered column names joined by "\n", including final "\n"
key_sha256    = SHA-256 of sorted "|"-joined key rows, each including final "\n"
```

- [x] **Step 4: Run final quality gates**

Run:

```bash
git diff --check
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' -m unittest EthiopiaForecastingExperiment.tests.test_growing_season_features EthiopiaForecastingExperiment.tests.test_conflict_intensity EthiopiaForecastingExperiment.tests.test_local_partition_experiment -v
```

Parse every line of `data_lineage.jsonl` with `json.loads`, then independently confirm the five new recorded file hashes match disk.

The current Windows Store Python lacks `netCDF4`, so full test discovery may fail while importing the pre-existing SPI test. Do not add a dependency or weaken that test as part of this feature. If a Python 3.12 environment with `netCDF4` is already available, also run:

```bash
python -m unittest discover -s EthiopiaForecastingExperiment/tests -v
```

- [x] **Step 5: Run GitNexus change detection and review scope**

Run `detect_changes(scope="all")`. Confirm only the new lookup generator/test, the ETH horizon-alignment flow, the approved spec/plan, and appended lineage are affected. Treat an unindexed ETH module as `UNKNOWN`, not zero impact.

- [x] **Step 6: Stop at the commit gate**

Report the exact changed files, generated ignored artifacts, checks, and known `netCDF4` limitation. Do not stage or commit until separately authorized. When authorized, keep the pre-existing conflict-intensity work and growing-season work in clearly messaged commits without reverting either.
