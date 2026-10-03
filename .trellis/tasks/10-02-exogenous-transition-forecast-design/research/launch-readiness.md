# Launch readiness: 648 Stage 1 candidates + 72 development folds (preparation only)

2026-10-03, Claude executor, from checkpoint `b43ef6a`; the package tree is clean. No product or test edit, no fit, no 2025 outcome access. Source identity is user-attested (no further source research). Prepared by `research/probes/build_alignment.py`, `annual_semantics.py`, `build_release_ledger.py` and `timing_stage1_input.py`.

## Ready

| Input | Status |
|---|---|
| Pinned sources | `prepare_fourclass.py` already pins the panel, FEWSNET.csv, coordinates and shapefile hashes, which were verified in the D7 pass |
| Covariate alignment `research/launch/alignment.json` | built from the agreed D7 rules; passes `check_alignment(schema, ·, real=True)` under the pinned Windows Python; 69 sources → **129 ordered features** of 162 (see breakdown below). The file is tracked: a scoped `.gitignore` negation makes it travel with the reviewed config |
| Annual convention | **Inherited reference-year convention, user-attested sources**: the panel row year is taken as the indicator's reference year, read at `value_month` = 12. GDP and CC never vary within a calendar year in the pinned panel (0 of 76,916 and 0 of 80,052 area-years; values 2010–2023). That is consistent with this convention but does not prove it by itself; disclosed as an assumption, with no producer tracing |
| Release-ledger builder `research/probes/build_release_ledger.py` | (country ISO, cycle) rows from the pinned panel's historical CS presence only; dry run: 954 rows, 22 countries, 51 cycles, 2010-01..2024-10. It matches the observations' country codes, so the real-mode coverage check will pass. **There is no default rule.** |
| Runtime | the frozen Windows py3.12.10 stack, verified in D7 |
| Timing and storage (real panel, H4, 2018-06; scratch probe with a provisional in-memory ledger, never passed to prepare or manifests) | `stage1_input` A/k0 = 42–81 s (177,026 rows × 210 columns; 5.9 MB parquet); B/k2 = 88–136 s (327,232 rows; 7.5 MB). The 108 prepared inputs take about **2–3.3 h** and about **0.7 GB** |

Alignment breakdown (69 sources):
- 28 static (stable terrain, soil and geometry plus the fixed AEZ grouping), as disclosed fixed reconstruction with identity/units caveats;
- 21 monthly at L=1: 19 ACLED and 2 FLDAS raw means;
- 2 annual: GDP (WDI, Y−1 from July) and CC (WGI, Y−2);
- 18 excluded: both z-scores, CPI, EVI, nightlight, nightlight_sd, gpp_mean, FAO_price, market_distance, Food_CPI, Food_food_inflation, WFP_Price, WFP_Price_std, gini and pop, plus **crop, range and market_access** (historical as-of semantics unestablished, D1 temporal-eligibility exclusion).

Ordered features: 162 − 18 excluded sources − 15 legacy columns derived from excluded sources = **129**.

## Remaining configuration fields (only these)

1. **Historical IPC release convention.** `build_release_ledger.py --rule {reference_month_end | following_month_end} --source "<decision citation>"`. This is the scientific alignment decision being made by the coordinator and user. It sets origin-month CS visibility and every cycle mask; nothing runs until it is chosen.
2. **Run directory**: `C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1` (fresh, outside Dropbox), used after the timing decision.
3. **Stage 1 workers**: `--workers 1` initially, per D7's sequential start.

Not needed for the 648/72 development: the 2025 crosswalk, the expert table, the covariate-extension manifest and the actual-availability table. These are used only by `scen-actual`/`scen-evaluate`.

## Exact commands

Run from WSL; PY is the pinned interpreter. Every step refuses to overwrite completed output.

```bash
PY=/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe
T=".trellis/tasks/10-02-exogenous-transition-forecast-design/research"
# 1. ledger from the DECIDED rule (no default)
python3 $T/probes/build_release_ledger.py --panel "<1.Source Data>/FEWSNET_forecast_unadjusted_bm.csv" \
    --rule <DECIDED> --source "<decision citation>" --out $T/launch/release_ledger.csv
cd FEWSNETGeoXGBExperiment
# 2. preparation (pinned sources, preflight, snapshots, 108 scenario inputs, schedule); ~2 h+
$PY -B scripts/prepare_fourclass.py --run-dir 'C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1' \
    --release-ledger '..\.trellis\tasks\10-02-exogenous-transition-forecast-design\research\launch\release_ledger.csv' \
    --alignment '..\.trellis\tasks\10-02-exogenous-transition-forecast-design\research\launch\alignment.json'
# 3. 648 Stage 1 scenario roots (fixed G, ceiling 40,824 fits)
$PY -B scripts/run_stage1.py --run-dir 'C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1' --split-mode scen --workers 1
# 4. 72 complete development folds, then the A/B stop rule
$PY -B scripts/run_experiment.py --run-dir 'C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1' scen-develop
$PY -B scripts/run_experiment.py --run-dir 'C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1' scen-select
```

## Preconditions and known limits

- Metric contract (binding): PRD R3, design D4 and G3–G4. Primary = pooled crisis F1. Per H:
  - normal-scenario matched-persistence delta ≥ −0.02;
  - rank qualifying A/B by the mean of the k1 and k2 F1, exact ties A;
  - no qualifier → that horizon stops.

  The Stage 3 local gate stays strictly > 0.01 with support. There is no superiority requirement and no capacity/threshold/grid expansion.
- Preparation refuses package code that differs from the committed HEAD. Commit any change before step 2; the task-research files are outside the package code identity.
- The real-mode ledger refuses `synthetic` evidence. The builder writes `reconstructed` with the decided-rule citation: a disclosed source-labelled reconstruction, not a verified vintage.
- `scen-develop` runs folds sequentially in-process (`--workers` is not used there). Each fold refits up to 1 + 6 gate globals plus locals. Fit ceiling: provisional 147,889 total, with Stage 1 ≤ 40,824.
- The 2025 actual phases stay blocked by their own contracts. Historical 2021–2024 phases (`scen-freeze`, `scen-historical`, `scen-report`) need only the development outputs above.
- The partial FDW linkage probe was stopped by user steering; its result is recorded as a limitation in PROGRESS.
