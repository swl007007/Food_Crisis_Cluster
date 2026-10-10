# Temporal support for calendar grilling — 2026-10-10

> Consolidation note (2026-10-10): this file preserves the evidence and decision sequence. Earlier wording that a scientific detail was open is historical; accepted D1–D29 are in ../decisions.md and the current final-review contract is ../design.md. Unverified source/asset/resource facts remain unverified, with checkpoints in ../implement.md. No execution is authorized by these notes.

Read-only descriptive probe, Python standard-library csv/gzip/hashlib; no fit, prediction, rescore, or feature computation. Sources:

- `IPCCHGeoXGBExperiment/runs/p6-formal-20261004b/prepared/target_ledger_valid.csv.gz`
- Same directory `keys_h01.csv.gz`, `keys_h03.csv.gz`, `keys_h06.csv.gz`, `keys_h12.csv.gz`.
- Canonical valid-ledger SHA256 recomputed: `21d7701fe60fb9ed3084c34dc7852a9ce56e312c78587238179c5ea8ef63e717`.
- Main calendar reference: `IPCCHGeoXGBExperiment/ipcch_geoxgb/schedule.py:58-71`; origin = target − H. User accepted this outer origin boundary for the new task (D4); inner protocol and final horizon set remain open.

## Existing QC-valid original outcomes by target year

| Year | Rows | Distinct areas | Crisis rows |
|---|---:|---:|---:|
| 2014 | 460 | 261 | 22 |
| 2015 | 518 | 264 | 19 |
| 2016 | 529 | 270 | 22 |
| 2017 | 1,726 | 1,062 | 436 |
| 2018 | 2,163 | 1,557 | 808 |
| 2019 | 2,685 | 2,031 | 954 |
| 2020 | 4,111 | 2,449 | 1,604 |
| 2021 | 3,246 | 2,130 | 1,055 |
| 2022 | 4,153 | 2,863 | 1,702 |
| 2023 | 4,925 | 3,238 | 2,059 |
| 2024 | 4,295 | 2,960 | 1,822 |
| 2025 | 9,792 | 5,740 | 4,893 |
| 2026 | 4,092 | 3,710 | 2,411 |

Sum: 42,695. Counts reflect the frozen P6 source/QC, not a newly refreshed outcome source. Repeated rows in the same area are not independent samples.

## Potential supervised-development keys after the representation year

Predicate: `origin_month >= '2018-01'` AND `target_month <= '2022-12'` (month strings YYYY-MM). No new local-support filter or actual AlphaEarth coverage applied.

| H | All eligible development keys | Distinct areas | Crisis rows | Targets through 2020 | Target 2021 | Target 2022 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 15,708 | 4,430 | 5,974 | 8,309 | 3,246 | 4,153 |
| 3 | 15,660 | 4,408 | 5,942 | 8,261 | 3,246 | 4,153 |
| 6 | 15,555 | 4,363 | 5,900 | 8,156 | 3,246 | 4,153 |
| 12 | 14,195 | 4,266 | 5,315 | 6,796 | 3,246 | 4,153 |

These are candidate keys, not a claim that every historical fold has sufficient past training data. Inner temporal splitting must use labels available by the validation **origin**, particularly for H12; a target-year split alone does not establish this.

## Potential main evaluation under the existing origin boundary

Predicate: `origin_month >= '2023-01'` AND `target_month <= '2025-12'`.

| H | First scheduled target | Existing valid keys | Distinct areas | Crisis rows |
|---|---|---:|---:|---:|
| 1 | 2023-02 | 17,322 | 5,999 | 8,270 |
| 3 | 2023-04 | 16,919 | 5,999 | 8,005 |
| 6 | 2023-07 | 16,413 | 5,939 | 7,599 |
| 12 | 2024-01 | 14,087 | 5,921 | 6,715 |

This retains several calendar years for retrospective evaluation and separates map-selection outcomes (<=2022) from evaluated forecast origins (>=2023). It does not make 2023–2026 scientifically untouched after prior experiments. If map learning uses labels through end-2022, H12 targets in 2023 cannot be called forecasts from that frozen map made before those labels were available.

User accepted this outer boundary, with 2026 reported separately (D4). Inner development split, predictive-model updating, horizon set, publication lag, and statistical acceptance remain unresolved; pre-2018 training treatment was subsequently accepted in D5. Actual feature joins and AlphaEarth coverage may alter support and require a no-fit manifest before any execution release.

## Accepted pre-2018 training treatment (D5)

The saved valid ledger contains 3,233 outcomes in 2014–2017 (460 + 518 + 529 + 1,726). The current compact baseline uses expanding historical training, not a mandatory 2018 cutoff (`research/local-reuse-facts.md`, annual-fit references).

User accepted: keep these outcomes available for training predictors fitted at origins >=2018-01, subject to the chosen history/weighting/QC contract. For arms using AlphaEarth, attach the fixed 2017 admin descriptor to these training rows as geographical metadata available at the later **fit origin**. This is an explicit retrospective enrichment of training rows, not a claim that their original historical row origins had access to that representation. Dynamic historical features must still follow their original row origins; target labels must precede each actual model fit cutoff. Common training-key eligibility should apply across comparator arms to avoid confusing a sample-selection change with representation/partition gain.

Do not create pre-2018 pseudo-out-of-time predictions with the 2017 annual descriptor, and do not treat fitted residuals on these old rows as honest partition supervision. Partition supervision should use eligible historical predictions made at 2018+ origins with their labels held out of the corresponding base fits; exact inner protocol remains open.

Alternative: exclude pre-2018 training rows from every arm. This gives a stricter row-origin interpretation of every AlphaEarth training feature but removes initial training history and pushes the first adequately supported historical fits later. Neither choice has been compared numerically. Retaining history is not a guarantee of improved performance; the 2017 descriptor may be a poor match to older environmental conditions.

## Accepted two forward validation blocks (D6)

User accepted: use target years 2021 and 2022 as two development validation blocks. At each block's earliest forecast origin, fit all learned candidate components using only information then eligible, including the candidate partition and regional correction. Select complexity/regularization using the two later blocks. Then relearn the final map from eligible historical out-of-time predictions whose labels are available by end-2022 and freeze it for the outer evaluation. Relearning candidate maps inside development does not conflict with D1's final-map freeze.

Safe block anchor = January(target year) − H. The following descriptive counts use `supervision origin >=2018-01` and `supervision target <= anchor` in the same saved keyed views. This is a month-end availability convention for planning; actual release timestamps and fit supports remain unresolved. No prediction/fit performed.

| H | 2021 block anchor | Earlier candidate supervision keys | 2022 block anchor | Earlier candidate supervision keys |
|---|---|---:|---|---:|
| 1 | 2020-12 | 8,309 | 2021-12 | 11,555 |
| 3 | 2020-10 | 8,197 | 2021-10 | 11,351 |
| 6 | 2020-07 | 6,293 | 2021-07 | 9,868 |
| 12 | 2020-01 | 3,683 | 2021-01 | 7,793 |

Validation keys = 3,246 for 2021 and 4,153 for 2022 under each inspected H. These are existing valid keys before AlphaEarth joins and model-support rules; they do not establish effective independent sample size or fitting feasibility. In particular, long-horizon/early maps need strong complexity limits or a predefined global fallback.

This validates the map-learning procedure, rather than evaluating a map already trained on its validation labels. Residuals held out of base-model fits alone are insufficient if the map/regularization/preprocessing are then learned using the same validation outcomes. All data-dependent steps must be scoped to the appropriate training portion. The two years provide two temporal views for selection, not independent proof of a statistically stable win. Alternative: one final development holdout costs fewer candidate fits but gives less evidence of temporal consistency. Exact candidate budget and base-prediction schedule remain open.

## Accepted consistent annual predictor updates (D7)

Primary document read in full: `../IPCCH/.trellis/tasks/archive/2026-10/10-08-compact-climate-weather-oracle/design.md`, especially lines 123–133. It specifies one fit per target-year/H, all eligible historical labels with `w(U) = 0.5 ** ((fit_origin - U) / 24)`, and dynamic features of each training row built at that row's own origin. This source also specifies reported-phase truth/raw threshold decoding/H0, which are not adopted merely by reusing its training schedule.

User accepted: use annual target-year blocks with expanding eligible training history and 24-month half-life consistently for base out-of-time predictions and formal pooled/regional-correction parameter updates. The final partition remains fixed. Historical residual labels used to train a correction must themselves be available at the current fit cutoff; they cannot be borrowed from future blocks. Apply the same age definition to eligible correction supervision, subject to later confirmation of the correction model/loss. Fixed-map regional parameters may be refitted as new labels become available; no post-freeze partition/recipe selection on evaluation outcomes.

For each block, fit at the earliest eligible scheduled forecast origin. A first partial target year must be truncated by the D3 or D4 origin boundary before choosing that anchor. For example, the first main H1 block starts at target 2023-02 / origin 2023-01, not at the otherwise nominal 2022-12 origin for target January. The initial historical blocks likewise cannot fit AlphaEarth models before 2018-01. Do not set the anchor from whichever area happens to have the first observed label.

Annual updates lower fit count and reuse existing tooling; monthly updates can react sooner but require more fits and define a different reference protocol. Previous yearly fixed-map tests did not establish geographic gains, so annual updating is a design-control choice, not a claimed remedy or causal explanation of prior null results. Budget, exact availability timestamp, support rules and correction architecture remain open.

## Read-only support-detail probe — 2026-10-10

A read-only scout recomputed the eight cohorts from `IPCCHGeoXGBExperiment/runs/p6-formal-20261004b/prepared/keys_h{01,03,06,12}.csv.gz`, using `admin_code`, `origin_month`, `target_month`, `target_ord` and the same predicates as above. It also checked that ordinals match month strings, target minus origin equals H, and no cohort contains duplicate admin/target keys. All eight earlier row counts reproduced. These are eligible candidate keys, not verified available OOT predictions or AlphaEarth-covered training rows.

| Validation year | H | Anchor | Rows | Admins | Nonempty target months | First / last target | Span, max-minus-min months | Nonempty-month rows min/max | Sum w | Weighted row n_eff |
|---|---:|---|---:|---:|---:|---|---:|---|---:|---:|
| 2021 | 1 | 2020-12 | 8309 | 3107 | 33 | 2018-02 / 2020-12 | 34 | 6 / 1082 | 5797.041041 | 7838.647153 |
| 2021 | 3 | 2020-10 | 8197 | 3064 | 29 | 2018-04 / 2020-10 | 30 | 8 / 1082 | 6055.795587 | 7744.521674 |
| 2021 | 6 | 2020-07 | 6293 | 2810 | 23 | 2018-07 / 2020-07 | 24 | 8 / 1003 | 4634.780616 | 6036.171180 |
| 2021 | 12 | 2020-01 | 3683 | 2065 | 12 | 2019-01 / 2020-01 | 12 | 22 / 1003 | 3215.378812 | 3633.960260 |
| 2022 | 1 | 2021-12 | 11555 | 3551 | 44 | 2018-02 / 2021-12 | 46 | 6 / 1339 | 6813.130495 | 10457.015309 |
| 2022 | 3 | 2021-10 | 11351 | 3476 | 40 | 2018-04 / 2021-10 | 42 | 8 / 1339 | 7043.331376 | 10308.845008 |
| 2022 | 6 | 2021-07 | 9868 | 3269 | 35 | 2018-07 / 2021-07 | 36 | 8 / 1082 | 6184.822002 | 9191.177260 |
| 2022 | 12 | 2021-01 | 7793 | 2951 | 23 | 2019-01 / 2021-01 | 24 | 8 / 1082 | 5953.630105 | 7476.629122 |

Weights are `w=2^(-(anchor_ord-target_ord)/24)` and `n_eff=(sum w)^2/sum(w^2)`. This n_eff measures concentration of row weights, not independent sample size; repeated admins, shared observation rounds and spatial/time dependence remain. Every cohort has an empty month somewhere within its calendar span; min rows would be zero if empty months were included.

Earliest H12 month counts: 2019-01:631, 02:60, 03:0, 04:39, 05:120, 06:240, 07:355, 08:109, 09:1003, 10:64, 11:42, 12:22; 2020-01:998. Coordinator independently spot-checked H12 total/admin/month counts and the top three months: 2632/3683 = 71.46348% of rows. This motivates reporting temporal concentration in addition to simple row counts; it does not establish a validated minimum leaf size.

Coordinator-calculated source SHA256 (compressed file bytes):

| File | SHA256 |
|---|---|
| keys_h01.csv.gz | 414cba066f9953e7b70e2f586450537621ea312ad80a9fbc2b84ad367f2e49cd |
| keys_h03.csv.gz | 5e44eb8359a4910a5d9d94b80986d3a44faf123d89ebdf2b30e5389bf1267f7a |
| keys_h06.csv.gz | d325da25a69399b496a33b1f21f3fb2fdd24f387b3074f399e5f2836444495d6 |
| keys_h12.csv.gz | 9755df742b15d5b1556671633c2fa80908d8b9b7ffee9c9192b895aa64bba4bd |

No AlphaEarth extraction, actual OOT prediction generation, feature matrix construction, or fitting was performed.

## Main-period country and calendar support — read-only probe, 2026-10-10

A separate read-only scout counted the same four canonical keyed tables with the exact predicate `origin_month >= "2023-01" and target_month <= "2025-12"`. It checked H and month ordinals, unique admin/target keys, valid-target ledger membership, nonmissing country keys, and consistent admin-to-country mapping. Source compressed-file hashes match the table above; the valid-target ledger hash is `21d7701fe60fb9ed3084c34dc7852a9ce56e312c78587238179c5ea8ef63e717`. The coordinator independently reproduced H12's total, country count, yearly counts, top-three countries and file hash. No predictions were scored.

| H | Rows | Admins | Countries | Scheduled / nonempty months | 2023 rows | 2024 rows | 2025 rows | Top-three-country row share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17322 | 5999 | 53 | 35 / 32 | 3235 | 4295 | 9792 | 40.8556% |
| 3 | 16919 | 5999 | 53 | 33 / 30 | 2832 | 4295 | 9792 | 39.9255% |
| 6 | 16413 | 5939 | 53 | 30 / 27 | 2326 | 4295 | 9792 | 40.6751% |
| 12 | 14087 | 5921 | 52 | 24 / 21 | not scheduled | 4295 | 9792 | 40.2002% |

Scheduled-month counts are derived from D4 and H (`2023-01 + H` through `2025-12`), not taken from a new fitted run. All H have no valid target rows in `2024-12`, `2025-11`, `2025-12`. The 2024–2025 admin/target/country key sets are identical across H. H1/H3/H6 country sets are identical; H12 lacks only Timor-Leste. Country counts by actual target year are H1:46/45/50, H3:42/45/50, H6:39/45/50, H12:not-scheduled/45/50.

Somalia, Nigeria and Central African Republic are the three largest country cohorts in all H. Country row-count min/median/max are H1:8/158/3256, H3:8/158/2934, H6:7/158/2934, H12:6/133.5/2586. Country nonempty-month min/median/max are H1:1/5/9, H3:1/5/9, H6:1/5/8, H12:1/4/7. Thus global monthly coverage is not continuous monthly coverage in every country. The 2025 fraction is 56.53%, 57.88%, 59.66%, 69.51%, respectively.

These are canonical eligible evaluation rows before feature/AlphaEarth extraction, not proof of complete model outputs or independent observations. There are only three observed target years (two for H12). Country resampling preserves their observed calendar and cannot independently establish future-year stability or account for shocks shared across countries. Counting countries does not validate the between-country independence/exchangeability assumption.

Minimal reproduction from repo root (stdlib only, stdout only):

```python
import csv, gzip
from collections import Counter
from pathlib import Path
base = Path("IPCCHGeoXGBExperiment/runs/p6-formal-20261004b/prepared")
for h in (1, 3, 6, 12):
    with gzip.open(base / f"keys_h{h:02d}.csv.gz", "rt", newline="") as stream:
        rows = [r for r in csv.DictReader(stream)
                if r["origin_month"] >= "2023-01" and r["target_month"] <= "2025-12"]
    countries = Counter(r["country_key"] for r in rows)
    print(h, len(rows), len(countries),
          Counter(r["target_month"][:4] for r in rows), countries.most_common(3))
```
