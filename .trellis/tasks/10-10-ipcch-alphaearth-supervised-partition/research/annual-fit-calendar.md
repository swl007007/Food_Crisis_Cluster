# Annual fit calendar and 2023 supervision bridge — accepted D27

> Consolidation note (2026-10-10): this file preserves the evidence and decision sequence. Earlier wording that a scientific detail was open is historical; accepted D1–D29 are in ../decisions.md and the current final-review contract is ../design.md. Unverified source/asset/resource facts remain unverified, with checkpoints in ../implement.md. No execution is authorized by these notes.

2026-10-10. Read-only calendar enumeration; no fits, predictions or scoring. A scout read the accepted D3–D7/D18–D25 contracts and compared the existing yearly schedule/planner. The coordinator checked `IPCCHYearlyGeoXGBExperiment/ipcch_yearly_xgb/schedule.py:24-50`: its first-main-year special case and scheduled anchors are reusable concepts, but its ordinary January rule does not apply unmodified to the new first historical partial year.

## Calendar fixed by existing decisions

For each H, an annual block fits at the earliest scheduled eligible target minus H, independent of whether that month has observed labels. A quartet contains four independent q2–q5 XGB regressors. B2/G and all nine correction recipes reuse B1; they do not each fit another backbone.

| H | First historical OOT block and fit origin | Later historical target years through 2022 | Historical B1 quartets |
|---|---|---|---:|
| 1 | 2018-02..12 @ 2018-01 | 2019 @ 2018-12; 2020 @ 2019-12; 2021 @ 2020-12; 2022 @ 2021-12 | 5 |
| 3 | 2018-04..12 @ 2018-01 | 2019 @ 2018-10; 2020 @ 2019-10; 2021 @ 2020-10; 2022 @ 2021-10 | 5 |
| 6 | 2018-07..12 @ 2018-01 | 2019 @ 2018-07; 2020 @ 2019-07; 2021 @ 2020-07; 2022 @ 2021-07 | 5 |
| 12 | 2019-01..12 @ 2018-01 | 2020 @ 2019-01; 2021 @ 2020-01; 2022 @ 2021-01 | 4 |

These 19 blocks include the two development-year B1 fits per H. The final-map cutoff of 2022-12 requires relearning the map on eligible saved OOT residuals, not a separate B1 fit at that cutoff. First historical blocks must not use nominal anchors before 2018-01.

| H | First main block and fit origin | Later main blocks | 2026 Jan–Apr supplement fit origin | Main / supplement quartets per arm |
|---|---|---|---|---:|
| 1 | 2023-02..12 @ 2023-01 | 2024 @ 2023-12; 2025 @ 2024-12 | 2025-12 | 3 / 1 |
| 3 | 2023-04..12 @ 2023-01 | 2024 @ 2023-10; 2025 @ 2024-10 | 2025-10 | 3 / 1 |
| 6 | 2023-07..12 @ 2023-01 | 2024 @ 2023-07; 2025 @ 2024-07 | 2025-07 | 3 / 1 |
| 12 | 2024-01..12 @ 2023-01 | 2025 @ 2024-01 | 2025-01 | 2 / 1 |

Main calendar has 122 H/month slots; the current 2026 Jan–Apr supplement has 16. The latter range follows the existing keyed IPCCH target coverage inspected during this task; it is not authorization to extend the data or add later targets.

## Accepted 2023 bridge (D27)

The following targets satisfy the 2017-representation time condition but have prediction origins before the main boundary. They are neither developer-map labels through 2022 nor main evaluation targets under D4:

| H | Bridge targets | Row-origin range | Historical block fit origin | Extra B1 quartets |
|---|---|---|---|---:|
| 1 | 2023-01 | 2022-12 | 2022-12 | 1 |
| 3 | 2023-01..03 | 2022-10..12 | 2022-10 | 1 |
| 6 | 2023-01..06 | 2022-07..12 | 2022-07 | 1 |
| 12 | 2023-01..12 | 2022-01..12 | 2022-01 | 1 |

User accepted on 2026-10-10: generate these **22 H/month slots only** as additional B1 OOT supervision, using four additional quartets (16 scalar fits). Do not generate overlapping full-2023 alternative historical predictions. From the main boundary onward, reuse the saved formal B1 raw predictions as the OOT provider for future correction updates. Each H/admin/target has one designated historical provider, identified by fit origin and fit identity; never overwrite, double-count or pick a provider by score.

Bridge outcomes cannot enter development recipe selection or final-map learning because their target year is 2023. They may enter subsequent fixed-map correction updates only once their labels meet that update's availability cutoff; the B1 provider itself must not have fitted on those target labels. Predictors remain row-origin-safe. Under accepted D29's month-end proxy, target2023-01 labels are eligible at the first formal fit at 2023-01; this is not certification of historical publication timing. All bridge targets are before the next main annual refit cutoff for their H. These records never expand the main evaluation cohort.

Without the bridge, base B0/B1 fits can still consume these labels later, but correction training lacks their honest OOT residuals. H12 loses the full 2023 target year's correction supervision. Re-predicting them with a later model already trained on their labels is not an acceptable replacement.

## Accepted calendar backbone fit ceiling

With fresh fits and no compatible old-result reuse, B0 needs only the two development comparison years plus main/supplement blocks; it does not need 2018–2020 OOT because correction supervision is based on B1.

| H | B1 without bridge (unselected) | B1 with accepted bridge | B0 | Total quartets with bridge |
|---|---:|---:|---:|---:|
| 1 | 9 | 10 | 6 | 16 |
| 3 | 9 | 10 | 6 | 16 |
| 6 | 9 | 10 | 6 | 16 |
| 12 | 7 | 8 | 5 | 13 |
| Total | 34 | 38 | 23 | 61 |

Thus the unselected no-bridge budget was 57 quartets / 228 scalar XGB fits; the accepted bridge budget is **61 quartets / 244 scalar fits**. These are calendar identity upper counts, not fit-support certification, timing measurements or counts of tree/ridge solves. Actual input identities and support must be enumerated before fits. Nine regional recipes do not multiply the XGB budget. No extra B0 bridge/history experiment or additional main evaluation period is implied.

Existing yearly code carries historical/current fit-origin and provider fields (`ipcch_yearly_xgb/engine.py:220-224,279-287`) and keys fitting identity to actual origins/data (`engine.py:91-103,158-166`); those are reuse evidence, not authority to inherit its old six-date gate or 36-month fitting policy.
