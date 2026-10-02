# D53: zero-fit exact-origin-state probability-level transfer (2026-10-02)

**Scope:** contract `d53-state-probability-transfer-plan.md`, planning commit `7f0e909`. Descriptive only; the primary crisis-F1 endpoint, four-class main contract and final criterion are unchanged.
- **Inputs:** only the 63 saved D52 `rows_{FIT,C,E3}.csv.gz` files (21 roots), hash-checked against `research/d52_completion.json`.
- **Groups:** exact origin state **code** 0 / 1 / 2 / 3 = **IPC** 1 / 2 / 3 / 4-or-5, plus `missing` (descriptive, never a state). Codes and IPC labels are written together below to avoid confusion.
- **Scores:** original normalised crisis mass `s_original` and D52 binary `p_binary`; label `truth_crisis`.
- **Interpretation limit:** role composition (areas, label months, eras, calendar regimes) differs between FIT, C and E3, so these contrasts cannot isolate causal temporal drift or pure overfitting. The Brier residual is not called refinement and AUC is not a Brier decomposition. No new transition reference (the D38 empirical prior is not repeated), calibration, threshold or deployable correction.

## Producer and run (executor factual record, `d0015fd`)

- **Producer** `192f0ce`: `research/d53_state_probability_transfer.py` (418 lines, numpy/pandas/stdlib only), git blob `7f85c1dc747ec3149b63f80531d12a579072aa7b`, sha256 `a118434ea96e4971d8580be4b049457cefd62eb431aa7f8de0e9a8dfd1f1e938`. The native check added the supervisor-required `float_precision="round_trip"` row parser, fixed the input-check order and added two selftest rejections.
- **Run:** `C:\Users\swl00\geoxgb_runs\geoxgb-d53-state-probability-transfer-20261002`, frozen Windows Python 3.12.10 with assertions on (`sys.flags.optimize = 0` recorded by a launch of the same interpreter immediately before), exit 0 in 6 s; zero fits. Outputs: 315 role cells, 3,535 month cells (1,269 empty), 168 contrast records; every code 0–3 × H contrast has 7/7 valid roots.

## Verification (supervisor, distinct from the executor record)

`research/d53_supervisor_check.py` → `research/d53_supervisor_results.json` (log `research/d53-supervisor.log`): **PASS**.
- Python stdlib only (csv, gzip, `math.fsum`); no producer imports and no fits; 66,451 checks over 1,660,244 rows.
- All 63 source hashes; 315 role cells, 3,535 month cells, 168 contrasts and 120 summary cells; cell support, rate, mean, bias and Brier including empties; weighted monthly recomposition; contrast identities and means; sign counts recomputed from independently validated saved raw deltas.

These are numerical checks, not statistical tests.

## Results (unweighted means over the 7 roots per H)

**Levels — observed rate / original mean `s` / binary mean `p`:**

| H | Code (IPC) | FIT | C | E3 |
|---|---|---|---|---|
| 4 | 1 (IPC 2) | .1514 / .1567 / .1547 | .1569 / .1569 / .1553 | .1793 / .1693 / .1570 |
| 4 | 2 (IPC 3) | .6162 / .5972 / .6054 | .6062 / .5925 / .6009 | .6624 / .6430 / .6464 |
| 4 | 3 (IPC 4-or-5) | .9670 / .8936 / .9067 | .9689 / .8898 / .9035 | .9164 / .8893 / .9096 |
| 8 | 1 (IPC 2) | .1957 / .1958 / .1959 | .1987 / .1933 / .1936 | .2209 / .2197 / .2032 |
| 8 | 2 (IPC 3) | .4570 / .4606 / .4615 | .4461 / .4568 / .4584 | .5727 / .6003 / .5986 |
| 8 | 3 (IPC 4-or-5) | .9073 / .8780 / .8837 | .8876 / .8706 / .8782 | .8683 / .8887 / .8953 |
| 12 | 1 (IPC 2) | .1361 / .1376 / .1371 | .1373 / .1374 / .1369 | .2166 / .1634 / .1586 |
| 12 | 2 (IPC 3) | .3773 / .3745 / .3803 | .3774 / .3706 / .3771 | .5692 / .5001 / .5231 |
| 12 | 3 (IPC 4-or-5) | .8301 / .7632 / .7824 | .7864 / .7339 / .7496 | .8838 / .7739 / .8154 |

Code 0 (IPC 1) rates are .015–.038 in every role, with all E3 − FIT shifts below .013 in absolute value. Code 3 (IPC 4-or-5) E3 cells hold only 19–123 rows per root.

**E3 − FIT contrasts — Δ rate / Δ original mean / Δ original bias (roots up / down):**

| H | Code 1 (IPC 2) | Code 2 (IPC 3) | Code 3 (IPC 4-or-5) |
|---|---|---|---|
| 4 | +.0279 (4/3) / +.0125 / −.0154 (4/3) | +.0462 (6/1) / +.0458 / −.0005 (4/3) | −.0506 (4/3) / −.0044 / +.0463 (5/2) |
| 8 | +.0252 (3/4) / +.0239 / −.0013 (2/5) | +.1158 (5/2) / +.1396 (7/0) / +.0239 (3/4) | −.0390 (3/4) / +.0107 / +.0497 (4/3) |
| 12 | +.0805 (6/1) / +.0258 (7/0) / −.0547 (2/5) | +.1919 (7/0) / +.1256 (7/0) / −.0663 (1/6) | +.0538 (6/1) / +.0107 / −.0431 (1/6) |

E3 − C contrasts are close to E3 − FIT, because C levels sit near FIT levels (for example H12 code 2 (IPC 3): +.1919 / +.1295 / −.0624). Binary-score bias contrasts are in `d53_contrasts.csv` and `d53_summary.json`.

**Temporal variation (descriptive, all months and roots):**
- Within each root's FIT window, label-month rates vary widely. Code 1 (IPC 2): per-root monthly minima about .02–.09 and maxima about .23–.41. Code 2 (IPC 3): minima about .07–.41 and maxima about .51–.83.
- E3 rates also vary by target date, for example H12 code 2 (IPC 3) .378 (2018-06) to .665 (2019-06) and H12 code 1 (IPC 2) .140 to .373.
- The pooled FIT level is therefore a mixture of very different months.

## Supervisor synthesis

- **The diagnostic is complete. No new prediction policy or model is adopted, and Stage 1 remains unresolved.**
- **H12 code 2 (IPC 3):** FIT actual .3773 with original mean .3745 → E3 actual .5692 with original .5001 and binary .5231. E3 − FIT rate +.1919 (7/7 up), original mean +.1256, bias −.0663 (6/7 down).
- **H12 code 1 (IPC 2):** actual .1361 → .2166 while the original mean moves .1376 → .1634, so underprediction grows.
- **H8 code 2 (IPC 3):** actual .4570 → .5727 while the original mean moves .4606 → .6003, a slight E3 overprediction instead.
- **H4 code 2 (IPC 3):** the mean tracks the rate change fairly closely (actual .6162 → .6624, original .5972 → .6430).
- This does **not** support one uniform level-collapse diagnosis across H, nor a single offset fix. A near-correct group mean does not establish individual calibration, ranking or F1.
- The binary score stays below the F1 baselines (D52) despite its smaller H12 code 2 (IPC 3) mean bias, so improving mean bias alone is insufficient.
- These are exposed, overlapping development roots with area, month and regime composition changes, not proof of causal temporal drift or pure overfitting. Monthly variation and supports, including rare-state caveats, are preserved. The D38 empirical transition-prior repeat was avoided.
- The next policy choice has not been made; no automatic D54, and no fits, calibrators, thresholds, adoption, Stage 2/3 or close.

## Evidence

**Task research holds exact byte copies** (byte-compared with the originals):

| File | Bytes | sha256 |
|---|---|---|
| `d53_role_cells.csv` | 72,468 | `6b8e7f3cba4aa27fa469a86d9a97f2ee89fcc25c76e76800ff5d5444e2fffe8a` |
| `d53_month_cells.csv` | 645,761 | `4cd776b8826d5a2c2040b47509300183b1d85dccc9ab845e489b602a0b2b3480` |
| `d53_contrasts.csv` | 28,965 | `a716ab2245ddaee6a2d0f3639df7bb34889fab42e164dee4467794d61da2b5b0` |
| `d53_summary.json` | 41,087 | `47157b4edd8616a311a648183de62e50d0be4138fabab68244053da879caf75a` |
| `d53_identity.json` | 8,398 | `7062093b5ad97d8850a869a992a8206d734b1c3f629dd9a6906ff3e41389beca` |
| `d53_completion.json` | 503 | `68af9061c85f01674b45b051f11137b4a6f257ec64b48269cfc2b7f93bf9d329` |
| `d53_supervisor_check.py` | 7,981 | `2c4ffa6a73e784bb4458cfd3fde98c496537f46914da95b092afd38ee9970a8b` |
| `d53_supervisor_results.json` | 613 | `2fde0c6e28591541b23017e06448c9e4d1dd019c38663c96b175343b3e07670e` |
| `d53-supervisor.log` | 4,761 | `e28de22f22b315f042968a4f544f6e2f4f07bce8467e7cff82d4f58d7c7927ea` |
| `d53-run.log` | 174 | `b44b6999dd3c466a1bb2b80e6e7054bf9b3e0eec21625ea9205affb8530895b8` |

All six D53 outputs are copied; nothing remains external except the originals.
