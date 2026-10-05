# P6 results: p6-formal-20261004b predict → report → replay

Release: `p6-stage3-release.md`, copied verbatim from
/tmp/ipcch-geoxgb-stage3-release-20261004b.md after the supervisor accepted the
learn-map checkpoint. Implementation code identity **6798df2**; package and
configs unchanged; same run, maps and pinned runtime; Dropbox sync paused. No
retuning. The audit run 7ced754e… / base 6c98f73 was not closed. Task not
closed; final acceptance and the lifecycle step are pending.

## Execution

| Stage | Result | Wall time | Identity |
|---|---|---|---|
| predict | exit 0, status passed | 60 m 41 s | stage3-summary sha256 1df8165ab8aa9e99… |
| report | exit 0, status passed | 4 s | report.json sha256 142d717dedaf8afb573b544743cc6856c0d33d18e1b79a82543bb2dc514691f0 |
| replay | exit 0, **passed, 91 880 checks, 0 failures** | 11 m 44 s | replay sha256 1195716881b8509c53efd47b1d22920b6cb5c6d3952185f9ce316953682e30e0 |

Logs are in `evidence/p6-logs/{predict,report,replay}-b.log` (LF). The run
inventory is in `evidence/p6-formal-20261004b-inventory.json`. The model store
holds 1225 entries (429 Stage1 + 796 Stage3), 6125 files, 2.0 GB, with no
`.tmp` dir and no INCOMPLETE marker. Prediction rows: H1 21 414, H3 21 011,
H6 20 505, H12 18 179. Of these, local-routed rows are 944 / 722 / 739 / 348.

## Stage3 requests and fits vs the checkpoint enumeration

`evidence/stage3_reconcile.py` writes
`evidence/p6-formal-20261004b-stage3-reconcile.json`, status **passed**.

| H | Current global | Historical global | Historical local | Current local (≤ upper) | Distinct global / hist-local / current-only local | Quartet fits | Hits | Failed | Scalar fits (enumerated exact–upper) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 36 | 216 | 931 | 12 (≤76) | 43 / 183 / 2 (≤9) | 228 | 967 | 0 | 912 (904–940) |
| 3 | 34 | 204 | 736 | 13 (≤82) | 43 / 143 / 3 (≤16) | 189 | 798 | 0 | 756 (744–808) |
| 6 | 31 | 186 | 614 | 4 (≤59) | 43 / 129 / 1 (≤17) | 173 | 662 | 0 | 692 (688–756) |
| 12 | 25 | 150 | 723 | 11 (≤37) | 43 / 157 / 6 (≤21) | 206 | 703 | 0 | 824 (800–884) |
| Total | 126 | 756 | 3004 | 40 (≤254) | 172 / 612 / 12 (≤63) | **796** | 3130 | **0** | **3184 (3136–3388)** |

Every exact category equals the enumeration. Current locals and current-only
distinct locals fall inside the upper bounds. Requests total 3926 ≤ 4140.
Fits × 4 equal the distinct scalar fits; the R48 ceiling is 80 920.

## Final metrics (main 2023–2025 folds vs supplementary 2026 folds)

GeoXGB is the frozen-map routed predictor. Pooled is the matched R44 global
recipe on the same E_all keys. Persistence is paired on E_persist. Crisis
binary F1 Δ intervals are the R49 country-cluster bootstrap (main period only:
seed 42, 2000 draws; all defined). Supplementary is point estimates only, by
R49. Full metric/delta/confusion/per-class detail is in
`evidence/p6-formal-20261004b-report.json`; per-month and per-country
diagnostics are in `evidence/p6-diag/`.

### main

| H | E_all keys / countries | GeoXGB F1 | Pooled F1 | Δ F1 geo−pool [95% CI] (defined/2000) | GeoXGB 4-class acc / macroF1 | Pooled 4-class acc / macroF1 | q3 R² geo / pool (projected) |
|---|---|---|---|---|---|---|---|
| 1 | 17322 / 53 | 0.7777 | 0.7780 | -0.0003 [-0.0010, 0.0000] (2000) | 0.6345 / 0.4596 | 0.6349 / 0.4598 | 0.3026 / 0.3053 |
| 3 | 16919 / 53 | 0.7756 | 0.7759 | -0.0003 [-0.0012, 0.0004] (2000) | 0.6413 / 0.4627 | 0.6417 / 0.4633 | 0.3670 / 0.3672 |
| 6 | 16413 / 53 | 0.7699 | 0.7699 | 0.0001 [-0.0001, 0.0004] (2000) | 0.6278 / 0.4807 | 0.6277 / 0.4811 | 0.2543 / 0.2541 |
| 12 | 14087 / 52 | 0.7561 | 0.7564 | -0.0003 [-0.0008, 0.0000] (2000) | 0.6314 / 0.4643 | 0.6318 / 0.4646 | 0.2521 / 0.2525 |

| H | E_persist keys (coverage) | GeoXGB F1 | Persistence F1 | Δ F1 geo−persistence [95% CI] (defined/2000) | GeoXGB 4-class acc / macroF1 | Persistence 4-class acc / macroF1 | GeoXGB q3 R² |
|---|---|---|---|---|---|---|---|
| 1 | 16002 (0.924) | 0.7742 | 0.7691 | 0.0050 [-0.0098, 0.0242] (2000) | 0.6391 / 0.4665 | 0.6444 / 0.5567 | 0.3131 |
| 3 | 15907 (0.940) | 0.7719 | 0.7645 | 0.0074 [-0.0098, 0.0326] (2000) | 0.6450 / 0.4694 | 0.6422 / 0.5591 | 0.3826 |
| 6 | 14960 (0.911) | 0.7684 | 0.7657 | 0.0027 [-0.0151, 0.0157] (2000) | 0.6387 / 0.4961 | 0.6485 / 0.5661 | 0.3136 |
| 12 | 12404 (0.881) | 0.7600 | 0.7525 | 0.0075 [-0.0073, 0.0305] (2000) | 0.6460 / 0.4821 | 0.6167 / 0.5293 | 0.2989 |

| H | cohort rows | local rows | global fallback | unmapped-area global | gate region-folds enabled / adopted / decisions | cohort areas in map / unmapped |
|---|---|---|---|---|---|---|
| 1 | 17322 | 588 | 9355 | 7379 | 14 / 7 / 288 | 3191 / 2808 |
| 3 | 16919 | 580 | 9286 | 7053 | 12 / 10 / 210 | 3191 / 2808 |
| 6 | 16413 | 509 | 9013 | 6891 | 7 / 3 / 162 | 3132 / 2807 |
| 12 | 14087 | 111 | 7793 | 6183 | 10 / 6 / 189 | 3129 / 2792 |

### supplementary

| H | E_all keys / countries | GeoXGB F1 | Pooled F1 | Δ F1 geo−pool [95% CI] (defined/2000) | GeoXGB 4-class acc / macroF1 | Pooled 4-class acc / macroF1 | q3 R² geo / pool (projected) |
|---|---|---|---|---|---|---|---|
| 1 | 4092 / 32 | 0.7762 | 0.7766 | -0.0004 (point only; no interval by R49) | 0.6007 / 0.3256 | 0.5997 / 0.3247 | 0.3486 / 0.3498 |
| 3 | 4092 / 32 | 0.7822 | 0.7825 | -0.0003 (point only; no interval by R49) | 0.6127 / 0.3696 | 0.6129 / 0.3688 | 0.3626 / 0.3626 |
| 6 | 4092 / 32 | 0.7867 | 0.7854 | 0.0013 (point only; no interval by R49) | 0.6039 / 0.3740 | 0.6014 / 0.3720 | 0.3288 / 0.3265 |
| 12 | 4092 / 32 | 0.7747 | 0.7747 | -0.0000 (point only; no interval by R49) | 0.5828 / 0.3237 | 0.5828 / 0.3235 | 0.2294 / 0.2299 |

| H | E_persist keys (coverage) | GeoXGB F1 | Persistence F1 | Δ F1 geo−persistence [95% CI] (defined/2000) | GeoXGB 4-class acc / macroF1 | Persistence 4-class acc / macroF1 | GeoXGB q3 R² |
|---|---|---|---|---|---|---|---|
| 1 | 4089 (0.999) | 0.7759 | 0.7277 | 0.0482 (point only; no interval by R49) | 0.6006 / 0.3231 | 0.5701 / 0.3725 | 0.3480 |
| 3 | 4089 (0.999) | 0.7820 | 0.7249 | 0.0571 (point only; no interval by R49) | 0.6131 / 0.3701 | 0.5762 / 0.3993 | 0.3617 |
| 6 | 4089 (0.999) | 0.7865 | 0.7259 | 0.0606 (point only; no interval by R49) | 0.6041 / 0.3743 | 0.5713 / 0.3952 | 0.3274 |
| 12 | 4040 (0.987) | 0.7801 | 0.8082 | -0.0281 (point only; no interval by R49) | 0.5884 / 0.3235 | 0.6386 / 0.4829 | 0.2295 |

| H | cohort rows | local rows | global fallback | unmapped-area global | gate region-folds enabled / adopted / decisions | cohort areas in map / unmapped |
|---|---|---|---|---|---|---|
| 1 | 4092 | 356 | 1473 | 2263 | 8 / 5 / 36 | 1608 / 2102 |
| 3 | 4092 | 142 | 1687 | 2263 | 3 / 3 / 28 | 1608 / 2102 |
| 6 | 4092 | 230 | 1599 | 2263 | 1 / 1 / 24 | 1608 / 2102 |
| 12 | 4092 | 237 | 1592 | 2263 | 6 / 5 / 36 | 1608 / 2102 |

Interpretation: pointwise descriptive 95% country-cluster bootstrap intervals conditional on the saved predictions and observed cohort; no training/map-selection, shared-shock or future-year uncertainty; no multiplicity adjustment

Main-period deltas vs persistence (GeoXGB − persistence, E_persist), other metrics:

| H | binary acc | precision | recall | F2 | 4-class acc | 4-class macro F1 | q3 R² (projected) |
|---|---|---|---|---|---|---|---|
| 1 | −0.0089 | −0.0380 | +0.0560 | +0.0345 | −0.0053 | −0.0902 | +0.1191 |
| 3 | −0.0155 | −0.0578 | +0.0904 | +0.0547 | +0.0028 | −0.0897 | +0.1878 |
| 6 | −0.0174 | −0.0565 | +0.0767 | +0.0450 | −0.0098 | −0.0700 | +0.0863 |
| 12 | −0.0131 | −0.0486 | +0.0774 | +0.0475 | +0.0293 | −0.0472 | +0.2522 |

## Reading (descriptive; no claim beyond the contract)

1. **The partition does almost nothing.** Main-period GeoXGB vs pooled crisis
   F1 Δ is between −0.0003 and +0.0001 at every H. All four intervals straddle
   or touch 0, and all other metric deltas are ≤ 0.003 in absolute value. Only
   3.4 / 3.4 / 3.1 / 0.8 % of main cohort rows are routed local. Gates are
   enabled in 14 / 12 / 7 / 10 of 288 / 210 / 162 / 189 region-folds and
   adopted in 7 / 10 / 3 / 6.
2. **Half the cohort is outside the learned map.** 2808 of 5999 main-cohort
   areas (H1) are not in the 3264-area Stage1 map, which covers 2014–2022
   keys. By design they are always served by the global quartet. Among the
   mapped rows, most are global fallbacks (gate support or gain ≤ .01).
3. **Versus persistence (main):** crisis F1 is higher by +0.0050 / +0.0074 /
   +0.0027 / +0.0075, but every 95% interval includes 0. GeoXGB trades
   precision (−0.04 to −0.06) for recall (+0.06 to +0.09). Four-class macro F1
   is clearly worse than persistence (−0.05 to −0.09), mostly in phases 1 and
   4/5. Binary accuracy is lower at every H.
4. **Supplementary 2026 (point only, 32 countries, 4092 keys):** GeoXGB
   crisis F1 exceeds persistence at H1–H6 (+0.048 to +0.061) but trails it at
   H12 (−0.028). GeoXGB − pooled is −0.0004 to +0.0013.
5. Stage1 selection F1s (0.83–0.85) are in-sample S-half values and are not
   these evaluation results.

## Limitations

- The R41 file-lock stop of run a is preserved (see `p6-restart-evidence.md`).
  Run b completed with Dropbox sync paused, the suspected (not proven) lock
  source.
- The intervals are descriptive and conditional on the saved predictions, per
  the report's `interpretation` field. There is no training/map-selection or
  shared-shock uncertainty and no multiplicity adjustment.
- 2026 supplementary results are point estimates on four target months only.
