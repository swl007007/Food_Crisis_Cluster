# Results — IPCCH fixed-map MLP residual adaptation (run `mlp-formal-20261005`)

Executed 2026-10-05/06 under the frozen spec (PRD R1–R25, design v0.2), code
`cd7bcd5`, isolated venv (torch 2.6.0+cu124, CUDA). Exploratory: the
2023–2026 evaluation period had already been inspected. Full per-seed tables:
`evidence/formal/summary-tables.md`; complete metrics: `evidence/formal/report.json`.

## Execution

| Stage | Result | Wall time |
| --- | --- | --- |
| preflight | passed; 28 pinned P6 files, 13,260-fit inventory reproduced | <1 min |
| develop (3 replicate workers) | 288 fits, 0 failed | 6.3 min |
| predict (3 replicate workers) | 3 × 4,324 fits, 0 failed | 3.1 h |
| report | passed; support-eligible main keys equal the predeclared 3,211/4,997/3,423/845 | <1 min |
| replay (serial re-execution, read-only store) | **passed, 28,084 checks, 0 failures**; 64,284 model loads, 0 fits | 11 min |

Replay re-runs develop and Stage3 serially and reproduced every artifact of
the parallel run byte-for-byte. The fit inventory is exactly 288 + 3 × 4,324 =
13,260. Run directory (outside Dropbox, 2.7 GB):
`C:\Users\swl00\AppData\Local\Temp\ipcch-mlp-runs\mlp-formal-20261005`
(inventory `evidence/formal/run-inventory.tsv`).

## Recipe selection on development S

Mean crisis F1 of B+P over three seeds (adaptive internal evidence, not a test result):

| H | Winner | Ranking (mean S F1) |
| --- | --- | --- |
| 1 | G2R2 | G2R2 .7912 · G2R1 .7885 · G1R1 .7840 · G1R2 .7815 |
| 3 | G2R1 | G2R1 .7900 · G2R2 .7857 · G1R1 .7743 · G1R2 .7710 |
| 6 | G2R2 | G2R2 .7801 · G2R1 .7791 · G1R2 .7735 · G1R1 .7710 |
| 12 | G1R1 | G1R1 .7830 · G1R2 .7808 · G2R2 .7659 · G2R1 .7589 |

## Main period 2023–2025 (identical keys; seed mean, range in brackets)

Crisis F1, E_all:

| H | keys | B | P | G | Pooled XGB (P6) | G−P | P−B | B − pooled XGB |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | 17,322 | .7543 | .7558 | .7550 | .7780 | −.0008 [−.0012, −.0005] | +.0015 [−.0021, +.0047] | −.0237 [−.0267, −.0210] |
| 3 | 16,919 | .7420 | .7485 | .7481 | .7759 | −.0005 [−.0011, +.0001] | +.0066 [+.0007, +.0147] | −.0340 [−.0345, −.0334] |
| 6 | 16,413 | .7352 | .7374 | .7377 | .7699 | +.0003 [−.0002, +.0010] | +.0022 [−.0042, +.0085] | −.0346 [−.0365, −.0336] |
| 12 | 14,087 | .7047 | .7099 | .7101 | .7564 | +.0002 [−.0006, +.0011] | +.0052 [+.0003, +.0085] | −.0517 [−.0569, −.0415] |

Per-seed G−P country bootstrap (12 seed×H intervals): 10 include zero; H3
seed 44 is −.0011 [−.0024, −.0000] and H12 seed 43 is +.0011 [+.0002, +.0026],
opposite signs, no multiplicity adjustment.

Ungated supported-cohort diagnostic, L−P crisis F1 on all L-eligible keys
(7,784–9,866 keys per H): H1 −.0041 [−.0055, −.0013], H3 −.0033 [−.0039, −.0022],
H6 +.0032 [−.0003, +.0056], H12 +.0001 [−.0037, +.0045]. On keys the gate
actually adopted, L−P is negative in 9 of 12 seed×H cells.

Versus persistence on E_persist (G − persistence crisis F1): H1 −.0218, H3 −.0252,
H6 −.0347, H12 −.0439 (seed means). Per-seed intervals exclude zero for 2/3 seeds
at H3 and all seeds at H6 and H12; all three include zero at H1. P6 GeoXGB was
+.003 to +.008 above persistence on the same keys.

Other metrics: projected q3 R² of P is .08–.14 at H1/H3 and negative at H6/H12
(P6 pooled XGB .25–.37). Four-class macro-F1 of G is .450–.492, around the
P6 XGB values (.460–.481) and below persistence (.529–.566).

## Supplementary 2026 (4 target months, point only)

G−P is −.0049 to +.0037; P−B is −.0077 to +.0227. B is below pooled XGB at
H1–H6 (−.031 to −.052) and about equal at H12 (+.001 mean, range −.023 to
+.022). G − persistence is positive on average at H1–H6 (seed means +.013 to +.019;
one H3 seed −.004) and negative at H12 (−.031), the same sign pattern as P6. Projected q3 R² at H12 is about −0.9
in every seed, so the 2026 H12 shares are badly miscalibrated even where the
crisis label is close.

## Reading

1. **The MLP learner is weaker than XGB here.** Pooled MLP (B) is .021–.057
   crisis F1 below the matched pooled XGB in every main-period seed and horizon,
   with much lower q3 R². Adding a pooled residual network (P) recovers only
   .002–.007 on average.
2. **Regional residual correction adds nothing measurable on the fixed XGB maps.**
   G−P stays within ±.0012 in every main-period seed×H. This is bounded by
   coverage, as pre-registered: G can change at most 6.0–29.5% of main keys.
   But the ungated L−P on the supported cohort is also small and mixed in sign
   (−.006 to +.006), and adopted routes are mostly worse than P. So the near-zero
   G−P is not only a coverage artifact.
3. **The MLP system is below persistence** in crisis F1 at every main H; the
   gap grows with H.
4. Seeds move P and G by up to .015 crisis F1 within an H, larger than any G−P
   difference; the three replicates are not independent datasets.

## Limits

- XGB-selected maps, reused development S, already-viewed evaluation period;
  a negative regional result applies to these maps, not to MLP-specific partitions.
- One architecture grid (2×2), fixed epochs (100/40) and final weights, no
  early stopping; recipe selection favours pooled fitting. Residual networks
  in small regions get far fewer optimizer updates (80–320 vs 1,520–3,000 for P);
  L−P compares these fixed procedures, not geography at equal budgets.
- Missingness encoding (median + flags) differs from XGB native NaN handling;
  the B−XGB gap mixes learner and encoding.
- Bootstrap intervals are per-seed, descriptive, conditional on saved
  predictions, without training/map-selection uncertainty or multiplicity
  adjustment. Seed mean/range is descriptive only.
- Observation-month-end availability is inherited; release vintages are not
  reconstructed.
