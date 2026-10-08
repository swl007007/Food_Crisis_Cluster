# Results — fixed-map yearly GeoXGB (run `yearly-formal-20261007`)

Code `29c5fee` (`fit_source_sha256` `d7a69395…`), locked P6 Windows runtime,
36 inputs staged outside Dropbox. Released by the supervisor at `29c5fee`;
exploratory: 2023–2026 had already been inspected, and historical gate replay
is conditional on the through-2022 maps/recipes. Complete metrics:
`evidence/formal/report.json`.

## Execution

| Stage | Result | Wall time |
| --- | --- | --- |
| preflight | passed; 36 inputs staged and verified; 724-fit inventory reproduced | <1 min |
| predict | 181 quartets fitted = 724 scalar fits, 0 failed; 747 requests (181 fit + 566 in-memory reuse) | 9.9 min |
| report | passed | <1 min |
| replay | **passed, 10,246 checks, 0 failures; zero fits**; 724-fit inventory exact | 2.1 min |

Per H global/local quartets 5/45, 5/35, 6/36, 5/44, as planned; 78 identities
served both current and gate requests (planned 9 global + 69 local). Rows
21,414/21,011/20,505/18,179 cover all 138 folds (126 scored). Gate-supported
current keys equal the planning enumeration in every block. Final inventory:
1,398 files, 1.13 GB (`evidence/formal/final-inventory.tsv`).

## Main period 2023–2025 (identical keys and truth)

Crisis F1, E_all (all evaluation keys). P = annual pooled, G = annual gated GeoXGB.

| H | keys | P | G | P6 pooled | P6 GeoXGB | G−P [95% CI] | P − P6 pooled | G − P6 GeoXGB |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 17,322 | .7524 | .7527 | .7780 | .7777 | +.0003 [−.0008, +.0018] | −.0256 | −.0250 |
| 3 | 16,919 | .7791 | .7787 | .7759 | .7756 | −.0004 [−.0017, +.0004] | +.0032 | +.0031 |
| 6 | 16,413 | .7657 | .7653 | .7699 | .7699 | −.0003 [−.0013, +.0002] | −.0042 | −.0046 |
| 12 | 14,087 | .7576 | .7576 | .7564 | .7561 | +.0001 [.0000, +.0002] | +.0012 | +.0015 |

Every G−P interval includes zero (H12's lower bound is exactly 0).

Gate and coverage, main (decisions counted per region × annual block):

| H | decisions | historical support | enabled = adopted | adopted keys | ungated L−P on L-eligible keys | L−P on adopted keys |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 27 | 8 | 2 | 395 | −.0027 (9,943) | +.0127 |
| 3 | 21 | 9 | 3 | 699 | −.0021 (9,866) | −.0320 |
| 6 | 18 | 7 | 3 | 341 | −.0015 (9,522) | −.0168 |
| 12 | 18 | 4 | 1 | 56 | −.0071 (7,904) | +.0089 |

Versus persistence. Columns 3–5 use E_persist (persistence-available keys); the last column (projected q3 R²) uses E_all and is not a persistence comparison:

| H | E_persist keys | G − persistence [95% CI] (E_persist) | P6 GeoXGB − persistence (E_persist) | four-class macro-F1 G / P6 GeoXGB / persistence (E_persist) | projected q3 R² P / P6 pooled (E_all) |
| --- | ---: | --- | ---: | --- | --- |
| 1 | 16,002 | −.0272 [−.0631, +.0076] | +.0050 | .438 / .467 / .557 | .204 / .305 |
| 3 | 15,907 | +.0077 [−.0102, +.0378] | +.0074 | .482 / .469 / .559 | .300 / .367 |
| 6 | 14,960 | −.0015 [−.0256, +.0256] | +.0027 | .502 / .496 / .566 | .189 / .254 |
| 12 | 12,404 | +.0111 [−.0008, +.0262] | +.0075 | .478 / .482 / .529 | .300 / .253 |

## Supplementary 2026 (point estimates)

G−P is .0000 to +.0001 at every H; one (H1) and two (H3) regions adopt
locally, none at H6/H12. P − P6 pooled is +.017, +.006, −.036, −.030 for
H1/H3/H6/H12; G − persistence is +.066, +.064, +.024, −.058.

## Reading

1. **No clear geographic gain under the annual protocol.** G−P point
   estimates are near zero (about ±.0004; raw H3 −.000402) in every main
   horizon, and all intervals include zero. This is not evidence of an exact
   zero or of equivalence. Few
   region-blocks pass historical support (4–9 per H) and fewer pass the gain
   gate (1–3). The ungated supported-cohort L−P is also slightly negative
   at every H (−.0071 to −.0015), which weakens a coverage-only explanation
   but does not show the effect is zero; adopted regions are mixed (two of
   four horizons negative).
2. **The annual pooled protocol is not uniformly better or worse than P6
   monthly pooled.** It is .026 lower at H1, within ±.005 at H3–H12. This is
   a bundled change (yearly refit, full history, 24-month decay, annual gate
   timing) and cannot be attributed to one ingredient.
3. **Persistence:** the main crisis-F1 difference is negative at H1 and H6
   and positive at H3 and H12; all intervals include zero. Four-class
   macro-F1 stays clearly below persistence, as with P6.
4. Projected q3 R² of annual pooled is lower than P6 pooled at H1–H6 and
   higher at H12.

## Limits

- Fixed through-2022 XGB-selected maps and recipes; historical gate dates are
  scored with models whose map/recipe was chosen with later development data.
- Decay weights reduce effective mass (ESS about 15.7k–19.6k of 21k–29k main
  pool rows) while min_child_weight/regularization are unchanged; this is part
  of the adopted protocol, not tuned.
- Intervals are per-H country bootstraps conditional on saved predictions, no
  multiplicity adjustment; other contrasts are point estimates.
- Single seed (42) by the retained recipe; no seed study.
