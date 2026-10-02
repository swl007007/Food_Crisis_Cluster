# Root E1 crisis exposure and evidence-free side assignment (zero fit, 2026-10-01)

**Scope:** saved D29 root predictions on S (`validation_predictions.csv.gz` `y_root`) and saved `s_branch.pkl`, plus original v1 root-decision rows for two unexposed H4 targets. No fits; this is not a fix.

## 1. Exposure counts

The E1 crisis masses are `D_g = 2TP_g + FP_g + FN_g`, `Y = D_g/D`, `A = 2TP_g/D` (`src/metrics/fourclass.py:195-207`, used at `partition_opt.py:132`). Areas whose S rows are all true negatives therefore have zero E1 exposure, although they do have S rows. This is **not** the D32 no-search condition. These areas must not be masked as `s-1`, and this is not called a code bug.

**D29 six roots** (independent recount matches the supervisor's `/tmp/d33_e1_mass_support.json` on every field; `mixed` = ≥1 TP and ≥1 FP/FN):

| Quantity | Range over the six roots |
|---|---|
| S areas | 5082–5286 |
| Exposed areas (D_g > 0) | 844–1189 |
| Areas with exactly one TP/FP/FN row | 722–958 |
| TN-only areas | 4097–4316 |
| Total D | 1374–2185 |
| Largest single-area share of D | ≤ 0.3% |

**Unexposed targets** (v1 G3 roots, full r80 validation = S ∪ C, about 4 rows per area; root-decision `y_parent` in `e2_predictions.csv.gz`):

| Target | Areas | Zero-exposure areas | One-event areas |
|---|---|---|---|
| H4 2018-06 | 5363 | 3950 (74%) | 790 |
| H4 2018-10 | 5365 | 3958 (74%) | 796 |

Sparse exposure is therefore not specific to the six exposed cases. Under the pre-D26 four-class masses, every correct row contributed mass to its class, so this condition arose with the D26 binary E1 endpoint.

## 2. How zero-exposure areas are assigned at the root split (mechanism, not proven cause)

- A zero-mass group has `c = b = 0`, so its scan score is `g = c·log q + b(1−q) = 0`.
- `get_top_cells` (`partition_opt.py`) sorts `g` with `np.argsort` and fills `s0` up to about half of the **groups**: `FLEX_TYPE='n_group'`, `FLEX_RATIO=0.1` (`config.py:216-220`).
- The side of most zero-mass groups is therefore set by sort/tie order on the area index, then smoothed by contiguity refinement.

**Saved D29 depth-1 sides:**
- 1803–1927 zero-mass areas sit on side 0.
- Sorted by admin code, the side of zero-mass areas switches only 48–84 times over more than 4,000 areas.
- Only 7.5–15% of side-0 zero-mass areas lie below the median code.
- Exposed areas on side 0: 254–491 of 844–1189.

So most of the root split's membership follows index order rather than E1 evidence. This plausibly contributes to root-split instability, but it is not shown to cause the transfer failure.

**Reproduction:** `python3 research/e1_exposure_tie.py <D29 stage1_rootconf dir>`.
