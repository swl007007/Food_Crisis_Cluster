# Zero-fit development baseline and root-label map routing (2026-10-01)

**Scope:** NO fits, NO code or map-rule changes. Inputs are the original saved files only. Final 2021–2024 labels and scores were not opened (`baselines.csv` unread). Script `/tmp/zero_fit_baseline.py` → `/tmp/zero_fit_baseline.json`. The supervisor's independent stdlib rescore (`/tmp/d31_dev_baseline_independent.json`) agrees on all main-cohort scores.

## 1. Global root vs persistence/expert on the 18 development folds

**Inputs:** `geoxgb-v1-20261001/gscreen/predictions.csv.gz` (390,408 rows = 4 G × 3 H × 32,534) and `prepared/ledgers/dev_baselines.csv` (97,602 rows, targets 2019-02..2020-10). The selected G is the D26 crisis re-selection G1/G4/G2 used in D29–D31. The v1 file's own `selection.json` says G3/G2/G2 (the pre-D26 four-class rule).

**Checks:** 0 duplicate prediction identities; `y_pred_code` equals the probability argmax on every row; every origin is T−H; the keyed merge (area, target, H) is 1:1 complete with 0 truth mismatches; 0 duplicate ledger keys. Crisis = the four-class argmax collapsed to code ≥ 2 (IPC ≥ 3), not a probability sum.

**Main cohort:** H4/H8 need persistence and expert; H12 needs persistence. Pooled confusion, not an average of monthly F1. All 18 folds are kept; H12 2019-02 has no prior map and stays as pooled.

| H | N | Crisis prevalence | Root F1 (P/R; TP/FP/FN) | Persistence F1 (P/R) | Expert F1 (P/R) | Root − persistence | Root − expert | Four-class root / persistence |
|---|---|---|---|---|---|---|---|---|
| 4 | 32083 | .189 | .6339 (.689/.587; 3551/1602/2499) | .6410 (.641/.641) | .6820 (.666/.698) | −.0070 | −.0481 | .585 / .654 |
| 8 | 32232 | .188 | .5397 (.561/.520; 3147/2466/2902) | .5423 (.552/.533) | .6096 (.627/.593) | −.0026 | −.0699 | .493 / .582 |
| 12 | 32109 | .189 | .4938 (.730/.373; 2261/838/3797) | .5677 (.584/.552) | — | −.0739 | — | .499 / .581 |

- **Excluded from the main cohort:** 451 (H4), 302 (H8) and 425 (H12) rows.
- **Supplementary cohort (persistence available; expert excluded from this comparison):** H4 N 32393, root .633756548 / persistence .641565275; H8 N 32251, .539897260 / .542708508; H12 N 32109, .493829857 / .567718941. Matches the supervisor.
- **Coverage diagnostic (all truth keys, root only; not a baseline comparison):** H4 .6329 (N 32534), H8 .5385, H12 .4923.
- **Class counts, all keys** (codes 0/1/2/3): 14674 / 11767 / 5879 / 214.
- **Per-fold scores** are in the JSON. The root beats persistence in 6/18 folds (H4 3, H8 2, H12 1) and the expert in 1/12 (h8_2020-02).

**Selection bias:** G was chosen on these same folds. Main-cohort crisis F1 for G1/G2/G3/G4:
- H4: .6339 / .6231 / .6281 / .6181
- H8: .5321 / .5371 / .5326 / .5397
- H12: .4623 / .4938 / .4844 / .4881

These are configuration scores, not an estimate or bound of selection optimism. The selection bias is disclosed and left unquantified.

**Reading:** H4/H8 are close to persistence but well below the expert. H12 has a large deficit with low recall. Root weakness depends on the horizon; the global root is not hopeless everywhere. The Stage 1 local increments observed on the six targets (|Δ| ≲ .01 on the target month) have not closed these gaps. That says nothing about whether future or gated-map experiments could.

## 2. The "root" label in Stage 1 maps and how Stage 2 consumes it

**Producer:**
- `correspondence_table.csv` lists every area in the search input `gtrain` and labels an empty branch ID as `"root"` (`FEWSNETGeoXGBExperiment/app/main_model_GF.py:121-128`).
- An area's branch defaults to root and is overwritten only if `s_branch` holds the area in a branch (`src/helper/helper.py:285-301`).

**Saved 2018-02 maps:**
- Root-labelled areas are exactly the covered areas with no search (S) rows: in every split candidate, 0 root areas have S and 0 non-root areas lack S.
- All of them have fitting and target rows and are predicted by the root booster (an inherited root model), but the label carries no partition evidence.

| Source | Root-labelled areas | Of which |
|---|---|---|
| D28 | 3 / 3 / 200 (H4 / H8 / H12) | no validation rows at all |
| D29 | 181 / 180 / 277 | 178 / 177 / 77 have only C rows |
| D30 and every split D31 seed | 2267 / 2384 / 2457 (42–46% of coverage) | 2086 / 2204 / 2180 have only older, unused S |

The one genuine all-root map is the unsplit D31 H8 m102: all 5361 areas, including 2977 that have S. Its weight is 0.

**Unassigned geometry:** the coordinate CSV and the shapefile have 5718 polygons (unique IDs 0–5717). The preparation's area universe is 5716 areas with at least one valid label (`prepared/manifests/geometry.json` `areas`); polygons 216 and 2786 never carry a label. Not covered by the 2018-02 maps: 353 / 355 / 357 of the 5716 labelled areas (H4 / H8 / H12), equal to 355 / 357 / 359 of the 5718 polygons. 1–5 of them have target rows.

**Consumer:**
- Step 3 fills uncovered areas with `"s-1"` (`scripts/step3_create_linked_tables.py:111-125`).
- Step 4 maps `"s-1"` to −1 and excludes it (`scripts/step4_similarity_matrix.py:17, 120-136`).
- Every other label, **including `"root"`, is an ordinary region**. In each positive-weight plan, all root-labelled areas become co-members of one group (`step4_similarity_matrix.py:140-160`), then the result is multiplied by the spatial Gaussian.
- In-scope areas are those with any non-`s-1` label, so root areas count (`step4_similarity_matrix.py:83-100`).
- Nothing downstream excludes them. The Stage 3 unmapped fallback applies only to areas missing from the consensus map (`src/experiment/stage3.py:292-301`), so these areas receive real cluster IDs and are eligible for gated local fits.

**Consequence:** a consensus built from D30 or D31 positive plans would treat about 2.3–2.5k unsearched areas as one strongly co-assigned pseudo-region. D31's H12 seeds share exactly the same root set (2457 areas, because the per-area counts match D30), so that co-membership is repeated across plans. Its actual weight in the weighted, spatially multiplied matrix has not been measured. In D28/D29 the same problem exists but is small (3–277 areas).

Deciding whether "root" should count as no information (like `s-1`) is a change to the consensus rule. That decision belongs to the next spec and has not been made here.

## 3. D31 map agreement with and without shared root membership (no fit)

The supervisor recomputed ARI from the raw D31 correspondence tables (stdlib contingency counts; `/tmp/d31_ari_without_root.json`). It was scored first on all shared areas, which matches the reporter's values, and then only on areas that are non-root in **both** maps. The executor spot-checked it with sklearn `adjusted_rand_score` (`/tmp/ari_spot.py`), and the values are identical:

| Early 2018-02 pair | All shared areas: ARI (areas) | Non-root in both: ARI (areas) |
|---|---|---|
| H4 m101 vs m102 | .793 (5363) | .469 (3096) |
| H12 m101 vs m103 | .816 (5359) | .384 (2902) |
| H8 m101 vs m103 | .724 (5361) | .248 (2977) |

Supervisor's other values:
- Early H4 pairs fall from .793 / .689 / .812 to .469 / .233 / .380.
- Early H12 pairs fall from .814 / .816 / .949 to .380 / .384 / .756.
- Later pairs where both maps split fall too, for example H8 from .950 / .908 / .897 to .745 / .472 / .409.

**Reading:** the high whole-map agreement is largely inflated by the shared root (no-search) membership described in §2. It does not prove the maps are noise, and not every non-root group is independent, meaningful evidence. The all-area ARI remains a correct description of whole-map agreement, but it needs this qualifier.

Root semantics in the maps must be resolved before these candidate pools are used downstream.

## 4. Proposed interpretation for the next design (not implemented, not approved for execution)

Separate **predictive fallback** from **spatial co-membership evidence**. An area with no search rows is legitimately predicted by the root model. That does not license the claim that it shares a learned region with every other area that had no search rows.

A candidate for examination is a minimal per-candidate assignment mask that reuses the existing `s-1` exclusion, so that unsearched root-labelled areas contribute no co-membership. Consensus math would stay unchanged. This is a proposal only. D31 science remains inconclusive; no final evaluation.
