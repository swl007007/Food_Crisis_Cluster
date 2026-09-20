# Design

Use an explicit local-import closure from the GeoRF entrypoint and Stage 2/3 scripts. Do not copy raw data, cached maps, outputs, alternative model implementations or machine-specific batch launchers. Preserve production namespace layouts so existing relative imports work inside the copied package. Remove the eager XGB visualization dependency in the copy.

## F1
For each group keep D=2TP+FP+FN and A=2TP. Existing scan algebra then obtains C=D-A, B=D*sum(C)/sum(D), and q=sum(C_subset)/sum(B_subset). Zero denominators give no scan evidence; do not create NaN or infinite q. Preserve real positive-count branch support checks separately; D is F1 exposure, not a sample count. Scan remains a proposal heuristic and geometric refinement remains unchanged.

At each split, evaluate parent predictions and trained child predictions on the same complete validation set. Compare the four existing parent/child checkpoint combinations by overall F1, keep parent on exact ties, and require strict delta >0.01 before accepting any split at any depth. No statistical p-value is attached. Synchronize inherited checkpoint predictions for metric recording. No effect-size or t-test path for this F1 mode.

Stage 2 continues consuming actual f1(1) and f1_base(1). Stage 1 SMOTE is already default-off; remove resampling support in the copy. Stage 3 partition fitting passes original rows directly to the same RF used by pooled. Record smote_enabled=false.

## Release
Record original source commit/hashes, changed copy hashes and local verified environment. Test extracted package outside the parent repository to detect accidental imports. Keep a local versioned archive and checksums; do not alter prior releases or publish remotely. Existing source feature and horizon conventions are carried forward, with limits listed for the next IPCCH spec.

## Confirmed retained behavior (2026-09-19)
User explicitly confirmed keeping the existing Stage 1 class-recovery pseudo rows: one zero-feature row per class, included in fitting. No-SMOTE does not mean zero synthetic rows. Tests must preserve this original count and distinguish it from Stage 3, which uses original rows only. Their effect in small or single-class branches is unquantified; revisit in IPCCH planning, not this release.

## Executable release contracts
1. Scope/trigger: package-only changes; standalone Stage 1-to-2 handoff and F1 split decisions.
2. Signatures: `select_f1_children(y0,y1,parent0,parent1,child0,child1,min_improvement=.01)` returns accepted, checkpoint choice, predictions, base/best F1; `scripts/run_stage1.py --data --polygons --experiment-dir --year --month --scope` runs one candidate.
3. Contracts: `GEORF_POLYGONS` selects boundaries. Step4 reads `<experiment-dir>/FEWSNET_admin_code_lat_lon.csv`. Root labels export as `root`, binary paths as strings. F1 is aggregated across the current parent branch's identical validation rows. Fraction arithmetic enforces exact strict delta>.01. NaN removal also filters correspondence metadata.
4. Validation/error matrix: unsorted `(FEWSNET_admin_code,date)` input raises before feature generation; empty candidate children retain parent; explicit SMOTE raises; incomplete monthly handoff or pre-existing destination raises without replacing results.
5. Good/base/bad: improved combined F1 accepts; tied parent remains; recall-only gains with excess FP reject. Sorted input works; unsorted input stops rather than silently misaligning groups.
6. Tests: `GeoRFBaseline/tests/test_baseline.py` and task `verify_release.py`; numerical boundary, root/nonroot gate, checkpoint persistence, unchanged original training rows plus retained Stage1 pseudo rows, standalone extracted imports and synthetic Stage2 chain.
7. Wrong/correct: per-row TP indicators on positive rows are recall, not F1; aggregate TP/FP/FN first. Averaged child F1 is not pooled parent-branch F1. No-SMOTE is not no-synthetic-rows because Stage1 retains two zero-feature class rows.

Spec-sync review: these contracts belong to the isolated package and are recorded here and in its README/tests. Root `.trellis/spec` and workflow documentation are intentionally unchanged under the approved two-directory boundary.
