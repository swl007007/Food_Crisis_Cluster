# Minimal GeoRF F1 baseline

## Goal and approved scope
Create an independent GeoRF stages 1–3 source package as the baseline for later IPCCH specification/task work. Keep parent code, raw data, experiments and published results unchanged. User approved full F1 alignment (regional statistics, q proposals, checkpoint/split selection) and SMOTE removal. User explicitly approved strict overall validation delta-F1 > 0.01 at EVERY depth, including the first; this is a performance gate, not a statistical significance claim. Stop and grill on any unresolved method change.

## Requirements and acceptance
- Regional F1 statistics include FP and FN, aggregate counts before computing F1, and preserve group validation membership.
- Scan proposals use C=FP+FN and D=2TP+FP+FN; q is subset (C/D) divided by parent (C/D). TN-only groups contribute no F1 exposure; no-error branches cannot improve.
- Parent versus combined child predictions is scored on identical full validation rows. No positive-class row filtering, weighted mean of child F1, or old per-row t-test. Checkpoints and recorded predictions must agree.
- No SMOTE in Stage 1 or either Stage 3 arm; preserve other RF parameters and data behavior.
- Preserve true Stage 2 F1/logit weighting and spatial constraints. No claim that proposal scan globally optimizes final F1.
- Deliver standalone source, runnable tests, dependency/environment evidence, manifest/checksums, README and a versioned local ZIP. No remote publication or full scientific rerun.
- Verify real code behavior with FP counterexamples, all-depth gates, zero-denominator cases, regional q, saved-child consistency and unchanged training sample counts; verify standalone imports and Stage 2/3 entrypoints.

## Boundary
Only GeoRFBaseline/ and this task are editable. IPCCH dataset decisions and model fitting belong to the subsequent task. This is a corrected code baseline, not a claim that all inherited preprocessing is forecast-time safe or IPCCH-ready.
