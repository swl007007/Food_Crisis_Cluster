# Contract: GeoDT Branch Tree Diagnostic CLI

This contract documents the implemented exploratory diagnostic interface.

## Command

```bash
python scripts/plot_geodt_branch_tree_comparison.py [OPTIONS]
```

## Modes

Exactly one primary mode is selected:

- default figure-generation mode: run preflight, select pair, render figure, write metadata/audit outputs;
- `--audit-only`: run preflight, selection, scoring, readability checks, and audit output without rendering PNG/PDF;
- `--reproduce-from <metadata.json>`: reproduce a prior selection from metadata without reselecting by default.

## Archive discovery options

- `--archive-path <path>`: explicit preferred archive path evaluated first.
- `--archive-list <path1,path2,...>`: explicit deterministic candidate list.
- `--archive-root <path>`: bounded discovery root for GeoDT monthly archive candidates.
- `--artifact-provider-path <path>`: optional same-run/source folder that provides missing branch artifacts for the selected visual archive.

At least one archive input or configured project result/archive root must be available. The diagnostic must not search arbitrary user directories or the full filesystem.

## Output options

- `--output-dir <path>`: diagnostics output directory.
- `--overwrite`: allow overwriting existing diagnostic outputs. Default is no overwrite.
- `--png-only` / `--pdf-only`: optional output format narrowing if implemented.

If `--output-dir` is omitted, write under selected archive `diagnostics/output/` or configured diagnostic output root.

## Selection options

- `--k <int>`: top-K depth for split-feature signatures; default `3`.
- `--max-plot-depth <int>`: plotted depth; default follows K unless implementation chooses a stricter readable default.

Default ranking must remain top-K split-feature Jaccard distance. Threshold/direction fields are supplemental only and must not change selection.

## Required outputs by mode

### Audit-only

- Audit summary written or displayed.
- Metadata JSON or audit JSON if output directory is available.
- No PNG/PDF rendered.

### Figure generation

- Metadata JSON.
- PNG or PDF figure; PNG must be at least 300 DPI when produced.
- Audit summary or embedded audit record.

### Reproduction

- Reproduction report JSON containing the recorded selection and `reselected_pair == false`.
- No PNG/PDF rendered in the implemented reproduction report path.
- Structured reproduction failure summary on missing/changed recorded artifacts.

## Error contract

Failures after preflight begins, and reproduction mismatches after metadata loading, must produce or display a structured failure summary with:

- failure stage;
- missing or invalid artifacts;
- evidence mismatch if any;
- candidate archive decisions;
- recommended next action.

The CLI must not silently fall back to root/global `dt_rules` and must not create a misleading partial figure.

## Non-goals

- No GeoDT training.
- No prediction dispatch changes.
- No branch training changes.
- No `dt_rules` export changes.
- No writes into checkpoints, partition artifacts, production deliverables, existing `dt_rules`, or evaluation outputs.
