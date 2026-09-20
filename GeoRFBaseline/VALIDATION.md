# Validation — 2026-09-19

Validated using Windows Python 3.12.10 and the versions in
`TESTED_ENVIRONMENT.json`. All checks below passed on a source archive extracted
into a fresh temporary directory outside the parent repository, with PYTHONPATH
removed. Imports were checked to resolve inside that extraction.

## Executed checks

- `python -B tests/test_baseline.py`: **11 tests passed**. Covers FP-sensitive F1;
  regional exposure/q; exact delta=0.01 rejection and above-boundary acceptance;
  parent ties; zero exposure/error; singleton group proposals; first-depth and
  non-root rejection; real checkpoint save/reload and reported metric agreement;
  inherited Stage 1 pseudo rows versus original-only Stage 3 training; identical
  pooled/single-partition RF probabilities; group validation membership; retained
  Stage 2 logit-F1 weights; correspondence root/leading-zero labels; NaN metadata
  alignment; sorted-input alignment and unsorted-input rejection; monthly handoff.
- `--help` succeeded for Stage 1, the monthly wrapper, Stage 2 steps 1/3/4/5/6,
  and Stage 3 (eight entrypoints).
- Executed Stage 2 steps 1/3/4/5/6 on two synthetic monthly partitions over 60
  areas. Verified a complete nonmissing `cluster_mapping_k40_nc3_general.csv`.
  This is software integration evidence, not a scientific result.
- Parsed every Python source; verified every extracted payload SHA-256.
- Verified all 37 original source-file hashes against `SOURCE_PROVENANCE.json`.
- Two independent static review passes checked the objective and package paths.
  Findings addressed: exact threshold arithmetic, empty candidate handling,
  portable Stage 1 handoff, missing fallback import, coordinates path,
  correspondence row/label preservation and explicit input ordering contract.

## Verification limits

No full Stage 1 or Stage 3 run on real data, clean dependency installation,
optional basemap download/rendering, performance benchmark, or IPCCH adaptation
was performed. Tests isolate spatial proposals to verify split gates; they do not
establish spatial-optimum quality. Stage 1's two class-recovery pseudo rows are
retained by explicit user decision; their predictive effect is unmeasured.
Inherited preprocessing, temporal availability and validation-selection risks
remain as described in README.

The task's `verify_release.py` rebuilds and checks this source archive. Exact
archive hash, per-command logs and final status are retained in the task's
`validation/` directory. The archive checksum is also in `releases/SHA256SUMS`.
