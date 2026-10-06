# Execution plan — IPCCH fixed-map MLP

Status: approved 2026-10-05. P0 checkpoint passed; P1+P2 released with replicate-parallel execution (PRD R25). The formal develop → predict → report → replay sequence is started by the user as a separate goal run. Executor: verified Claude Opus 5.5 1M session. The user supervises; no substitute model is authorized silently. Trellis audit waived for this task only.

## P0 — Establish identity, isolated package, and runtime

- Read PRD, design, research inventories and applicable repository instructions. Verify branch/worktree and preserve unrelated edits. Commit approved planning artifacts before implementation. The audit controller lifecycle is waived for this task (native start/archive).
- Create `IPCCHMLPExperiment/` with a small `ipcch_mlp` package, a CLI, tests, fixed scientific configuration, source-provenance and runtime locks. Keep original packages unchanged. Do not add a general estimator framework.
- Reuse minimal attributed copies of schedule/projection/metrics/support/report helpers. Record source-to-destination mapping, hashes and changes. Model, preprocessing, development selection, Stage3 and replay need new/adapted MLP-specific implementations; an old XGB replay pass is not sufficient.
- Before editing existing symbols, attempt required GitNexus impact analysis; report the known LadybugDB failure if it persists and use scoped source tracing. Do not reindex unrelated source packages to fix this task. Before commits run detect_changes and document its result/failure plus the actual Git scope check.
- Build isolated Windows Python3.12.10 environment and pinned PyTorch2.6.0+cu124 stack outside Dropbox. Record wheels/versions; do not upgrade the existing Python environment. Check RAM/GPU and free space on the actual Windows environment/run volume (normally C:), not the WSL volume.
- Implement data/source verification and no-fit request enumeration. Reproduce the planning counts, hashes, 19,052-row maximum global pool, and historical-support eligible main-period counts in design section 8; also enumerate their intersection with current fitting support. Fail before training on mismatch. Reuse existing prepared artifacts, not raw feature regeneration.
- Implement scalar network fits, preprocessing, additive quartet inference, exact-identity persistence and synthetic tests. The CLI must support at least `preflight`, `develop`, `predict`, `report`, `replay` with explicit config/run paths and immutable run identity.
- Implement zero-initialized P/L output layers with randomly initialized hidden layers; B initialization remains unchanged. Check initial zero corrections, zero-target behavior, nonzero-target learning, and serial identity-derived RNG isolation across intervening fits/cache hits. Record actual optimizer updates and unseen training-all-missing feature observations as specified in design.
- Run the fixed synthetic CPU/CUDA timing and determinism probes, including n=19,052; freeze the eligible device, environment and numerical recipe. Report estimated full-run wall time/storage, 13,260 scalar fit inventory, source verification, code identity and synthetic checks to the user.
- **Checkpoint:** supervisor checks concrete frozen code/runtime/count evidence before real-data fitting. No project-data pilot or tuning is implied by P0.

## P1 — Development and recipe freeze

- Run the three replicates as concurrent worker processes (PRD R25); the parent selects only after all three finish.
- Fit exactly 288 scalar networks across H/seed/target/global size/residual size; globals reused across residual-size candidates.
- Fit preprocessing and all learners on original F; score complete original S in inference mode. Save raw components, summed/projected outputs, labels, counts, seed scores and exact ranking.
- Select one recipe per H by mean seed P crisis F1; ties use fewer B+P parameters then fixed candidate ID. Freeze winners and source identity; do not refit on F+S or train development regional networks.
- Record and report candidate losses/selection evidence, distinguishing internal S from independent testing. Undefined-all or technical failures stop according to the contract.

## P2 — Stage3 with ungated diagnostic

- Run the three replicates as concurrent worker processes with separate ledgers (PRD R25); finalize the Stage3 summary only after all three succeed.
- Use the fixed original maps and all 138 planned folds for each replicate. Preserve no-valid-target rows and original cohort keys.
- Fit B/P and all supported historical L quartets at their lawful origins; cache by full identity. Recompute historical gates per seed and origin, with P used on unsupported historical dates.
- Fit all supported current L quartets with current keys, irrespective of gate; export diagnostic predictions. Build G only using approved gate routes, otherwise P.
- Preserve full per-fold support, route reasons, historical pairs, model requests and current B/P/G plus eligible L predictions. Keep all four targets atomic.
- Reconcile 12,972 Stage3 scalar fits and complete request/provider inventory; no extra seed, capacity, epoch or region retry after errors. Scientific failures and technical failures have different reporting outcomes.

## P3 — Report, replay, and supervisor acceptance

- Recompute complete per-seed metrics for B/P/G, original XGB references and persistence on exact cohorts. Report main/supplementary separately, seed means/ranges and all matched deltas. Retain G−P as the primary full-cohort deployment contrast and L−P as the supported-cohort diagnostic. Report predeclared historical coverage, its current-support intersection and actual adoption separately; do not interpret row shares as F1 or significance bounds. Include optimizer-step differences and unseen missingness-pattern counts among the interpretation limits.
- Reproduce per-seed main-period G−P and G−persistence country bootstrap with the original policy; do not pool independent-looking seed copies or invent seed-averaged confidence intervals.
- Run saved-model replay for development, historical and current predictions and independently reconstruct all selection, gate, route, metric and bootstrap results.
- Validate synthetic negative cases for data/map/model tampering and true adopted-local behavior. Verify production source diff stays within the new package, task evidence and authorized notes.
- Retain a complete final inventory with hashes, model counts, runtime/code identity, failed/partial-run history and commands. Large scratch model files remain outside Git.
- Prepare a results note and concise additions to the existing IPCCH meeting/future-direction notes. Report negative results without retuning. The user checks the delivered evidence before scientific acceptance.
- Commit reviewable deliverables only after required checks; do not infer push/PR/close-audit authorization from this planning task. Follow current user instructions for final lifecycle actions.

## Required reproducible command surface

Locked interpreter `C:\Users\swl00\.venvs\ipcch-mlp\Scripts\python.exe`; package `IPCCHMLPExperiment/`;
config fixed inside the package (`config/*.json`); scratch runs under
`C:\Users\swl00\AppData\Local\Temp\ipcch-mlp-runs`. From WSL in `IPCCHMLPExperiment/`:

```text
VP=/mnt/c/Users/swl00/.venvs/ipcch-mlp/Scripts/python.exe
RUN='C:\Users\swl00\AppData\Local\Temp\ipcch-mlp-runs\<formal-run-id>'
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m pytest -q
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp preflight --run-dir "$RUN"   # no fitting
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp develop   --run-dir "$RUN"   # P1, 3 replicate workers
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp predict   --run-dir "$RUN"   # P2, 3 replicate workers
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp report    --run-dir "$RUN"
PYTHONPATH=. WSLENV=PYTHONPATH/p "$VP" -m ipcch_mlp replay    --run-dir "$RUN"
```

`develop`/`predict` accept `--serial` for the serial mode. `run_experiment.py <command> --run-dir DIR`
is an equivalent wrapper. `preflight` and `timing` never fit project data.

## Rollback and recovery

Before P1, implementation fixes require synthetic revalidation and a new code freeze. After scientific fits start, technical changes require preserved INCOMPLETE evidence and an explicit impact assessment; never overwrite an existing identity or reuse incompatible models. A scientific recipe change returns to planning. Recovery does not permit dropping failed cohorts or replacing seeds. Original P6 artifacts remain immutable throughout.
