# IPCCH planning handoff — 2026-09-20

Authoritative requirements: prd.md v1.0. design.md and implement.md v1.0 now exist.
Task remains **planning**. All scientific decisions through Q9b and Q8g are
approved; final-summary implementation approval is pending. No model fitting,
geometry repair, transformed panel, task activation, commit or push occurred.

## Latest approval and convergence

User approved Q8g: make_valid only on invalid shapes in an experiment-local copy,
preserving original source, IDs/reference coordinates and valid shapes. Audit
changes; stop on remaining invalid/empty/non-polygon results or ambiguous identity.
No silent component extraction, area deletion or alternate boundary source.
Live read-only check found253invalid of6,227shapes; main thread reproduced counts
with Windows Python/GeoPandas1.0.1/Shapely2.1.0. Evidence: source-audit.md and
convergence-check.md. Repair success is not yet known.

Q9b stays approved:1000paired country-cluster bootstrap draws, seed42,95percent
F1/difference intervals, main horizon/cohort only, fixed predictions, explicit
undefined/effective-replicate rules. Exact protocol is in evaluation.md.

The PRD convergence pass condensed the decision history into R1–R6/A1–A8 without
authorizing new science. Research files retain supporting traces and superseded
proposals as historical evidence. Two independent read-only checks found no lost
scientific decisions/acceptance IDs or spec/design/plan contradictions. Citation
paths flagged during retention review were restored. Main thread read the final
PRD end-to-end and verified decision-ID/acceptance retention, source references,
93-feature arithmetic,122-fold schedule, planning state and context sizes.
Trellis validation passed with10entries in each context manifest; no truncation.

## Design boundaries

- New IPCCHGeoRFExperiment only: prepare_data.py, run_pipeline.py,
  report_results.py, compact test_contracts.py and README; no new framework.
- Extract pinned baseline into each fresh run; keep source archive/root modules intact.
- Only required core compatibility patch: skip final grid-only refinement when
  contiguity_type=polygon. Retain training's polygon scan/refinement and F1/q gates;
  reconcile accepted s_branch/checkpoints/export. This avoids post-fit grid reassignment.
- Prepared X uses93columns:70raw,15derivatives,2calendar,3history,2recency,1horizon.
  Explicit split and train-only shared RF imputer; no legacy feature preprocessing.
- One2014–2022Stage1, per-area chronological half split before horizon expansion.
  Genuine support counted as original outcomes; current zero/nonempty core support
  gates need no q rescaling. Singletons supplementary only.
- Donor reference-coordinate distance<=100km; genuine learned support/assignment
  required, no chaining; unresolved/unseen/small/single-class local models use pooled RF.
- Stage3 monthly36calendar-month fits at horizons1/3/6/12, origins>=2023-01,
  main through2025-12, partial2026separate. Same training eligibility and paired cohorts.
- Reuse Stage3 probability fallback path, not hard-label helper's unseen-group gap.
  Fixed p1>.5; XGB uses explicit constructor; persistence latest valid history<=O.
- Preserve baseline diagnostic behavior, including optional inherited CV outputs;
  they are not another partition/model arm or final-test evidence.

## Evidence and limits

Pinned input SHA256:
ae696087c3bbb280537ae269a05924133acdb51060d31290523404fa8a717673

Pinned GeoRFBaseline ZIP SHA256:
39a26138e3fafb0be2bbd22e9760095d6cdefa7d79b98b4f798cb3aa79b500a0

R1 source audit:42,695valid/15,206positive. Approved Stage1 capacity:
19,591original labels;3,264n>=2areas with8,561fit/9,558validation outcomes;
1,472singleton and1,491zero-label areas. No model results measured.

Publication timing is unverified; feature construction assumes observation-month
availability. Candidate upstream EVI/nightlight issues are not proven inherited
by selected CSV. Geometry topology repair will not prove area identity. Missing
ISO3 must not drop Cote d'Ivoire country rows. Internal per-area cutoffs are not a
global forward-validation split. Scientific success does not require superiority.

## Next action

Planning artifacts are ready for the final summary. Wait for the user's subsequent
explicit approval before task.py start or implementation. If a consequential
ambiguity emerges during implementation, ask one grill question.
Do not interpret Q8g approval as final-summary approval.

Previous minimal-georf-f1-baseline task is archived; its package and journal changes
remain uncommitted and must be preserved. This task is the only active planning task.

---

# Closure — 2026-09-20

Implemented, run, verified and committed as `3addf9a`. Authoritative run
`IPCCHGeoRFExperiment/runs/ipcch-v1-20260920d`; results and limitations in
`IPCCHGeoRFExperiment/README.md`; every scope change in `DECISIONS_LOG.md` D1-D14.

## Acceptance

| ID | Status |
|---|---|
| A1 | Source, release and shapefile hashes match and are unchanged after the run; runtime, the single local patch and output lineage recorded. |
| A2 | R1 order, fill, bounds, sum and normalization reproduce 42,695 / 15,206 / 27,489 / 6,227 / 84 / 2,601 with reconciled exclusions; the five normalized components are preserved. |
| A3 | `origin == target - horizon` on all 170,780 feature rows and all 81,109 predictions; Stage 3 training is exactly 36 calendar months ending at the origin; every origin >= 2023-01. |
| A4 | One pooled Stage 1, no Stage 2; released F1/q mathematics untouched; split reproduces 19,591 / 3,264 / 8,561 / 9,558 / 1,472 / 1,491; Q8g stop honored and its narrow extension recorded; donor distances <= 100 km with no chaining. |
| A5 | `E_persist` is the history-available subset of `E_all`, verified against the R1 ledger rather than the runner's column; arms share identical keys; missing persistence and fallback counts explicit. |
| A6 | Saved rows reproduce metrics, deltas and intervals; `p1 > .5` with ties to 0; XGB parameters verified from the booster; no outcome-driven retuning. |
| A7 | All specified comparisons reported, including the negative ones; no winning-model claim. |
| A8 | 70 raw + 15 derived + 2 calendar + 3 history + 2 recency + 1 horizon = 93; train-only RF fills; XGB native NaN. |

122 contract checks pass under the pinned Windows runtime.

## Known gaps, deliberately not closed

- GitNexus MCP tools were unavailable in every agent session; blast radius was
  established by explicit caller search instead, and that limitation is recorded rather
  than papered over.
- Three run directories from development remain under the ignored `runs/`; `…20260920a`
  is preserved failure/dev evidence per design.md.
- PRD R2's original prohibition text is intentionally left unedited beside its amendment,
  so the original requirement and the approved narrowing are both visible.

## Next

Relaxing the 0.01 gate, parameter tuning and feature engineering are out of scope here and
need a new task with fresh pre-registration (`DECISIONS_LOG.md` D14). The rejected split's
margin is 0.007434, so any relaxed threshold must be declared before the run.
