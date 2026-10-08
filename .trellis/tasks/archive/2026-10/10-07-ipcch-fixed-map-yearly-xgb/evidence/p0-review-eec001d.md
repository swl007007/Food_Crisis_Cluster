# P0 supervisor review — pinned eec001d

Decision: HOLD P1 until the fixed pre-fit list below is addressed. User already authorized implementation; these are spec-compliance fixes, not new scientific choices. Do not refit/tune/change the approved scientific contract or 724-fit inventory. Review is read-only; this file is the handoff to the executor, not an audit result.

Verified: committed/current source inventory22 matches; actual input36 hashes match; no-fit enumeration independently equals research JSON:21G+160L quartets/724 scalar fits (main584). No demonstrated temporal-cutoff, weight-normalization, fresh-root or annual-routing error. Reviewers found no change needed to model recipes or maps.

## Fixed pre-fit list

1. Fit provenance (design103,107). engine.py95-97 records only hashes of selected rows/keys and the whole X artifact, while modelstore.py280-288 saves only boosters and record. Save actual ordered fit-row references and canonical fit keys in a small sidecar, bind its digest in identity/record, and hash the actual selected parsed X with explicit shape/dtype (y digests already exist). Validate sidecar/row/key/X digests against reconstructed inputs in cache/replay. Do not copy full X per model. A local fit still binds its own global. Add a focused row/keys/X tamper check. This must precede real fits so identities/evidence are complete.

2. Complete source freeze (design81,97,99). cli.py127-139 compares only env, whose FIT_SOURCES subset omits projection/metrics/report/replay/cli. Compare the complete saved source inventory in the run preflight against current inventory for predict/report/replay. cmd_report153-162 currently bypasses runtime/input context entirely: apply the same verification without fitting. If authorized report-only changes are needed after a run, preserve original inventory and explicitly reconcile permitted changes; never silently overwrite source identity. Test a projection/evaluator source mismatch fails before fitting/reporting. Current hashes did match, so this is enforcement failure rather than observed drift.

3. Input staging (design97). sources.py125-130 and contract.py69-74 still load frozen inputs from Dropbox. Stage the already-listed 36 input files once outside Dropbox under the fresh run or a hash-bound input snapshot; verify copied hashes, preserve source-relative identity, and read training/maps/comparators from that snapshot. No general storage framework or schema redesign. No need to copy unrelated source run artifacts. Re-run no-fit preflight and confirm unchanged724 inventory.

## Required before final acceptance (can fix with the same P0 patch)

4. Report local/persistence matched comparison (design9,87). report.py179-195 omits local-eligible intersect persistence-available cohort. Add its local and persistence panels/deltas with exact shared keys and coverage, preserving full-cohort primary G−P. Point estimates suffice; no new fits/CI required.

5. Independent metric coverage (design85,105). replay.py257-297 independently verifies E_all binary counts/F1 and main CI only; remaining comparisons via report.run_report reproduce the same producer. Independently verify the requested metric panel on E_all, E_persist and local cohorts: four-class confusion/accuracy/macroF1, binary accuracy/F1/precision/recall/F2, projected/raw q3 R2 and defined/NA semantics, plus the reported deltas. Use a short independent checker and focused synthetic mutation of a non-F1 metric/cohort; no broad framework or adversarial audit expansion.

## Return evidence

Commit bounded repairs, preserve scientific configs (if staging requires source-location metadata, distinguish it from the scientific recipe), run focused tests plus the existing small suite, rerun no-fit preflight/source inventory, and report new SHA plus exact counts. No need to rerun unchanged timing probes unless affected. Then stop for supervisor delta review and P1 release. Copy this fixed list or reference its contents in task evidence. Supervisor will not restart a whole-package audit after these fixes; subsequent pure report/replay issues are handled against actual outputs.
