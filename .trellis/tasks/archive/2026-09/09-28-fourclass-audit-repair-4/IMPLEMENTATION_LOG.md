# Round-4 repair log

Findings: close-audit 2152a5d3 (repair-3: A01 Stage 2 ledger not reconciled at its Stage 3
consumer; A02 set-level route check; A03 verifier-mismatch test not driving production),
recheck ed300952 (repair-2 gate: same A01/A03 at an older commit, plus a RESULTS.md
misattribution), recheck b5eb9f46 (repair-1 gate: manifest.folds unchecked yet used by the
verifier loops and replay endpoint selection; the same RESULTS.md misattribution).
Recheck 8b837ad9 of the ORIGINAL task's gate fa38f19a completed with no major finding and
cleared that gate.

Class addressed: consumers trusted persisted records, and checks were aggregated.

- `src/utils/acceptance.py`: one chain prepared -> Stage 1 -> Stage 2 -> Stage 3 used by
  every consumer (Stage 2 builder, Stage 3 load_consensus and publish, report, verifier).
  - Stage 2: ledger and plan weights are re-derived from accepted Stage 1 evidence and
    must equal the persisted ones row by row; route must match the weights.
  - Stage 3 rows: cluster == consensus map; route == that cluster's single local_support
    decision (or unmapped / null reuse); pooled-routed rows carry pooled probabilities.
  - Stage 3 manifest: `manifest.folds` and `fold_records` must equal the schedule-derived
    population one-to-one, in order, with statuses and origins; fold records agree.
    Verifier loops and replay endpoints iterate `validated_folds`, never the manifest list.
- Exact CSV float parsing (float_precision="round_trip") on every read that acceptance or
  the verifier compares numerically; the v7 dry run showed the default parser changed two
  ledger values in the last bit.
- Cluster map recorded relative to stage2/ (a copied or cloned run resolves its own map).
- RESULTS.md: round-1 and round-2 paragraphs restored to the runs they introduced (v3, v6)
  with their producers; a run-history table lists every run with producer commit from its
  identity record.
- Tests (45, clean clone): consumer refuses forged ledger (omitted candidate, altered
  score) and altered weights with self-consistent hashes; row-level mixed route, wrong
  cluster, non-pooled probabilities on a pooled row refused; manifest.folds omission,
  reordering and relabelling refused; verifier-only mismatch rejected by the production
  `identity_problems` with a matching control.
- Auditors' fixtures: the area-79 mixed-route mutation is refused by the row-level check;
  omitting 2021-06 from manifest.folds on v8 is refused.
- Evidence gap (historical task.json for fourclass-audit-repair at completion c155ef3):
  that task record was not committed at the time (root *.json ignore rule, fixed in round 2)
  and no captured copy exists; it cannot be supplied without reconstruction, which the
  auditor forbids. The controller's durable run record (exported in round 3) is the
  available historical evidence. Disclosed, not reconstructed.
- Runs: v8 (39aa806) superseded within this round when the manifest.folds reconciliation
  followed; v9 from committed 23297d4 is authoritative.

## Stopped by user (2026-09-28)

The user stopped the work: "马上停掉，汇报科学结果，不值得继续跑。修改控制器状态，close掉这个任务。"
Run v9 (from 23297d4) finished but was NOT verified and is NOT committed. Committed evidence
remains run v7 (235c87b). The round-4 code (23297d4) is committed but not re-audited.
Controller gates 026f8908, 140f13de, 2152a5d3 were waived by that instruction (status
'waived', findings left unresolved; DB backed up first). This is not an audit pass.
The scientific result is unchanged across every compared run v2-v8 (tables byte-identical); v9 was not compared.
