# Repair round 4 — one acceptance chain at every consumer; row-level evidence

Remediates close-audit `2152a5d3a35974b5ff7b0eba` of `fourclass-audit-repair-3`. Earlier
requirements and the original approved specification remain binding.

## Findings and their class

- A01 (major): persisted Stage 2 ledger not reconciled at its Stage 3 consumer.
- A02 (major): route checks compared cluster SETS, so a single row routed against its
  cluster's decision passed.
- A03 (minor): verifier-mismatch test never drove the production predicate.

Class: acceptance logic lived at producers or was aggregated; consumers trusted records
and tests exercised helpers rather than the production gate.

## Requirements

- R1. One acceptance chain (prepared -> Stage 1 -> Stage 2 -> Stage 3) used by every
  consumer (Stage 2 builder, Stage 3 load/publish, report, verifier). Stage 2 acceptance
  re-derives the ledger and plan weights from accepted Stage 1 evidence and requires the
  persisted ones to equal them row by row, and the route to match the weights.
- R2. Stage 3 row-level: each prediction's cluster equals the consensus map; each route
  equals its cluster's single local_support decision (or unmapped/null); pooled-routed
  rows carry the pooled probabilities; null consensus rows are pooled reuse.
- R3. Tests drive production functions: consumer refuses forged ledgers/weights with
  self-consistent hashes; row-level conflicts refused; verifier-only mismatch rejected
  by `identity_problems` (with matching control).
- R4. Exact CSV float parsing everywhere acceptance compares numbers; cluster map
  recorded relative to stage2/. Fresh committed-code run; tables identical; verify.
