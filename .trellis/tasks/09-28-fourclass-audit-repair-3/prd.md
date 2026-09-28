# Repair round 3 — dynamically required inventories and verifier binding

Remediates close-audit `140f13de0d542e0668bb23d4` of `fourclass-audit-repair-2`. Rounds 1-2
requirements and the original task's approved specification remain binding; no scientific
choice changes.

## Findings

- A01 (major, data integrity). Inventories are checked against self-declared lists:
  Stage 3 `fold_records` may omit fitted folds; estimator entries may omit routed local
  models; retained Stage 1 checkpoints require only `rf_` though routing needs every
  branch checkpoint. Class: every inventory whose membership is determined by other
  evidence must be derived from that evidence, never from the record being checked.
- A02 (minor). The verifier is excluded from the committed-code equality check and its
  hash is marked passed unconditionally. R5 requires the verifier to be committed.

## Requirements

- R1. Derive required membership from authoritative sources and check both directions:
  - Stage 3 horizon: fold_records == scheduled Stage 3 months for that horizon (from the
    prepared schedule), each status == schedule status (fitted iff scheduled with rows);
    prediction rows == baseline truth keys for the horizon.
  - Stage 3 fold: model files == {pooled} + {local_c : routes in predictions and
    local_support say local_model}; local_support clusters == clusters in the target
    month; predictions' routes consistent with local_support.
  - Stage 1 fold (retain): checkpoints for every branch id in the retained s_branch /
    X_branch_id routing plus all ancestors saved; space_partitions complete.
  - Stage 1 population: completion records exist for exactly the scheduled folds.
  - Stage 2: candidate ledger rows == schedule; consensus outputs include every required
    file for its route.
- R2. The verifier file's hash is compared with its blob at the recorded run `git_head`
  and at HEAD; mismatch fails. Producer identity stays separate.
- R3. Focused rejection tests for each omission (omitted fitted fold, omitted routed local
  model, omitted branch checkpoint, missing Stage 1 fold, verifier-only mismatch).
- R4. Fresh run from committed code; tables identical; verification all pass; clean clone.
