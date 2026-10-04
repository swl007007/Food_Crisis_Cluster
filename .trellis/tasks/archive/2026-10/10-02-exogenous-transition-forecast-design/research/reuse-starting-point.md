# Reuse starting-point inspection

2026-10-02, planning-only. No model fitting, scoring or final-label access. This note locates existing contracts; it does not certify executable compatibility or select a model.

- Old task `experiment-plan.md:15–24`: XGBoost 3.0.0 native four-class CPU training, seed 42, nthread 4; G1 depth3/200 rounds; G4 depth4/400 rounds; L1 depth1/20 additional rounds.
- Same document `:176,179,233–234`: D26-selected H4=G1/H8=G4; D28 tested children initialised from the candidate root plus one L1, rather than cumulatively extending ancestors. Starting-checkpoint choice must be reconciled separately from capacity choice.
- `FEWSNETGeoXGBExperiment/README.md:19–23`: fork lineage and read-only four-class v7 RF references; old RF outputs use a 35-month window, so they are not a matched-window XGB control.
- README `:30–34,60–77`: 59-month fitting, native missing inputs, support requirements and existing split/local gating. README still describes no sample weights and older macro-F1 local gate. New augmentation weights and current crisis-F1 contracts therefore require code-level reconciliation; copying README defaults is insufficient.
- README `:87–99`: numerical environment and limits, including previously exposed 2021–2024 baseline results and unverified historical publication availability.
- Old task `research/stage1-research-decision-d54.md:7–18`: root and partition transfer weaknesses, binary-objective contrast not adopted, state-dependent drift, and no justification for rerunning adjacent parameter grids. D55 exception did not become a real run in the closed task.

Proposed reuse: fix G1/G4 and L1 capacities for the bounded A/B training comparison, retain original evidence, and inspect actual adapters/routing before implementation. The current code's sample-weight support, exact chosen local starting rule, reusable Stage2 maps, complete fitting counts and output lineage are not verified by this documentation probe. Do not infer that a completed/accepted GeoXGB Stage2 map exists from scripts or README commands alone.


## Follow-up source inspection

- Prior `d29-confirmation-plan.md:5,21–25` retained D28 shared-root single-L1 increments, comparison against the current routing parent, crisis-F1 Stage1 evaluation and distinct S/C/E3 roles. This is inherited design history, not proof of a now-generalising local model.
- `FEWSNETGeoXGBExperiment/src/model/native_xgb.py:175–205` supports validated global sample weights; `:211–223` continuation does not accept or pass weights. Reuse the weight validation, but propagate the new augmentation contract through local training during later authorised implementation.
- Same file `:259–265` support counts array rows; original-key support must be computed before augmentation or deduplicated by original identity.
- Same file `:297,325–350` defaults to parent increments but supports explicit root mode; in root mode a child gets only one local increment while routing parent remains its comparator. New execution must select the intended mode explicitly.
- `src/experiment/stage3.py:135–145` shared locals continue the fold global. `:151–171` still computes macro-F1 and uses strict configured gain (documented +0.01). Crisis-F1 alignment is necessary, not yet performed.
- GitNexus query was attempted and returned the existing LadybugDB read-only shadow-page replay error. Direct source inspection substituted; no index or product file was changed.
