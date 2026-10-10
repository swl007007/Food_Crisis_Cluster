# P0 bounded model/search sample, supervisor follow-up

Read-only scouts plus supervisor source spot-check. Ordinary implementation repairs under current P0 release; no project-data fit or EE. Samples stable: correction.py eb58910cb617d052d22c3528268706bba2544b556b5644c6853d25003458165b; partition.py 56c2a6ecd9bf98e19206b741d9f8c53a6b208cd4aac93fc805542c3c8c8e341f; backbone.py 62c65d081b524d79f309d2ecf8cc2e7c153236e6a43e13f454ffcc37cb9cccb2.

## Search

1. partition.py:252-257: tied set permits a selected gain <= tolerance even though best gain exceeds it. Require each eligible tied candidate itself to pass the strict gain floor. Repro without mocking: four admins, D=arange(4)[:,None], intercept-only V, equal weights, all four residuals sqrt(1.6e-12)*[1,1,-1,-1], lambda=.1, test support 1/1/1. Root objective 1.6e-12, gain at threshold0 is 4e-13, threshold1 1.333333e-12, tolerance 1e-12. Current code selects threshold0; contract requires gain>1e-12.

2. Nonempty correction pool with all routable=False must retain B2, but Search.__init__ -> admin_stats at correction.py:158 reshapes zero rows with -1 and fails. Even fixing reshape alone leaves partition.py:195-196 min/max on empty vals. Cover the complete all-unroutable path with exact B2 output, no swallowed technical error.

Positive sample evidence: independent row-level constrained KKT matched joint coefficients <=1.39e-16/objectives <=4.45e-16 with unsupported leaf, missing-AE rows, unequal weights, inactive slots. One supported leaf reused exact B2. Observed admin quantiles matched n=2,3,5,10,11,25. No complaint about core objective, centering or whole-joint rescoring.

## Backbone store (one bounded identity/recovery class fix)

3. backbone.py:107-124 checks pre-existing .partial only after four scalar fit calls. Check partial before fitting; regression must assert zero fits on refusal (probe observed four).

4. _load:138-145 verifies booster bytes but not stored resolved_config hash/rounds/base_score consistency. Changing only q2 record resolved_config learner_model_param.base_score to '999' while retaining its old hash is accepted. Verify retained config self-digest and model-recoverable fields; be mindful UBJ reload need not reproduce all training-only save_config fields.

5. load_by_digest:148-151 passes record identity back to _load, so identity is compared to itself. Changing only record.identity.fit_origin while retaining directory and identity_digest is accepted. Require requested digest, directory, recomputed identity digest and stored identity_digest to agree.

6. get_or_fit accepts identity made from Y, then fits Y+.5 and saves under the old y digests. Same boundary applies to actual X/weights/params/shape. Bind/check actual fit slices before cache reuse or fitting. This was an API-level synthetic reproduction, not observed production numerical drift. Preserve requested fit-key/row evidence in the eventual runner/replay as specified; no broad unrelated tamper suite needed.

Use small discriminating regressions, preserve old scratch evidence, continue the rest of offline P0. Send consolidated completion evidence at the checkpoint rather than waiting for supervisor approval after each routine fix.
