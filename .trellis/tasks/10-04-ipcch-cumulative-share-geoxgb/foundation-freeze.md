# P0 foundation freeze and Claude execution release — 2026-10-04

Codex accepts the foundation at **e1f0e6568398d5a6c2164909279abcff89ab6ab5**
(initial P0 fe2ea0dc plus bounded review corrections). The user's R51 sequence
is satisfied: the independent infrastructure is checked and frozen before
scientific implementation/execution. This record releases the same bound Claude
to **P1–P6 of implement.md**, without another user permission loop.

## Evidence reviewed

- Science/execution baseline remains `6c98f73c34272101ddc124cf20ed5ef338563646`.
  No frozen scientific value changed during P0 review.
- Independent checks inspected imports/CLI/config/schema/source and geography/
  cache against the P0 contract. Current frozen inputs and mappings passed.
  Two verifier gaps (missing raw-source area and fractional mapping truncation)
  were reproduced, then corrected by Claude; the main supervisor inspected the
  correction diff and the targeted tests.
- Codex independently reran the full P0 suite on the pinned Windows Python:
  `PYTHONPATH=. WSLENV=PYTHONPATH/p <Windows-python3.12.exe> -B -m pytest -q -p no:cacheprovider`
  from `IPCCHGeoXGBExperiment/`: **70 passed, 1 warning in 2.56s**, exit 0.
  The warning is geopandas' existing shapely.geos deprecation.
- `git diff 6c98f73c34272101ddc124cf20ed5ef338563646 HEAD --check` passed.
- All 21 package entries in `evidence/P0-code-identity.txt` independently matched
  their SHA256 and the Git blobs at the accepted foundation commit.
- Committed and run-local `p0-preflight-20261004b` reports are byte-identical:
  SHA256 `13777a955d6de5e406a57f76ee57280f4cbdaff97c22b14b00599a3f3e2e3724`.
  Status passed, 13 pinned inputs matched; 6227 areas, zero absent raw-source
  areas, 12411 undirected edges and 1231 isolated nodes. Full raw CSV preflight
  was run by Claude; Codex inspected its code-bound report and independently
  checked the geometry/cache subset rather than rereading 1.78GB again.
- Import/CLI checks from root and unrelated cwd found no legacy runtime imports,
  and pending science commands failed without changing run outputs. Ordered
  schema and source identities independently matched the frozen references.

The 1e-9-degree raw-panel/reference tolerance is accepted solely for decimal
serialization differences (observed max 5.0e-11); it never changes panel values
or evaluation keys. Missing ISO3/country_code and centroid/reference differences
remain disclosed inherited diagnostics. Upstream administrative-match/vintage
limitations are not resolved by this freeze.

## Execution and supervision

- Branch: `ipcch-cumulative-share-geoxgb`.
- Claude: pane `wN:p2`, session `afc97c82-11d4-4a79-ae5c-21d99cac82b5`,
  terminal `term_65d044bdb69412`.
- Audit run: `7ced754ea36c48c0a6d24ba2a17addec`, active; base_sha unchanged;
  controller running when checked. Do not restart, replace owner or close early.
- Codex supervises phase returns and evidence; Claude owns scientific code,
  focused tests, formal run and corrections. Keep the task-local PROGRESS.md
  current and report each phase commit/evidence before moving on. P1–P5 may
  proceed within the released plan; report implementation-ready identity before
  P6 formal fitting so Codex can confirm code/config/data boundaries.
- Freeze all required code/config/input/environment/schema identities for the
  formal run and winning maps before Stage3. Full P6 remains required; no claim
  of scientific completion from P0 tests or code-only milestones. Follow R41
  failure handling, no evaluation-led retuning, complete keyed evidence.
- GitNexus still has the recorded read-only shadow-page error; no index rebuild.
  Scoped Git/source checks remain the documented fallback while unavailable.

This is a **supervisor foundation acceptance**, not a controller close-audit
result. No labels/features/models/maps/predictions/bootstrap have run at this
freeze. Formal audit close and accepted result follow the full agreed delivery.
