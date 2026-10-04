# Baseline verification — 2026-10-04

- User R51 authorizes the ordered execution contract. Two independent read-only
  checks found no science-contract conflict. Inventory clarifications were added
  for geography dependencies, country/coordinate exceptions, mapping direction,
  bounded transitive dependencies and cross-repository source path bases.
- 14/14 reference sources match their recorded current byte hashes and committed
  versions. 13/13 input paths exist and byte lengths match; planning input hashes
  are preserved. Structural pickle/geometry cache consistency is still P0 work.
- Read-only small-table/DBF inspection found 6227 unique area IDs with matching
  sets across lookups/coordinates/DBF; 53 country keys; missing ISO/code values
  and centroid differences are permitted under the inherited semantics.
- Live Windows metadata verifies Python 3.12.10, XGBoost 3.0.0 and the locked
  package versions. Windows interop required sandbox escalation; no installation
  or numerical fitting occurred. Linux handles documentation/Git only.
- `task.py validate` passes both 23-entry contexts. Its expected design.md size
  warning requires an explicit full-file read, recorded in implement.md.
- `git diff --cached --check` passes. GitNexus `detect_changes(scope=staged)`
  was attempted and failed with LadybugDB read-only shadow-page replay error.
  No reindex was performed. Scoped staged inventory contains only this task's
  documentation/context/manifests; no code symbol changed.
- P0 tests, input/cache structural validation, foundation freeze, scientific
  implementation, fitting, bootstrap and controller acceptance remain pending.
