# Supervisor P1 release — IPCCH yearly GeoXGB

Pinned implementation: 29c5fee188d0c8bcf724221b50639a437c8bfd58.
Active lifecycle run: 252b7ae51dcb4e53905e16f0de483088; base 7bd681deb3e6935b59c0af12ed9f792a961d9188.
Verified executor: Claude Opus 5.5 1M, wN:p2, session 174ea213-be60-4f2a-9bcb-e5e3523ad738, terminal term_65d3f9b51d7fa2.
Authority: user approved complete spec/execution; this is the agreed Codex P0 checkpoint release, not an independent close-audit pass.

## Delta acceptance
Reviewed only eec001d..29c5fee against the fixed list. Confirmed training row/key sidecars plus selected-X digest and replay reconstruction, complete source-freeze checks shared across CLI paths, verified external input staging, matched local/persistence reporting, and independent requested-metric panel verification. No scientific config changes in the diff.

Supervisor checks: documented locked Windows Python invocation from package root, PYTHONPATH=. WSLENV=PYTHONPATH/p, python -m pytest -q -p no:cacheprovider: 35 passed in48.73s. Recomputed current23 source hashes, all36 actual staged input sizes/SHA256 and staging-manifest digest: all match p0b-preflight evidence. Configuration byte-identical to eec001d;724-fit inventory unchanged. Lifecycle identity/base remain correct.

## Released scope
Proceed under a fresh run ID outside Dropbox with unchanged package29c5fee and fit_source_sha256 d7a693959b5ed6fd7c1907b4d1401d713d7ea3a4d44c419c2424cd5a72a3ed36. Run fresh preflight then predict (exact21 global+160 local quartets=724 scalar fits), followed by approved P2 report and zero-fit replay. No new real-data pilots, map/recipe changes, seed search or retuning. Preserve original runs and all frozen-source snapshots. Copy this release into task evidence without changing frozen package files.

Report progress after predict, then provide report/replay and final inventory for supervisor result acceptance. For technical failure preserve INCOMPLETE and report; do not silently retry/refit. Any source reconciliation requires a concrete supervisor-approved change; an authorized_by string alone is not authorization, and projection/metrics changes must never be assumed harmless to gate/routing. Pure report/replay defects may be repaired against actual outputs with explicit provenance; do not reopen a whole-package review cycle. No lifecycle close/archive, push or PR until final acceptance and the applicable user scope.
