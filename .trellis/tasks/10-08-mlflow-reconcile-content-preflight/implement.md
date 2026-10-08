# Bounded execution

1. Start via controller remediation for 14cfd4ce; report run/base/status.
2. Phase-1 content identity: load the record's retained manifests (small JSON downloads)
   and compare SHA256 per archived source/extras path and per bundle member with the plan;
   refuse before writes. Parent log: write once per previous fingerprint; on resume merge
   existing entries with new ones.
3. Add the two regressions; run all tests. Confirm the new tests fail on c22112d.
4. Commit, controller close, archive bookkeeping, report. Stop if round 3 repeats the class.
