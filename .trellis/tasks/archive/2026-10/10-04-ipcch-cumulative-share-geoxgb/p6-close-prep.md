# P6 close preparation: local Birdview exclusion and evidence-tracking fix

## 1. Unrelated Birdview HTML (supervisor decision, local housekeeping)

The first `trellis-audit close` (exit 1) returned before queueing anything:
"Please commit implementation before close: .birdview/architecture.html,
.birdview/architecture.sources.html". These two files come from an unrelated
read-only Birdview architecture survey; they are not task evidence. They were
untracked (`??`) with:

| File | Bytes | sha256 |
|---|---|---|
| .birdview/architecture.html | 722366 | 206e0d8f45f4b306b34880f8e63faa048eb8229c32c6f946b31d830fe52ca700 |
| .birdview/architecture.sources.html | 1085464 | 41bbfe74bd9b9ea7ee76abb67c55a56c193d40051981a2a4ac365160c54a4e38 |

The other `.birdview/` files (*.json) were already ignored by `.gitignore:24`
(`*.json`) and are unaffected.

Per the supervisor decision, the original `.git/info/exclude` (552 bytes,
sha256 5ff91e1998ae9dcc1eb72ba0ff8408598fde2a260ff0b298b0c99d1248596aaa) was
kept as an exact byte prefix. One comment and exactly two anchored entries were
appended (local only, not committed, reversible by deleting these lines):

```
# ipcch-geoxgb close prep 2026-10-04: unrelated untracked Birdview HTML (supervisor-authorized, local only)
/.birdview/architecture.html
/.birdview/architecture.sources.html
```

The new exclude file is sha256
2b833ec558603ce1dd41e1889c0b25b9fe8e6dab605507077330e4bdc0e36aff.
Verification:
- `git check-ignore -v` attributes exactly these two paths to exclude
  lines 20/21.
- The other Birdview, task and package paths, and `x/.birdview/architecture.html`,
  are not matched by the new lines.
- `git status --porcelain --untracked-files=all` became empty.
- Both HTML hashes are unchanged after the edit. The files were not committed,
  moved or deleted.

## 2. Evidence files skipped by repository ignore rules (executor omission, fixed)

The scope check showed that `git add <task dir>` in 62f185b/500a838/1f2ebff had
silently skipped 18 evidence files that the committed documents reference.
`.gitignore:10` (`*.csv`) or `:24` (`*.json`) matched them:
- `evidence/p6-formal-20261004b-final-inventory-post-replay.csv` (sha256
  7c96007f7311604200523cbf9e03e65576c5ebb92c5afb98f1e955257aa2b6d0);
- the 16 `evidence/p6-diag/diag_h*.csv`;
- `evidence/p6-logs/p6-formal-20261004-INCOMPLETE.json`.

They are now force-added (`git add -f`, these paths only). `git ls-files
--others --ignored --exclude-standard` over the task dir is empty afterwards.
This is bookkeeping: no file content, scientific run, code or fit output
changed.
