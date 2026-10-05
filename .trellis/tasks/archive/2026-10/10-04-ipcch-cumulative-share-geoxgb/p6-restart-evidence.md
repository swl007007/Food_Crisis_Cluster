# P6 restart after the file-lock R41 stop (operational evidence)

Authority: user choice "pause Dropbox sync, rerun with frozen code and a new run
ID" (AskUserQuestion, 2026-10-04); supervisor note `p6-file-lock-restart.md`
(copied verbatim from /tmp/ipcch-geoxgb-p6-file-lock-restart-20261004.md);
user confirmation "已暂停" (Dropbox sync paused) before any rerun step.
Audit run 7ced754ea36c48c0a6d24ba2a17addec / base 6c98f73 unchanged: not
restarted, not rebound.

## Run IDs

| Role | Old (failed, preserved) | New |
|---|---|---|
| preflight | p6-preflight-20261004 (passed) | p6-preflight-20261004b (passed) |
| scientific run | p6-formal-20261004 (INCOMPLETE, R41) | p6-formal-20261004b |
| file probe | — | p6-fileprobe-20261004b (no fit) |

## Failed attempt p6-formal-20261004 (kept intact)

- prepare passed; learn-map stopped at the first fit (H1, G1, root_global
  quartet) at `os.replace(<identity>.tmp, <identity>)` in
  `ModelStore._get_or_fit`: `PermissionError [WinError 32]` (file in use by
  another process). WinError 32 is verified. Dropbox.exe was running and is the
  suspected lock source; the process that held the handle was not identified,
  so the attribution is not proven.
- Durable R41 evidence untouched by the restart:
  `RUN_INCOMPLETE.json` = `stage1/INCOMPLETE.json` sha256 4bac980c…d9ef2,
  `stage1/model_requests.jsonl` sha256 29f2fe5d…14cb (one failed request),
  partial entry `models/a1/a1b3f6a4…43fe.tmp/` (four .ubj + record.json),
  never promoted and never read by the new run.
- No Stage1 map was completed.

## No-fit probe (before any new scientific step)

`evidence/p6_file_probe.py` under the pinned Windows Python, against
`runs/p6-fileprobe-20261004b`. It mirrors the model-store persistence: sibling
`.tmp` dir, 4 × 256 KiB payloads + record.json, `os.replace`, sha256 read-back.
It ran 8 entries in one pass and would have stopped at the first failure, with
no retries. Result: **passed** 8/8, rename 0.65–0.77 ms each
(`evidence/p6-fileprobe-20261004b-result.json`). Dropbox.exe processes were
still present (sync paused, not exited).

## Frozen identity for the new run

- HEAD 44951cc; `git diff 6798df2 HEAD -- IPCCHGeoXGBExperiment/` empty
  (package code and configs = implementation identity 6798df2). No package code
  change, no retry logic, no output-root redirect, no Dropbox ignore marking.
- Runtime: the same pinned Windows Python 3.12.10 (`python3.12.exe` from
  WindowsApps). Preflight b passed; its report differs from preflight a only
  in `elapsed_seconds` (10.7 → 9.5).
- Fresh prepare b passed in 56 s. All 14 prepared artifacts are byte-identical
  to run a. The manifest differs only in `elapsed_seconds` (64.0 → 56.2), so
  its sha256 is d4cb9bfd… (b) vs 3d4866ba… (a).
- Determinism cross-check (read-only comparison, nothing reused): run b's
  first fit (H1 G1 root_global, identity 161e5ca0…) has four booster sha256
  byte-identical to the boosters inside run a's unpromoted `.tmp` entry. The
  two identity records differ only in `prepared_manifest_sha256`.

## Outcome of the restarted run

learn-map `p6-formal-20261004b` completed: exit 0, 429 fits, 0 failed
requests, no incomplete marker. See `p6-checkpoint-evidence.md`.
