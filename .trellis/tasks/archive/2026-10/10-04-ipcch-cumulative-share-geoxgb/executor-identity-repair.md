# Executor attribution correction — completed 2026-10-04

The user explicitly authorized the proposal at
`/tmp/ipcch-geoxgb-executor-identity-repair-proposal.md` as a one-time exception
for erroneous active-run attribution and prevention of recurrence.

The same foreground Claude process (PID 1827, pane `wN:p2`, terminal
`term_65d044bdb69412`) remains the executor. Its current main conversation is
`d148c921-36bd-4b42-9ff9-a4f16979e6b5`. The original registered identity
`afc97c82-11d4-4a79-ae5c-21d99cac82b5` and subsequent
`1a58b878-577f-4c05-8e2e-cd54a9ae077d` belong to claude-mem observers that
inherited the pane environment. This was erroneous attribution from the
original registration, not a replacement of the foreground executor.

## Applied correction and evidence

- Installed `/home/swl007007/.claude/hooks/herdr-main-session-guard.py` and
  changed only the existing Herdr SessionStart hook command in Claude settings.
  Known observer working/transcript directories are filtered. Other payloads
  pass byte-for-byte to the unchanged managed hook. Seven isolated tests pass.
- Restored Herdr session metadata through `pane.report_agent_session`.
  A request without a lifecycle source was acknowledged but ignored. Source
  inspection of Herdr v0.9.3 (`terminal/state.rs:1385-1396`) explains this.
  The successful current correction used `session_start_source=clear`, replaying
  the actual earlier `/clear` reported by the executor. No `/clear` command was
  sent, no process was restarted, and no historical timestamp was fabricated.
- In a guarded SQLite transaction corrected only `repos.executor` and
  `runs.executor` for run `7ced754ea36c48c0a6d24ba2a17addec`.
  Original records, settings, hook and a consistent SQLite backup were retained.
  A durable `meta` incident entry was added; no history was deleted.
- Post-write comparison against the original backup verifies all other fields
  in the run and registration are identical. Base remains
  `6c98f73c34272101ddc124cf20ed5ef338563646`; run remains active, task remains
  `in_progress`; controller remains running. No start, close, reset or Git
  history rewrite was performed.
- Both known observer resume payloads were submitted to the installed guard;
  live Herdr identity remained the correct main session. Herdr, registration
  and run identity agree.

Durable backup and complete incident JSON:
`/home/swl007007/.local/state/trellis-audit-controller/19ea69ef13d88fde/identity-repairs/20261004-ipcch-observer-attribution/`

Managed hook unchanged SHA-256:
`7f117c303ffc1a66975b76dda8189d70b718fc04af4cf7cb11449e4ee5b4ae86`.
Guard SHA-256:
`50fb241168460620f30e358f7a41ae8424fce19fff7605d28dd2a5a3df3e9ca6`.

This corrects lifecycle attribution only. It does not constitute scientific
acceptance or release of formal P6. P2 correction/P3 review remains pending.
