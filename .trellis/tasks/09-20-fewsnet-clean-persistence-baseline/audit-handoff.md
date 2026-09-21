# Audit enrollment and executor handoff — 2026-09-20

## Authorization update

After design closure the user authorized pre-implementation audit registration,
task start, committing the planning/handoff materials and shared user-level
instructions. This supersedes the earlier planning-only prohibition on those
setup operations. It does not declare the experiment complete. The user will
give implementation/run instructions to Opus 5 separately.

## Intended executor and baseline

Verified live executor: Claude Opus 5, Herdr pane `w9:p2`, terminal
`term_65bde6b47922413`, Claude session
`482b4f11-ee6f-4e4f-9fe8-483327f20ad5`.

Commit the approved planning files before wrapper start. The controller's
`base_sha` must equal that resulting commit. The controller database/run record,
not this document or a pane name, is authoritative for the frozen identity/SHA.
Do not replace this session while its audit run is active.

## Setup and later completion

From the repository root, register the verified executor; the executor then runs:

```sh
trellis-audit start fewsnet-clean-persistence-baseline
trellis-audit status
```

Verify active run, expected base SHA and task status `in_progress`. Start records
audit lifecycle only; wait for the user's implementation instruction. Before
implementation, fully read `research/context-index.md` and all documents it
requires, plus `prd.md`, `design.md` and `implement.md`.

After the user authorizes execution, implement and run the bounded experiment,
fulfill A1–A64, retain reproduction evidence and perform the required checks.
Commit the implementation and evidence before calling:

```sh
trellis-audit close
trellis-audit status
trellis-audit show JOB_ID
```

Use the actual returned job ID. `close` archives and queues the completion audit;
do not run it now. Neither a planning commit nor native `task.py finish` triggers
an audit. Ensure the controller is running; use `trellis-audit boot` if needed.
Audit launch is not a passing result. Preserve accepted findings and follow the
controller's repair/re-audit flow for gates. Do not use native archive to bypass
the audit lifecycle or mark missing experimental evidence as complete.

The independent reviewer is a fresh Codex/Astra session, not a continuation of
this planning conversation. It reviews pinned repository/task evidence read-only.
Shared user policy lives in `~/.codex/AGENTS.md`; `~/.claude/CLAUDE.md` points to
that lifecycle section so the executor can load it too.
