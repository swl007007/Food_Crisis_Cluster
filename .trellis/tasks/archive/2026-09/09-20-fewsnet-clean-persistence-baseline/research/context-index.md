# Required context beyond automatic injection

All paths below are relative to this task directory:
`.trellis/tasks/09-20-fewsnet-clean-persistence-baseline/`.

This index prevents long evidence files from being silently truncated by the
32,768-byte per-file injection limit. It does not replace their contracts.
Read required documents in paginated chunks through EOF; an injected prefix or
this index alone is not sufficient implementation or review context.

## Authority and reading order

1. Read `prd.md` in full: final requirements R1–R68 and acceptance A1–A64.
2. Read `design.md` and `implement.md` in full for the executable design and
   ordered work. Planning is closed; implementation/run authorization is separate.
3. Before feature implementation or review, read
   `research/approved-feature-sources.md` in full. In particular, retain the late
   D46–D54 contracts: removed climate z-scores and assistance features, population
   timing, static snapshots, coastline extraction, predictor IDs, exact source
   whitelist, E whitelist/recipe widths and corrected-reference transforms.
   Earlier D23–D36 contain the preprocessing, A–E formulas, finite recipe search,
   additional-source whitelist and monthly/annual/persistence endpoints.
   The initial source inventory is not the final approved whitelist.
4. Before scheduling, consensus, Stage 3 or reporting implementation/review, read
   `research/fs3-and-time-support.md` in full. This includes D17–D22 calendars,
   support and selection; D40–D45 evaluation/bootstrap/stop rules; D55–D62 graph,
   completion, smoothing, fitting and valid single-cluster routing.
5. Read the injected `research/target-label-contract.md`,
   `research/monthly-source-validation.md`, `research/runtime-and-preflight.md`
   and `research/reuse-and-boundaries.md` for D63 target validation, D64 WB tie
   handling, source evidence, runtime identities and reuse limits.
6. `research/decision-log.md` preserves D1–D64 chronologically. Consult the
   corresponding decision when tracing approval; later approved decisions
   supersede earlier provisional wording. Historical questions are not new
   unresolved decisions. Escalate a genuine unresolved contradiction rather than
   inventing a scientific policy.

Keep the original research files intact. No automatic-injection setting or
scientific requirement was changed by this context curation.
