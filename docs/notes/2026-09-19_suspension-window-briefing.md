# 2026-09-19 — Evidence since the last leadership meeting, and a proposed redirection

Status: briefing document, written to support a leadership discussion. It records what was
established, what was refuted, and a proposed new direction **that has not been decided**. Every
number was computed from source data or saved artifacts in this session; where a claim is not yet
supported by evidence, it says so.

---

## Part 1 — What the work since the last meeting established

### 1.1 The published FEWS NET benchmark was mis-dated, and correcting it refutes the headline

The archived baseline applied a per-admin **record** shift to a source that is quarterly then
tri-annual, not monthly, which resolves to 12-16 (fs1) and 24-32 (fs2) calendar months of staleness.

| FEWS NET expert, crisis-class F1, 2021-2024 | fs1 | fs2 |
|---|---|---|
| Archived (mis-dated) | 0.588 | 0.491 |
| **Calendar-aligned (correct)** | **0.807** | **0.763** |

GeoRF partitioned scores 0.682 / 0.664 and lies **outside the expert's PR curve at every operating
point** — at the expert's precision it reaches recall 0.425 against the expert's 0.777. "GeoRF beats
the FEWS NET expert" does not survive and has been abandoned.

### 1.2 GeoRF loses to a one-line persistence rule

Persistence = carry forward the last observed IPC phase at `T-H`.

| crisis-class F1, 2021-2024 | persistence | GeoRF partitioned | expert |
|---|---|---|---|
| fs1 | **0.776** | 0.682 | 0.807 |
| fs2 | **0.709** | 0.664 | 0.763 |

Not a threshold artifact: GeoRF's best F1 anywhere on its fs1 PR curve is 0.716, still below 0.776.
The correctly-dated lag feature is present in the model. This remains **unresolved** and is the
single most consequential open problem.

### 1.3 The 2-layer persistence-correction gate failed

Persistence as base layer, one frozen global threshold on the model's probability as layer 2,
adjudicated against criteria fixed before any result was seen.

| | persistence | 2-layer | delta | bootstrap CI95 |
|---|---|---|---|---|
| fs1 | 0.7761 | 0.7741 | **-0.0020** | [-0.0139, +0.0112] |
| fs2 | 0.7085 | 0.7217 | **+0.0131** | [-0.0029, +0.0322] |

FAIL on all three pre-registered criteria. The decisive reason is the **ceiling**, not tuning:
sweeping the threshold with test labels gives at best +0.0054 (fs1) and +0.0140 (fs2), both under
the +0.02 minimum effect. fs2 captured 94% of what was available.

### 1.4 Two defects found in the existing pipeline

- **The partition criterion is class-1 recall, not F1.** `partition_opt.py:72-82` returns a
  true-positive indicator; false positives never enter the split criterion. The real F1 helper at
  `:40` is dead code. Documentation and the constant name both say F1.
- **SMOTE confounds the partitioned-vs-pooled comparison.** `train_partitioned_model` applies SMOTE;
  `train_pooled_model` does not. In all three scopes the partitioned arm has lower precision and
  higher recall than pooled — the signature of oversampling. So "partitioned > pooled"
  (fs1 +0.041, fs2 +0.048, fs3 +0.037), the project's one surviving claim, measures spatial
  partitioning **and** oversampling at once and has not been tested like-for-like.

### 1.5 The finding that reframes everything

**The free baseline is already close to the expert.**

| | persistence (free) | expert | gap |
|---|---|---|---|
| fs1, 2021-2024 | 0.776 | 0.807 | **0.031** |
| fs2, 2021-2024 | 0.709 | 0.763 | 0.054 |
| fs1, 2024 only | 0.803 | 0.849 | 0.047 |
| fs2, 2024 only | 0.731 | 0.813 | 0.083 |

A rule with no parameters, no analysts, no meetings and no field collection lands within 0.03-0.05
F1 of a system that requires all four. This is a policy finding in its own right and it does not
depend on any model we build.

---

## Part 2 — The FEWS NET suspension window

FEWS NET was suspended by USAID; the assembled panel
(`1.Source Data/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025_combined.normalized-v1.csv`)
shows the consequence directly. Label months run `... 2024-02, 2024-06, 2024-10, 2025-10, 2026-02` —
**2025-02 and 2025-06 are absent**, a 12-month hole.

What exists in the two post-gap months:

| month | `fews_ipc` (truth) | `fews_proj_near` / `fews_proj_med` (expert) |
|---|---|---|
| 2024-10 | 5,336 rows | 5,336 rows |
| 2025-10 | **5,718 rows** | **0 rows** |
| 2026-02 | **5,718 rows** | **0 rows** |

Truth came back — more completely than before, in fact — but **the expert product did not**. There
is no projection to compare against in either month.

### What the suspension does to the fallback baseline

| target | persistence source | effective lag | F1 |
|---|---|---|---|
| **2025-10** | 2024-10 | **12 months** | **0.6860** |
| 2026-02 | 2025-10 | 4 months | 0.7745 |
| 2026-02 | 2024-10 | 16 months | 0.6390 |

Normal-period comparison, so this is not over-read:

| target | source | lag | F1 |
|---|---|---|---|
| 2024-10 | 2024-06 | 4 months | 0.7392 |
| 2024-10 | 2023-10 | 12 months | 0.6373 |
| 2024-02 | 2023-02 | 12 months | 0.7372 |
| 2024-06 | 2023-06 | 12 months | 0.7298 |

**Read this carefully: persistence during the suspension is not anomalously weak.** 0.6860 sits
inside the normal 12-month range (0.637-0.737). What the suspension does is *force* a longer lag —
from 0.739 at 4 months to 0.686 at 12, a loss of about 0.05 — and remove the expert entirely.

Two further constraints:

- Only **2025-10** is genuinely "in the hole". By 2026-02 the 2025-10 truth exists, so persistence
  is back to a 4-month lag and scores 0.7745, better than normal-period 4-month persistence.
- The operational semantics change. Forecasting 2025-10 from the last publication means a **12-month
  lead**, not fs1's 4 months. The existing fs1/fs2 framing does not apply in the gap; it needs fs3
  or a new horizon definition keyed to "months since last publication".

---

## Part 3 — Proposed redirection (leadership decision pending)

The proposal, from the 2026-09-19 discussion, is to make the suspension window the main line and to
build **two parallel models**, evaluated separately:

1. **A cheap operational model for the normal period.** Secondary data plus an autoregressive term on
   the last available FEWS NET lag. Expected to lose slightly to the expert, but at a tiny fraction
   of the cost — no analyst meetings, no expert salaries, no field micro-evidence collection.
2. **A suspension fallback.** The same machinery operating when no expert product exists and the last
   observation is 12+ months stale.

The argument shifts from accuracy to **flexibility and economy**: an operationally-ready prediction
can be produced anywhere, at any time, at near-zero marginal cost. That story requires neither
beating the expert (nobody does) nor a large gain over persistence (there is little room).

### Why this is attractive

- Close to the present, policy-salient, and — per the team's assessment — an uncrowded or unexplored
  question, unlike crisis-onset prediction.
- It removes the two framings that the evidence has already killed.
- Section 1.5 supplies its strongest single number, and that number needs no model at all.

### What it still has to prove, honestly

- **The cost argument needs evidence from the normal period**, because that is the only place the
  expert exists. "Loses only slightly" must be quantified there and then carried into the gap as an
  assumption. Current candidates: 2-layer fs1 0.7741 vs expert 0.8070 (-0.033); fs2 0.7217 vs 0.7630
  (-0.041). Whether -0.03 to -0.04 counts as "slight" against a large cost saving is a judgement for
  the policy audience, not a statistical result.
- **The model still has to beat persistence somewhere.** In the gap the bar is 0.686 at a 12-month
  lag; GeoRF at fs3 (H=12) scores 0.6095 on 2021-2024, i.e. it currently loses there too. Changing
  the scenario has not changed the core problem from 1.2.
- **Sample size is very small.** Two target months, one of them genuinely in the hole. No
  conventional fold-level uncertainty estimate will survive that; an admin-level cluster bootstrap
  would be needed, and even then the evidence is thin.
- **"Secondary data only" is not strictly true.** The autoregressive term is a FEWS NET product, just
  a stale one. The honest description is FEWS-NET-light, not FEWS-NET-free, and the cost claim should
  be worded accordingly.

### Open decisions for the leadership discussion

1. Main line, additional chapter, or a separate follow-on project?
2. If main line: what replaces the fs1/fs2 horizon framing in the gap — fs3, or a new
   "months since last publication" definition?
3. Does the cost argument need an actual cost estimate (analyst-hours, field budget) to be credible,
   and who can source it?
4. Given two usable target months, what standard of evidence is the team willing to defend?

---

## Provenance

- Section 1: `.trellis/tasks/archive/2026-09/09-18-persistence-correction-2layer/` (with its
  `DECISIONS_LOG.md`, C1-C11) and `PersistenceCorrectionExperiment/RESULTS.md`; background in
  `docs/notes/2026-09-18_benchmark_and_direction_review.md`.
- Section 2: computed in-session from the assembled panel named above; no artifact was modified.
- Section 3: the user's proposal of 2026-09-19, recorded as proposed, not decided.
