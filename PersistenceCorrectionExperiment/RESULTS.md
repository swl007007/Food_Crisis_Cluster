# Persistence-correction 2-layer architecture — gate verdict

Run 2026-09-19. Task `.trellis/tasks/09-18-persistence-correction-2layer`.
Numbers below come from `outputs/phase5_20260919/adjudication.json`.

> **Independent check, 2026-09-19 (`trellis-check`).** Phases 4-5 were implemented *and* reported by
> the same party, so the original draft's claim that every number had been "recomputed twice by
> independent code paths" was not earned and has been removed. A genuinely independent
> recomputation has since been run: every headline number in section 1 reproduces exactly
> (confusion counts built by hand, no `persistencecorrection` import), the three structural claims
> hold, and 31 new contract tests were added. **Two interpretive claims did not survive and are
> corrected inline below, marked `[corrected 2026-09-19]`. The verdict itself is unchanged.**

---

# VERDICT: FAIL — null result

| criterion | requirement | fs1 | fs2 | |
|---|---|---|---|---|
| **R19** | `tau` selected without test data; fold bootstrap 95% CI excludes zero; survives LOFO | CI **[-0.0139, +0.0112]** includes zero; 3 folds flip the sign | CI **[-0.0029, +0.0322]** includes zero; LOFO clean | **FAIL** |
| **R20** | delta over persistence >= **+0.02** | **-0.0020** | **+0.0131** | **FAIL** |
| **R21** | fs2 must clear R19+R20; fs1 same-direction | delta is **negative** | does not clear | **FAIL** |

**Conclusion.** A persistence base layer corrected by a single global threshold on calibrated GeoRF
probabilities does not produce a demonstrable gain. fs2 moves in the right direction but by
two-thirds of the required effect and with a confidence interval spanning zero; fs1 moves the wrong
way.

**[corrected 2026-09-19]** The original draft continued: "the mechanism is not selectable out of
sample: the threshold frozen on 2020 captured +0.0131 of a +0.033 available ceiling at fs2 and none
of the +0.010 at fs1." **That is wrong, and it flatters the mechanism.** `+0.033` / `+0.010` are the
test-tuned ceilings on the **raw** probability — the C9 diagnostic variant, not the input layer 2
actually consumes. Sweeping `tau` on `p_cal` with test labels, the ceiling for the mechanism **as
built** is:

| | test-tuned ceiling on `p_cal` | realised | transfer loss |
|---|---|---|---|
| fs1 | +0.0054 (`tau` 0.9091) | -0.0020 | 0.0074 |
| fs2 | +0.0140 (`tau` 0.70) | **+0.0131** | **0.0009** |

So at fs2 the frozen threshold captured **94% of everything that was available**, and the ceiling
itself (+0.0140) sits below the +0.02 MDE. The binding obstacle at fs2 is **an absent ceiling on the
calibrated probability, not threshold transfer.** Threshold transfer does cost fs1 (0.0074), but
even a test-label oracle could not clear +0.02 in either scope. This makes the failure *more*
fundamental than the original draft said, not less.

**Next step (R23, pre-committed before any result was seen).** Stop FEWS NET work on this mechanism
and start IPCCH. Per R22 no variant is proposed, and none may be added now.

**Handicap that must be read with this verdict (C7).** The Phase 3 model-calibration gate (R5)
**failed** before Phase 4 began, and the task proceeded deliberately. The model remains
miscalibrated. **Mechanism failure and model failure therefore cannot be separated by this
experiment.** A better-calibrated model could raise the ceiling; this run does not rule that out,
and it does not support it either.

---

## 1. Adjudication detail

Test window 2021-2024, 12 tri-annual folds, n = 62,189 per scope. Only **persistence** has
adjudication power (R18); everything else is reported for context.

| model | fs1 F1 | vs persistence | fs2 F1 | vs persistence |
|---|---|---|---|---|
| **persistence** (base layer) | **0.7761** | — | **0.7085** | — |
| **two-layer override** | 0.7741 | **-0.0020** | 0.7217 | **+0.0131** |
| GeoRF partitioned, uncalibrated | 0.6833 | -0.0927 | 0.6632 | -0.0453 |
| GeoRF calibrated, standalone @0.5 | 0.5316 | -0.2445 | 0.4788 | -0.2297 |
| expert (calendar-aligned FEWS NET) | 0.8070 | +0.0310 | 0.7630 | +0.0545 |

Frozen thresholds: fs1 `tau = 0.6667`, fs2 `tau = 0.4762`, selected on 2020 only and hashed before
any 2021-2024 row was read.

### Robustness

| | fs1 | fs2 |
|---|---|---|
| delta | -0.0020 | +0.0131 |
| fold bootstrap (2000 draws) | mean -0.0020, sd 0.0063, **CI95 [-0.0139, +0.0112]**, P(delta>0) = 0.355 | mean +0.0131, sd 0.0089, **CI95 [-0.0029, +0.0322]**, P(delta>0) = 0.936 |
| leave-one-fold-out | range [-0.0066, +0.0005]; **3 folds flip the sign** (2021-06, 2021-10, 2024-10) | range [+0.0066, +0.0167]; no sign flips |
| folds with positive delta | **5 / 12** | 8 / 12 |

fs2 is the stronger of the two and still misses: its CI includes zero and its point estimate is
two-thirds of the MDE. fs1 is indistinguishable from noise in the wrong direction.

### Flip accounting

| | flips | fixed | damaged | test flip precision | `1 -> 0` flips |
|---|---|---|---|---|---|
| fs1 | 1,718 | 631 | 1,087 | 0.367 | **0** |
| fs2 | 2,732 | 1,208 | 1,524 | 0.442 | **0** |

Both scopes damage more rows than they fix. fs2's F1 still improves because the recall gained on a
0.29-base-rate positive class outweighs the precision lost. Up-only held structurally: zero `1 -> 0`
flips, as required by R15 and enforced by construction in `persistencecorrection/override.py`
(the only write is an assignment of `1` restricted to `persistence == 0` rows).

## 2. Pre-registered diagnostics — no candidate status

These were declared before the run and may not be promoted to results (R15, R22, C9).

**R14 consistency check — a declared red flag.** The threshold the first six 2021-2024 folds would
have chosen differs sharply from the frozen one in both scopes:

| | frozen `tau` (from 2020) | `tau` from first 6 test folds |
|---|---|---|
| fs1 | 0.6667 | **0.9091** |
| fs2 | 0.4762 | **0.3608** |

The threshold is genuinely unstable across windows, and R14 anticipated exactly this. It does not
authorise re-selecting the window.

**[corrected 2026-09-19]** The original draft called this "the cleanest single explanation of the
verdict". It is not, and the claim fails at fs2. The fs2 first-six-fold `tau` (0.3608) sits on the
*opposite* side of the frozen 0.4762 from the test-tuned oracle (0.70), so selecting on those folds
would have moved fs2 further from the ceiling, not closer; and the frozen fs2 `tau` already captures
94% of that ceiling. At fs1 the diagnostic is informative — its first-six-fold `tau` of 0.9091 is
exactly the oracle `tau`, and the frozen 0.6667 costs 0.0074. Instability is an fs1 story only.

**Raw-probability variant** (test-tuned, therefore an upper bound, not an estimate): fs1 0.7856 at
`tau` 0.57 (+0.0095 over persistence); fs2 0.7416 at `tau` 0.52 (+0.0331). These reproduce the
ceilings measured before the task started (+0.010 / +0.033), confirming the ceiling estimate was
sound and that the shortfall is in threshold transfer, not in the ceiling.

**Down-flip variant**: fs1 0.6042, fs2 0.4392 — far below persistence.

**[corrected 2026-09-19]** The original draft read those two numbers as "allowing `1 -> 0` is
catastrophic". They do not support that. The implementation used an **undocumented, unoptimised**
down-threshold of `1 - tau` (fs1 0.333, fs2 0.524), which silences 6,341 of 16,483 (fs1) and 9,367 of
15,631 (fs2) persistence-positive rows; R15 pre-registered the diagnostic but never specified the
threshold. Sweeping the down-threshold with test labels instead, **the optimum in both scopes is
zero flips, delta exactly +0.0000.** The defensible statement is that down-flipping is **useless
here** — its best achievable contribution is nothing — not that it is catastrophic. That matches
`docs/notes/2026-09-18_benchmark_and_direction_review.md` §4, which found the optimal down-flip count
to be 1-2. The up-only restriction stands on its pre-registered asymmetric-cost justification (R15),
which never depended on this number.

## 3. Why the model handicap matters

The Phase 3 gate result, recomputed on 2020 out-of-sample, post-calibration:

| scope | group | mean `p_cal` | crisis rate | \|gap\| | vs 0.05 |
|---|---|---|---|---|---|
| fs1 | persist=0 | 0.1609 | 0.0735 | 0.0874 | FAIL |
| fs1 | persist=1 | 0.4489 | 0.5574 | 0.1085 | FAIL |
| fs2 | persist=0 | 0.1637 | 0.0899 | 0.0738 | FAIL |
| fs2 | persist=1 | 0.4458 | 0.4665 | 0.0207 | pass |

Under the **original** R5 wording, 3 of 4 groups fail. Under the **corrected** R5 (C8), which scopes
the criterion to `persist=0` because R15 makes layer 2 read `p_cal` on no other rows, both scopes
still fail (0.0874, 0.0738 against 0.05). The correction changes the diagnosis, not the outcome.

The original R5 contradicted R15, and `design.md`'s falsification clause aimed specifically at the
`persist=1` group — the one group the mechanism never consumes. That drafting defect was mine. It is
recorded here rather than silently replaced by the corrected form.

Calibration did improve the group that matters (fs1 `persist=0` 0.1481 -> 0.0874; fs2 0.1200 ->
0.0738) and did not reach the bar. Note also that calibrated-standalone GeoRF at a 0.5 cutoff scores
far *below* uncalibrated GeoRF (0.5316 vs 0.6833 at fs1): isotonic shrinks an over-predicting score,
so fewer rows clear 0.5. That is expected, and it is why the override's threshold is not 0.5.

## 4. What this does and does not establish

**Establishes.** With this model's probabilities, a single global threshold over a persistence base
cannot clear +0.02 on either scope — and **[corrected 2026-09-19]** not merely because the frozen
threshold transferred badly. With test labels chosen for it, the ceiling on `p_cal` is +0.0054 (fs1)
and +0.0140 (fs2), both under the MDE. Threshold instability costs fs1 0.0074 and fs2 0.0009.

**Does not establish.** That no persistence-correction architecture can work. The model was
knowingly miscalibrated (C7); a per-window or adaptively-selected threshold was excluded by design
in favour of statistical power; and the raw-probability ceiling (+0.0331 at fs2) is materially
higher than the calibrated one, so the C9 decision to feed layer 2 `p_cal` is itself implicated and
was never adjudicated.

**Unchanged from before this task.** The expert still leads everything (fs1 0.8070, fs2 0.7630), and
GeoRF still scores well below a one-line persistence rule (0.6833 vs 0.7761 at fs1). That second
fact remains the most consequential open problem for any future direction, IPCCH included.

## 5. Provenance

- Selection: `outputs/phase4_20260919/frozen_thresholds.json`, sha256
  `52be8ad81f3f8f2d4b50a7b0929344fcb2506efbf4425aedf9a8d9ec6b48a29d`, with the sha256 of each
  selection input recorded inside and re-verified at application time.
- Adjudication: `outputs/phase5_20260919/adjudication.json` and `predictions_2layer_fs{1,2}.csv`.
- Specification changes made after Phase 2 began, with reasons: `DECISIONS_LOG.md` (C1-C9).
- Protected-input hashes (35 files) unchanged before and after every run; no frozen paper artifact
  was read-modified or written. Re-verified 2026-09-19: 35/35 still unchanged.
- GeoRF figures here (fs1 0.6833) come from the **regenerated** forest of DECISIONS_LOG C2, not the
  frozen paper artifact, and so differ slightly from the paper's 0.682 / 0.664. C2 explains why;
  R18 denies GeoRF adjudication power either way.
- C6 promised a 2021-vs-2024 calibration-drift diagnostic. It was **not produced**: Phase 3's
  reliability outputs cover `fit_2018_2019` and `selection_2020` only. Good for test discipline
  (no test label was inspected before the freeze), but the C6 diagnostic is outstanding.
- Contract tests for Phases 4-5: `tests/test_selection_override.py` (35 tests, added 2026-09-19;
  74 total in this package, passing under `python3` and `.venv-geodt-diagnostic`).
