# D38 zero-fit prior/support diagnostic: findings (2026-10-02; no XGBoost fits)

**Scope:** same 21 D34 roots, snapshots restricted to ≤2020-12, D37 saved original E3 rows. No 2021+ labels or scores, no final ledgers, no production imports. Development evidence on exposed folds only.

## Inputs

- **Supervisor diagnostic:** `research/d38_prior_support.py` (sha256 `c09a2b91…`, byte-identical copy) → `research/d38_prior_support.json` (line endings normalised; original sha256 `bba1d4ac…`); originals in `C:\Users\swl00\geoxgb_runs\`. Fixed add-one 4×4 fitting-only transition counts; missing-origin fallback is the add-one unconditional fitting prior.
- **Executor independent spot-check:** `research/d38_executor_spotcheck.py` (sha256 `571a49de…`) → `research/d38_executor_spotcheck.json`; originals in `C:\Users\swl00\geoxgb_runs\d38_executor_spotcheck\`. It does not reuse the supervisor code. The origin label is taken from the label columns (`class_code`/`raw_phase`) of the frozen prepared snapshots at (area, m − H), over the union of the three snapshots, instead of from `hist_phase_o00`. This is independent of the derived feature, not an independent upstream raw-file lineage.

## Spot-check results (all agree with the supervisor's JSON)

- **Inputs reproduce:** fitting key hashes, fitting labels and E3 truth reproduce for all 21 roots.
- **Origin feature is exact:** the snapshot label at the origin month equals `hist_phase_o00 − 1` on every fitting row in all 21 roots, including the positions of missing values.
- **Argmax by origin class:**
  - (0,1,1,2) ×8
  - (0,1,2,3) ×7
  - (0,1,2,2) ×5
  - (0,1,1,1) ×1 (H12 2018-06)

  So the empirical prior is not a persistence anchor.
- **Pooled E3 matched keys** (112,795 rows):
  - prior-only F1 .456666, Brier .108995;
  - original root F1 .551516, Brier .101885;
  - persistence (one-hot) F1 .586406, Brier .146407.
- **Origin class 3 (4-or-5) support:** 268–682 fitting rows per root, so empty cells do not explain the argmax pattern. These non-empty counts do not by themselves prove sufficient effective support (rows of one area are autocorrelated).

## Label calendar shift (observed availability, not a bug)

| Years | Label months |
|---|---|
| 2010–2015 | Jan / Apr / Jul / Oct (quarterly) |
| 2016–2020 | Feb / Jun / Oct (triannual) |

- **H12 aligns within each regime;** the 2015→2016 regime boundary misaligns it (for example 2016 Feb/Jun targets have no label 12 months earlier). That is not all H12 missingness: areas newly covered or unlabelled at the origin month can also be missing.
- **H4 and H8 align only within the triannual regime:**
  - H4: Feb−4 = Oct, Jun−4 = Feb, Oct−4 = Jun.
  - H8: Feb−8 = Jun, Jun−8 = Oct, Oct−8 = Feb.
  - Under the quarterly schedule, the exact origin month carries no label.
- **Exact-origin-known fraction of fitting rows:**

  | H | Known fraction | Distinct label months, known-origin rows |
  |---|---|---|
  | H4 | .361 → .802 (2018-06 → 2020-06) | 6–12 |
  | H8 | .240 → .670 | 4–10 |
  | H12 | .856–.874 | 14–16 |

- **E3:** exact origin is missing for .00628 of rows overall.
- **Share of known-origin fitting rows with target year ≤2016:** H4 .494 → .243, H8 .495 → .195, H12 .935 → .480. By origin year the shares are higher, for example H4 .660 → .326.
- **Not supported by the data:** "rare or empty support" and "2013–2016 dominate the known-origin subset" are not claims to make for H4/H8.
- **Most missing-origin fitting rows still carry an earlier label.** `hist_latest_observed_age` is 2 for most H4 rows and 1–2 for most H8 and H12 rows, with only 272–1,322 NaN per root. This is recorded as a fact about the features only. It is not a proposal to substitute that label.

This is an observed shift in which exact-origin history is available between the fitting rows and E3. By itself it is neither a data-engineering defect nor proof of what causes the overfitting.

## E3 matched keys by origin class (pooled across 21 roots, descriptive)

| Origin | N | True crisis rate | Root crisis-call rate | Root Brier | Prior Brier |
|---|---|---|---|---|---|
| 0 | 57,633 | .027 | .003 | .0260 | .0266 |
| 1 | 35,841 | .204 | .059 | .1702 | .1657 |
| 2 | 18,365 | .593 | .635 | .2065 | .2557 |
| 3 | 956 | .848 | .886 | .1039 | .1310 |

- On origin-crisis rows (2 and 3), the root calls crisis about as often as crisis occurs, and its probabilities beat the empirical prior.
- Persistence calls crisis on 100% of those rows. Under crisis F1 that wins whenever the stay rate (.59) is high enough.
- One alternative explanation for part of the F1 gap is the effective threshold of the argmax decision rule, not missing persistence. This is a hypothesis, not established.
