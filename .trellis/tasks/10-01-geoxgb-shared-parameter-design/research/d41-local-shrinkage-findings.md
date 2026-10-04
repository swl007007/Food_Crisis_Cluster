# D41 frozen-map local-increment halving: findings (2026-10-02; zero fits)

**Scope:** D34's 21 Brier candidates (producer 7b2bf6f), maps frozen, C and E3 only. The half arm is a geometric shrinkage of the saved probabilities, `p_half ∝ sqrt(p_root·p_local)`. It is margin-equivalent for ideal probabilities but is not a byte-exact replay of the raw margins.
- Identity (zero-increment) rows copy the root.
- There is no alpha grid, endpoint change, map change, Stage 2/3 or 2021+ data.
- Planning commit 15b1085 preceded the run.
- This is a fixed-map sensitivity control. It neither shows that search overfitting is fixed nor that the local increments are not over-fitted.

## Evidence

- **Supervisor script:** `research/d41_local_shrinkage.py` (sha256 `aa24e4c2…`).
- **Summary:** `research/d41_summary.json`, line endings normalised; original sha256 `84140488…`.
- **Bulky per-row output:** kept external at `C:\Users\swl00\geoxgb_runs\d41-local-shrinkage-20261002\rows.csv.gz`, sha256 `a36104f6b2ffa966e8328b88a9960fd082dd2648b9a2b8ff01022027b2a7df3e`.
- **Run:** frozen Windows interpreter, self-checks before the diagnostic, exit 0. Root and full reproduce D36 exactly.
- **Executor independent check:** `research/d41_executor_check.py` (sha256 `39cccd43…`) → `research/d41_executor_check.json` (line endings normalised; original sha256 `0b7892be…`). Python 3.12.10, numpy 2.2.6, pandas 2.2.3. It does not import or rerun the main script.
  - **Inputs:** rebuilds keyed sources from the D34 candidates, roots, memberships and checkpoint records, and ≤2020-12 snapshot persistence.
  - **Half arm:** softmax of the mean log-probability.
  - **Result:** 2,938 comparisons, 0 issues. An earlier run counted 2 trivial no-op checks; they were removed and the checker rerun.
  - **Rows:** all 321,047 output keys are unique and complete. Root and full probabilities are exact; half differs by at most 4.4e-16; labels are exact; no near-ties in the half arm (none closer than 1e-9).
  - **Provenance:** 155 true-local continuations and 32 zero-increment branches (21 `root` ids plus 11 hash-equal fresh root copies). Identity rows equal the zero-increment rows exactly (C 60,234; E3 32,490).
  - **Summary blocks compared:** the root/half/full blocks (n, confusion, F1, Brier, changes vs root, half-vs-full) of all / matched / per horizon / per root / transitions / routes, and the fold means.
  - **Persistence:** counts and Brier compared only where present in the matched block and the four origin-known transition groups. Other optional persistence fields were not compared directly.
  - **Also checked:** D36 root/full reproduction.
- **Supervisor check:** all 217 reported F1 values were checked directly against their confusion counts, including persistence.

## Results (TP/FP/FN, crisis F1, crisis Brier)

**E3, all keys (113,508 rows):**

| Arm | TP / FP / FN | Crisis F1 | Crisis Brier |
|---|---|---|---|
| Root | 9,767 / 5,029 / 10,924 | .550455 | .101890546 |
| Half | 9,803 / 5,084 / 10,888 | .551071 | .101884060 |
| Full | 9,845 / 5,191 / 10,846 | .551124 | .101950206 |

**E3, persistence-matched keys (112,795 rows):**

| Arm | TP / FP / FN | Crisis F1 | Crisis Brier |
|---|---|---|---|
| Root | 9,761 / 5,029 / 10,846 | .551516 | .101884595 |
| Half | 9,797 / 5,084 / 10,810 | .552130 | .101878271 |
| Full | 9,839 / 5,191 / 10,768 | .552179 | .101945013 |
| Persistence | 11,707 / 7,614 / 8,900 | .586406 | .146407 (one-hot) |

**E3 detail:**
- **Fold means vs root:** half F1 +.000644, Brier −.0000086; full F1 +.000737, Brier +.0000555.
- **By horizon, Brier root / half / full:**
  - H4: .090834 / .090734 / .090718
  - H8: .110754 / .110941 / .111168 (worsens as amplitude grows)
  - H12: .104083 / .103977 / .103965
- **True-local rows only** (81,018): Brier .121295 / .121286 / .121379. Zero-increment rows (32,490) are identical across the three arms.
- **Half vs full:** 307 crisis-decision flips and 654 four-class flips.
- **Transition groups, half vs root:**
  - 00: FP +64, −11
  - 01: TP +16, −6
  - 10: FP +47, −45
  - 11: TP +55, −29

**C, in-window interpolation (diagnostic only):**

| Keys | Rows | Root F1 / Brier | Half F1 / Brier | Full F1 / Brier |
|---|---|---|---|---|
| All | 207,539 | .669541 / .047913908 | .674645 / .047476227 | .679823 / .047123033 |
| Matched | 129,522 | .677720 / .051206004 | .681710 / .050804583 | .686491 / .050487753 |

- **Persistence on matched C keys:** F1 .519468.
- **Fold means vs root:** half F1 +.00546, Brier −.000438; full F1 +.01073, Brier −.000791.
- **True-local C rows** (147,305): Brier .062443 / .061827 / .061329.

## Reading (supervisor decision)

**Half-shrinkage remains a diagnostic candidate only; it is not adopted and not the default.**
- On C, gains grow steadily with amplitude, but C is in-window interpolation.
- On E3, transfer is weak and heterogeneous. F1 shows a small positive change and H4/H12 Brier improve slightly, but H8 Brier gets worse as amplitude increases. This is insufficient evidence of a useful fix.
- A flat E3 result cannot establish that the local increments are not over-fitted.
- No further alpha, window or root-threshold tuning. Stage 1 partition overfitting remains unresolved.
