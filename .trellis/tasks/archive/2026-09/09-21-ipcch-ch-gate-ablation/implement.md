# Execution plan

No `design.md`: this task adds no architecture. It reuses `IPCCHGeoRFExperiment`
unchanged apart from two declared knobs — a cohort filter and a gate override — plus a
reporting addition. The design that matters is already frozen in that package and in
the prior run `ipcch-v1-20260920d`.

## 1. Two mechanisms that must be got right

### 1.1 The gate override binds in three namespaces, not one

`MIN_CLASS_1_IMPROVEMENT_THRESHOLD` is consumed at
`GeoRFBaseline/src/partition/transformation.py:720` and
`GeoRFBaseline/src/tests/sig_test.py:30`. Both modules do `from config import *`
(`transformation.py:13`, `sig_test.py:12`), and `transformation.py:17` additionally does
`from src.tests.sig_test import *`. The value is therefore copied into each module's
namespace at import time.

Setting it on `config` after those imports changes nothing that the partition learner
reads, while `REPORTED_CONFIG_KEYS` — which reads back from `config` — would faithfully
report `0.005`. That combination produces a run that gates at 0.01 and claims 0.005.
This is the same trap as the FEWS NET `feature_drop` lowercase alias.

- [ ] Set the value on `config`, `src.tests.sig_test` and `src.partition.transformation`
      inside the run's isolated import scope.
- [ ] Read it back from all three and record all three in the cell manifest. Fail the
      cell on any disagreement with the declared gate (R3/A2).
- [ ] Confirm empirically on one cell that the gate actually moved: C2 must evaluate the
      same candidate splits as C1 and accept a superset of them. If C1 and C2 accept
      identical splits, either the override failed or no candidate sits in
      [0.005, 0.01) — distinguish these by the recorded gate-evaluation counts (R6)
      before proceeding.

### 1.2 The cohort filter goes after the source gate, not before

`check_target_gate` compares the built ledger against hard-coded audited counts
(`EXPECTED_VALID = 42695`, `EXPECTED_POSITIVE = 15206`, `EXPECTED_NEGATIVE = 27489`,
`EXPECTED_AREAS = 6227`) and fails the run on mismatch. A cohort-restricted ledger will
mismatch all four by construction.

- [ ] Build the full ledger and pass the unchanged source gate first, so every cell
      still proves it read the same pinned source.
- [ ] Apply `admin_code >= 100000` as a declared restriction *after* that gate, and
      record areas/rows/class-1 counts kept and dropped (R4/A3).
- [ ] Reconcile those counts against the source independently, including the CAR split
      (72 of 298 areas are CH).

## 2. Run the six cells

- [ ] C1 `all` / 0.01 first, and verify it reproduces `ipcch-v1-20260920d`'s headline
      metrics before running anything else. A failed reproduction invalidates every
      comparison that follows, so stop and investigate rather than continuing (A4).
- [ ] Then C2 `all`/0.005, C3 `non-CH`/0.01, C4 `non-CH`/0.005, C5 `CH`/0.01,
      C6 `CH`/0.005. Fresh run directory each; never overwrite.
- [ ] After each cell, diff its `REPORTED_CONFIG_KEYS` against C1's and fail on any
      difference other than the gate (R2).
- [ ] Expect roughly 19 minutes per full-cohort cell and less for the restricted ones,
      from the prior run's 14:19:46 -> 14:38:31.

## 3. Comparability reference

- [ ] Recompute the prior baseline's metrics on the non-CH subset of its **existing
      stored per-row predictions**. Refit nothing. Report it beside C3/C4 (R5/A5).
- [ ] State explicitly what this separates: a model that got better versus a cohort that
      got easier.

## 4. Report

- [ ] One table, six cells x four arms x four horizons: class-1 precision/recall/F1 from
      pooled confusion counts, plus terminal partitions learned and gate-evaluation
      counts (R6).
- [ ] Persistence per cell on the same keys as that cell's RF arms (R7).
- [ ] Name the factor that moved the result and by how much — or state that neither did,
      if that is what the numbers say (A6). A null here is a usable answer: it would
      mean the IPCCH line's problem is neither the gate nor CH heterogeneity.
- [ ] Carry the CH base-rate asymmetry and the 2026 CH anomaly (crisis rate 0.9708 on
      137 rows, against 0.13-0.19 in every prior year) into the limitations, whichever
      way the result goes.

## Validation

    python -m unittest discover -s IPCCHGeoRFExperiment -v
    # plus the per-cell gates above; console output is not evidence

Reuse the pinned Windows Python 3.12.10. Run outputs stay out of git except the per-cell
manifests and the results table, following the convention settled in
`FEWSNETCleanPersistenceExperiment/.gitignore`: the completion audit for that task
reported evidence gaps precisely because the ledgers were on disk but not in the commit.

Supersede by moving to `superseded/<utc>/`, never by deleting.
