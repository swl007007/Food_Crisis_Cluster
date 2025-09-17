# GeoRF Architecture (Standard Markdown)

## 1) Overview

This document summarizes the end-to-end data pipeline, model training flow, partitioning algorithm, validation-coverage caveats, and inference routing for **GeoRF** with minimal edits to the original wording. File paths and key line references are retained for traceability.

---

## 2) Data Pipeline

* **Load & Prepare**

  * The pipeline loads the FEWSNET CSV, cleans columns, builds temporal lags, and encodes categorical context **before any model step**; this yields a pandas frame used by both feature prep and grouping (`src/preprocess/preprocess.py:123`, `src/preprocess/preprocess.py:197`).

* **Spatial Grouping**

  * Spatial groups are assigned at the admin-unit level so every temporal record inherits its `FEWSNET_admin_code` as `X_group`, along with lat/lon centroids and optional adjacency metadata (`src/preprocess/preprocess.py:216`, `src/preprocess/preprocess.py:328`).

* **Feature Assembly**

  * Feature assembly converts the frame into `X/y`, records quarter labels, retains the group IDs, and writes correspondence tables for later diagnostics (`src/feature/feature.py:22`, `src/feature/feature.py:96`, `src/feature/feature.py:183`).

* **Temporal Slicing**

  * Temporal train/test windows are carved out **before fitting** (rolling quarterly splits or full-year holdouts), so the GeoRF trainer sees arrays already filtered to the chosen window (`src/customize/customize.py:328`).

---

## 3) Training Entry Point

* **Fit Call**

  * Calling `GeoRF.fit(X, y, X_group, …)` immediately random-splits the provided training window into an internal **train (set id 0)** and **validation (set id 1)** pool, initializes empty branch tags for every sample, and trains a **root random forest** using only the training fold to stabilize the baseline (`src/model/GeoRF.py:191`, `src/model/GeoRF.py:211`, `src/partition/transformation.py:127`).
  * This root model is saved and, if enabled, **pre-partition CV diagnostics** run without touching held-out validation points to confirm the starting accuracy map before any branching (`src/model/GeoRF.py:209`, `src/partition/transformation.py:137`).

---

## 4) Partitioning Algorithm (Main Loop)

* Partitioning happens inside the main loop of `partition(...)`: for each **depth level** and **active branch**, the code:

  1. Gathers that branch’s train/val records,
  2. Scores the current branch model on the validation slice, and
  3. Aggregates performance **by spatial group** via `get_class_wise_stat`, ensuring every `X_group` is treated as an indivisible unit (`src/partition/transformation.py:205`, `src/partition/transformation.py:239`, `src/partition/transformation.py:252`).

---

## 5) Group-Level Aggregation (Why & How)

* **Concretely** in `partition(...)`:

  * The branch’s validation subset is evaluated once (`y_pred_before = base_eval_using_merged_branch_data(...)`) and then passed into
    `get_class_wise_stat(y_val, y_pred_before, X_group[val_list])` (`src/partition/transformation.py:252`).
  * Inside that helper, the targets/predictions are **one-hot encoded** and then **summed by group** via `groupby_sum`, yielding two matrices:

    * `y_val_value` (how many validation samples of each class a group has) and
    * `true_pred_value` (how many were predicted correctly)
      (`src/partition/partition_opt.py:142`). One row in those matrices corresponds to one `FEWSNET_admin_code`.

* **Why is that necessary?**
  The scan step decides which groups go to branch-0 vs branch-1. It needs **group-level** statistics so that a whole admin unit is moved together and the gain/loss for that unit is evaluated holistically. If we kept the raw sample list, the optimizer could split the same admin between children or overweight units with more time steps. Aggregating first enforces the spatial granularity (admin-level partitions) while still using all underlying validation samples to compute the group scores.

* **Scope of Validation**
  So yes: each branch uses a **single validation set**, but that set covers many groups. `get_class_wise_stat` simply **summarizes the per-group performance** within that set **before** the partition optimizer (scan) searches for a better split.

---

## 6) **CRITICAL WARNING** — Validation Coverage by Group

* **No hard guarantee per group**

  * Right now **nothing in the code enforces “at least one validation sample per group.”** When you call `GeoRF.fit` **without** supplying `X_set`, it falls back to `train_val_split`, which just tags individual rows as train (0) or val (1) by drawing from a uniform random distribution (`src/model/GeoRF.py:196`, `src/initialization/initialization.py:162`).
  * Because admins have many quarterly records, most groups end up with some validation rows, but there’s **no hard guarantee**—`partition(...)` even prints a warning if a branch’s validation slice comes back empty (`src/partition/transformation.py:230`).

* **What aggregation does/doesn’t do**

  * The aggregation step only compacts whatever validation rows exist: `get_class_wise_stat` groups the current branch’s validation records by `X_group` to produce one row per admin unit before the scan (`src/partition/transformation.py:252`, `src/partition/partition_opt.py:142`).
  * If a group had **no** validation records, it simply **doesn’t appear** in that summary; the partition optimizer then has **no evidence** for that admin and **defaults it into the “keep with parent”** side when the split is applied (`src/helper/helper.py:205`–`src/helper/helper.py:210`).

* **How to enforce coverage**

  * If you need every group to carry validation coverage, pass an explicit mask through the optional `X_set` parameter when calling `fit`.
    *Example:* build `X_set` so each admin’s most recent quarter is marked `1` (validation) and earlier quarters `0` (training); `GeoRF.fit` will use your mask verbatim and skip the random assignment.
  * Otherwise, the only safeguards are global—`MIN_BRANCH_SAMPLE_SIZE` checks and the early-out for empty validation slices—so the default workflow relies on the dataset’s density rather than an explicit per-group rule.

---

## 7) Scan Optimizer & Contiguity (Choosing Children)

* The scan optimizer searches for two candidate sets of groups `(s0, s1)` that maximize improvement, **optionally refining** them with polygon or grid **contiguity constraints** so neighboring units stay together (`src/partition/transformation.py:274`, `src/partition/transformation.py:283`).
* Once candidate groups are chosen:

  * `get_branch_data_by_group` splits the original arrays so the entire admin code moves into **one child**, and
  * `train_and_eval_two_branch` trains **provisional child forests** using the **parent branch weights as a warm start** (`src/partition/transformation.py:519`, `src/model/train_branch.py:26`).

---

## 8) Split Testing & Acceptance

* The statistical test for a split happens immediately after those provisional models generate validation predictions:

  * The parent branch’s baseline scores and both child scores are compared, and **sample-size guards** ensure each child has enough data before testing (`src/partition/transformation.py:560`, `src/partition/transformation.py:582`).
  * `sig_test` (or the **crisis-focused variant** when configured) computes the **improvement, standard error, effect size**, and compares them to critical values to decide whether the gain is significant enough to keep the new partitions; **in RF mode** it also checks whether either child underperforms the unsplit parent and can **copy back the parent weights** if needed (`src/partition/transformation.py:587`, `src/partition/transformation.py:626`, `src/tests/sig_test.py:17`).
  * **Acceptance rule:** Only when `sig_test` returns `1` (or the branch is still within the **forced minimum depth**) does the tree **accept the partition**.

---

## 9) State Update & Inference Routing

* When a split is accepted, the code:

  * Rewrites every affected sample’s `X_branch_id`,
  * Updates the **branch table** and the `s_branch` dictionary that map each `X_group` to its assigned branch, and
  * Persists those artifacts for downstream prediction (`src/partition/transformation.py:655`, `src/partition/transformation.py:682`).

* **Prediction**

  * Later, `GeoRF.predict` looks up each test sample’s group in `s_branch` to route it to the correct **local model**, ensuring inference **honours the learned partitions** (`src/model/GeoRF.py:320`, `src/model/GeoRF.py:340`).

* **Tuning**

  * If you need to tweak when a split is accepted, inspect the thresholds in `config.py` that `sig_test` references (e.g., `MIN_BRANCH_SAMPLE_SIZE`, `ES_THRD`, `CLASS_1_SIGNIFICANCE_TESTING`) and rerun to see how the branch formation changes.

---

## 10) Practical Next Step

* If you suspect the significance gate is too strict or lenient, consider **logging** the `mean_diff` and `test_stat` values from `sig_test` on your run and **adjust the thresholds accordingly** before another training pass.

---

## Appendix: Key Files (for quick navigation)

* **Preprocess / Feature**

  * `src/preprocess/preprocess.py:123, :197, :216, :328`
  * `src/feature/feature.py:22, :96, :183`
  * `src/customize/customize.py:328`

* **Model / Partition**

  * `src/model/GeoRF.py:191, :209, :211, :320, :340`
  * `src/partition/transformation.py:127, :137, :205, :230, :239, :252, :274, :283, :519, :560, :582, :587, :626, :655, :682`
  * `src/partition/partition_opt.py:142`
  * `src/model/train_branch.py:26`

* **Initialization / Helpers / Tests**

  * `src/initialization/initialization.py:162`
  * `src/helper/helper.py:205–210`
  * `src/tests/sig_test.py:17`
