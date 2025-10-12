---
**📋 MIGRATION NOTICE**

This document has been migrated to the new structured documentation system:
- **Current Usage**: See `CLAUDE.md` in root directory for user guide
- **Technical Design**: See `.ai/spatial-partitioning-optimization/design/architecture.md` for detailed design
- **Development Standards**: See `.ai/docs/` for foundation documents

For migration details, see `_migration_index.csv` and `_link_graph.md` in this directory.
---

  Partition Algorithm Details
  ================================
  
  
  - We compute c = y_true_value - true_pred_value, the observed errors per group/class.
  - We build b from get_c_b: it distributes the total errors (c_tot) across groups in proportion to their sample mass (base/base_tot). Intuitively b is what the error layout would look like if the parent branch were perfectly homogeneous.

  With c and b in place the loop maintains two ingredients:

  1. q – the relative risk (observed error / expected error) for each class inside the candidate hot-spot. It’s initialized by picking the top outliers per class and then re-estimated after every subset update (q = Σ c[s0] / Σ b[s0]). In spatial-scan language, q is the MLE of the “risk multiplier” inside the flagged subset.
  2. gscore (LTSS contribution) – for every group we compute
     g_i = Σ_class [ c_i * log(q_class) + b_i * (1 - q_class) ].
     That is the group’s contribution to the log-likelihood ratio comparing “this group lives in a shifted error regime (q ≠ 1)” vs the homogeneous null (q = 1). It’s exactly the form required by Linear-Time Subset Scanning (LTSS): once you calculate these per-item scores, the highest-scoring subset is the set of all records whose gscore exceeds some threshold.

  get_top_cells then sorts the g vector in descending order and chooses the top segment as s0 (the “needs a new branch” side) and the remainder as s1. Without flexing, it defaults to the top half, but FLEX_RATIO and flex_type tweak the subset size bounds so we don’t end up with a wildly unbalanced split. If flex_type uses sample counts ('n_sample'), the cumulative counts (cnt) derived
  from b_cnt are what enforce the allowable size range.
  So the sequence is:

  1. Aggregate validation stats per X_group.
  2. Build q=c/b.
  3. Iterate: compute g from current q, take the top-ranked groups (LTSS), recompute q from that subset, repeat.
  4. After convergence, s0/s1 define the binary partition that maximizes the log-likelihood ratio under the size constraints set by FLEX_RATIO.

  where c_i is the observed error mass for group i and b_i is what the error would look like there if the branch were perfectly homogeneous. A positive g_i means “this group is performing worse than the parent baseline under the current q.” get_top_cells then sorts those g_i values and, subject to the balance constraint from FLEX_RATIO / flex_type, collects the top segment into s0 (the
  candidate child branch) and the remainder into s1. That’s the LTSS step: once you have additive scores like g_i, the optimal subset with the highest total log-likelihood ratio is simply the set of items whose g_i exceed a threshold, so sorting and taking the top block gives you the best branch.

  Step 4 (parameter update)
  With that subset in hand, the algorithm recomputes the relative-risk parameters from the selected groups:

  q_class = Σ_{i∈s0} c_{i,class} / Σ_{i∈s0} b_{i,class}

  Those ratios are the MLE of “how much worse than baseline” the child branch behaves for each class. Once q is updated, the code loops: rebuild g using the new q, re-select s0, update q, and so on. In practice the subset stabilizes after one or two passes (because you’ve already picked the groups that inflate the log-likelihood the most), so the second iteration usually reproduces the
  same s0 and q, at which point you’re done. The comments about convergence thresholds are right—the code doesn’t currently stop on a tolerance, but the fixed-point behaviour means the loop naturally settles into a steady state.

Rank groups by their g_i under the current q, cut them according to the flex rules, recompute q from the chosen subset, and repeat until the ranking no longer changes.
  
  Contiguity smoothing and the later significance test happen on top of that, but the core split decision is exactly the LTSS ranking you described.