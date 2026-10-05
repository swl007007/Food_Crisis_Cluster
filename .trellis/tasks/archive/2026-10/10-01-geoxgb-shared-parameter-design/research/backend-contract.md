# Backend contract facts for the pending development budget

2026-10-01, planning only. Mother package `P=FEWSNETFourClassBaseline/`, inspected at `14c89bc150194452361bb495c601de070cd94ce7`. No model was fitted. These are research findings, not approval of a new weighting scheme or runtime.

## Current RF weights and class handling

- Active Stage 1 has no class/sample reweighting: `P/app/main_model_GF.py:124-128`, `P/src/model/GeoRF.py:50-54`, `P/src/model/model_RF.py:325-328,365-381`; branch training also passes no weights (`P/src/model/train_branch.py:41,48`). RF uses 100 estimators and unlimited individual-tree depth by default; partition depth is a separate setting.
- Stage 1 appends four nonzero-weight artificial training rows after fitting the real-row imputer: `P/src/model/model_RF.py:154-159,321-323,432-433`. Each has all-zero transformed predictors and one of the four labels. They are not real support and do not enter imputer statistics. The commented inverse-frequency helpers are not active behavior.
- Stage 3 uses `n_estimators=100`, `max_depth=None`, `class_weight=None`, real-row fitting, and no pseudo rows: `P/scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:53,75-82,117-134,148-173,200`.
- Fixed-four scoring and RF probability alignment live at `P/src/metrics/fourclass.py:14-17,123-136`. RF missing classes are zero-filled in the aligned probability matrix; native multiclass XGB need not have that behavior.

## Native XGBoost versus sklearn wrapper

The preferred Windows Python imported XGBoost **3.0.0** during read-only research. This verifies an installed version, not a model-compatibility test or binary-build identity.

Official sources below are pinned to [XGBoost v3.0.0](https://github.com/dmlc/xgboost/tree/v3.0.0):

- `src/objective/multiclass_obj.cu:49-56,91-111,125-138,190-192`: native `multi:softprob` with `num_class=4` checks labels lie in `[0,4)` and computes all four class outputs. It does not require all four labels to appear in a local fitting subset. This supports testing native `xgb.train` without artificial class-recovery rows.
- Installed `xgboost/sklearn.py:1633-1643,1654-1658` infers `n_classes_` from observed labels, checks contiguous encoding, and can overwrite `num_class`. Do not transfer the native API guarantee to `XGBClassifier.fit` without verification.
- Installed `xgboost/training.py:113-114,166,180-185` and `core.py:1849-1875`: continuation constructs a Booster from the supplied model then applies params and additional iterations. A supplied Booster is loaded from a memory snapshot into a new instance.
- Official `src/learner.cc:405-435`, `src/gbm/gbtree.h:207-208`: `boost_from_average` estimates the base score only when the model is not fitted. Local continuation must not explicitly override `base_score`; preserve and verify it with the tree prefix.
- `src/gbm/gbtree.cc:278-305`, `src/gbm/gbtree_model.h:129-134`: `gbtree`, `process_type=default` creates and appends trees; `update` reuses existing trees. DART has normalization behavior (`gbtree.cc:901-905`), so the frozen-prefix design should use standard `gbtree`, not infer DART compatibility.
- `src/gbm/gbtree.cc:249-256` and `gbtree_model.h:65-67`: fixed four-class, one-output-per-tree, `num_parallel_tree=1` gives four physical trees per boosting round.

Implementation must still verify prefix structure/leaf values/missing directions/base score, parent immutability, missing-class continuation, added rounds, and saved prediction replay in the pinned environment. Source inspection alone does not pass those checks.

## Weighting is an explicit design choice

No new class weighting is approved. For a fitting pool of N rows, K observed classes and positive class counts n_c, weights `N/(K*n_c)` have row mean 1; `1/n_c` has mean K/N; `N/(4*n_c)` has mean K/4 when classes are absent. XGBoost v3.0.0 multiclass gradients/Hessians multiply by supplied row weights (`src/objective/multiclass_obj.cu:96-103`) without implicit mean normalization. Scaling all weights can change the effects of fixed Hessian/regularization thresholds (`src/tree/param.h:245-265`). Do not introduce per-region inverse-frequency weights as an unnoticed implementation default.

## Consensus scope

`P/scripts/run_stage2.py:68-79` consumes the complete candidate ledger and explicitly records `general, fs1-fs3 together`. D5 preserves one general consensus map formed from all H candidates. Per-H model configurations do not imply independent per-H maps. Candidate-family expansion must preserve identities and the common information cutoff; missing candidate evidence is distinct from an all-zero-weight completed pool (`:43-51,80-85,101-103`).
