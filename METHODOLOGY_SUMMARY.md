# GeoRF: A Spatially Adaptive Machine Learning Framework for Acute Food Crisis Prediction — Methodology Summary

---

## 1. Overall Methodological Objective

Acute food insecurity prediction across Sub-Saharan Africa presents a fundamental challenge: the drivers and dynamics of food crises are spatially heterogeneous, varying substantially across agroecological zones, conflict corridors, and administrative boundaries. A single global predictive model, trained uniformly across all regions, implicitly assumes spatial stationarity in the relationship between predictors and crisis outcomes — an assumption widely violated in practice.

This framework, termed **GeoRF** (Geospatially Adaptive Random Forest), addresses this challenge through a three-stage methodological architecture that (i) discovers latent spatial structure through recursive binary partitioning of geographic space, (ii) stabilizes these partitions into consensus spatial clusters via weighted spectral clustering, and (iii) evaluates the predictive benefit of spatial adaptation by comparing locally specialized models against their globally pooled counterparts.

The central hypothesis is that **spatial partitioning of the prediction domain** — training region-specific models within data-driven geographic clusters — yields systematically higher crisis-class predictive performance than a single pooled model, because local models can capture regionally specific predictor–outcome relationships that a global model averages away.

The framework is designed to operate at the spatial resolution of FEWSNET administrative units (~5,700 polygons across Africa), at monthly temporal granularity, and across multiple forecasting horizons (4, 8, and 12 months ahead). It supports multiple base learners (Random Forest, XGBoost, Decision Tree) under a unified spatial partitioning architecture, enabling controlled comparison of both spatial strategy and base learner choice.

---

## 2. Pipeline-Level Architecture

The framework is organised as a three-stage computational pipeline, where each stage produces intermediate outputs that feed directly into the next. The stages are conceptually distinct but methodologically coupled: Stage 1 generates an ensemble of candidate spatial partitions under varying temporal conditions; Stage 2 synthesises these into stable, consensus spatial clusters; and Stage 3 evaluates the predictive value of the resulting spatial structure through systematic comparison experiments.

**Stage 1 — Temporal Partition Generation.** For each combination of year, month, and forecasting scope, a full GeoRF model is trained end-to-end, producing a spatial partition that reflects the optimal geographic decomposition for that specific temporal context. This yields an ensemble of ~144 partition "plans" (4 years × 12 months × 3 forecasting scopes), each capturing how the model would divide geographic space under a different slice of the data.

**Stage 2 — Spatial Weighted Consensus Clustering.** The ensemble of partitions from Stage 1 is aggregated into a single stable spatial decomposition. A weighted co-occurrence similarity matrix is constructed across all partition plans, where each plan's contribution is weighted by its predictive improvement over baseline. This similarity matrix is then subjected to KNN sparsification, spectral analysis, and spectral clustering to yield a small number of consensus spatial clusters. Both a general (all-months) partition and season-specific partitions (February, June, October — representing the three FEWSNET projection windows) are produced.

**Stage 3 — Partitioned Model Evaluation.** The consensus spatial clusters from Stage 2 define the geographic domains within which local models are trained. For each test month, the framework trains both a pooled (global) model and a set of partition-specific (local) models, then evaluates both against the same holdout data. This controlled comparison quantifies the marginal benefit of spatial partitioning at the level of individual administrative units, countries, and the full study domain.

The pipeline is designed so that each stage is independently reproducible and its outputs are self-documenting. Intermediate artifacts — partition correspondence tables, similarity matrices, cluster manifests, and per-month metrics — form a complete audit trail from raw data to final comparative evaluation.

---

## 3. Stage-by-Stage Method Narrative

### 3.1 Stage 1: Temporal Partition Generation via Hierarchical Binary Spatial Splitting

#### 3.1.1 Data Preparation and Temporal Structure

The input data consist of monthly panel observations at the FEWSNET administrative unit level, spanning approximately 70,000 temporal records across ~3,600 unique spatial units. Each record includes a binary crisis indicator (IPC Phase ≥ 3) as the target variable, along with a rich set of predictors drawn from satellite remote sensing (vegetation indices, nighttime lights, rainfall, temperature), conflict event databases (battles, explosions, violence against civilians), food price monitoring, and seasonal agricultural indicators.

Predictors are classified into two categories based on their temporal variability. **Level-1 (L1) features** are spatially or temporally stable (e.g., agroecological zone, country identity) and enter the model without temporal adjustment. **Level-2 (L2) features** are time-varying and are dynamically lagged according to the forecasting scope: 4 months for near-term forecasts, 8 months for medium-term, and 12 months for long-range projections. This lag structure enforces a strict information boundary, ensuring that no predictor values from the forecast gap period contaminate the training data.

Feature engineering augments the raw predictors with rolling temporal aggregations. Conflict indicators are summarised as 4-month and 12-month cumulative counts; nighttime light intensity is accumulated over 12-month windows; and environmental variables (EVI, rainfall, temperature, gross primary productivity) are lagged by 12 months to capture delayed ecological responses. Missing values are imputed using a sentinel strategy (replacement with a value far outside the observed range), allowing tree-based learners to naturally isolate missing-data patterns through their splitting logic.

#### 3.1.2 Rolling-Window Temporal Splitting

For each target month, the framework constructs a rolling-window train–test split that mimics a genuine real-time forecasting scenario. The **test set** consists solely of observations from the single target month. The **training set** comprises a fixed-length window (36 months) ending a specified number of months before the test month, with the intervening period excluded as a temporal buffer. For a 4-month forecasting scope targeting January 2023, for instance, training data span September 2019 through August 2022, with September–December 2022 excluded as the forecast gap.

This temporal protocol ensures that (i) no future information leaks into training, (ii) the training window is long enough to capture interannual variability, and (iii) the forecast gap matches the operational lead time of FEWSNET's early warning system.

Within the training set, a stratified train–validation split is constructed with explicit spatial coverage guarantees: each administrative unit that appears in the test set must also be represented in both the training and validation subsets. This group-aware splitting prevents the model from encountering entirely novel spatial units during partition evaluation.

#### 3.1.3 Spatial Group Definition

The minimum spatial unit for partitioning is the FEWSNET administrative polygon. Each polygon is treated as an **indivisible atom**: all temporal records associated with a given polygon are assigned to the same partition branch at every depth of the partition tree. This constraint ensures that the resulting spatial decomposition is geographically interpretable — each region belongs entirely to one partition, with no within-polygon splits.

Polygons are indexed by their centroid coordinates and linked to a precomputed adjacency structure derived from the FEWSNET shapefile. Two polygons are deemed adjacent if and only if they share a boundary segment of nonzero length (point contacts are excluded). This true-boundary adjacency captures the actual geographic neighbourhood structure far more accurately than distance-based heuristics, reducing spurious connections by approximately 86% relative to a centroid-distance threshold approach.

#### 3.1.4 Hierarchical Binary Partitioning Algorithm

The core of Stage 1 is a **recursive binary spatial splitting algorithm** that progressively divides the geographic domain into increasingly homogeneous subregions. The algorithm proceeds through up to six depth levels, producing a binary tree of partitions.

At each depth level, every existing partition branch with sufficient sample size is evaluated as a candidate for further splitting. For each candidate branch:

1. **Group-level performance aggregation.** The validation-set predictions of the current branch model are aggregated at the polygon level. For each polygon and each target class, the algorithm computes (a) the count of classification errors and (b) the total number of samples. These polygon-level error statistics form the input to the splitting optimiser.

2. **Optimal split search via coordinate descent.** A scan-based optimiser seeks the binary partition of polygons that maximises a log-likelihood ratio criterion. The optimiser alternates between (i) computing per-polygon scores based on class-conditional error rates, and (ii) assigning polygons to the partition branch (left or right) that best explains their error pattern. This coordinate descent converges to a locally optimal division of polygons into two groups with maximally different error structures. A flexibility constraint ensures that neither resulting group is disproportionately small relative to the other.

3. **Statistical significance testing.** The proposed split is accepted only if the improvement exceeds configurable thresholds for statistical significance, effect size, and minimum mean difference. Specifically, the class-1 (crisis) F1 improvement must surpass a minimum threshold, ensuring that splits are driven by genuine gains in crisis prediction rather than noise.

4. **Contiguity refinement via polygon-adjacency majority voting.** After each accepted split, the partition assignment of each polygon is reconsidered in light of its geographic neighbours. For each polygon, the algorithm examines the partition assignments of all adjacent polygons (as determined by the precomputed adjacency matrix) and reassigns the polygon to the majority partition of its neighbourhood if the current assignment is in the minority. This majority-voting procedure is repeated for multiple epochs (typically three), progressively smoothing the spatial partition and eliminating isolated enclaves. Safeguards prevent the elimination of very small but genuinely distinct spatial components (e.g., island territories).

5. **Local model training.** Upon acceptance of a split, separate base learners (Random Forest, XGBoost, or Decision Tree) are trained on the training data within each resulting partition. These local models replace the parent model for all polygons assigned to the respective branches.

The depth-by-depth recursion continues until (a) the maximum depth is reached, (b) no statistically significant splits remain, or (c) the remaining branches have insufficient sample sizes. The output of this process is a correspondence table mapping each administrative polygon to its terminal partition branch, together with performance metrics recording the F1 improvement at each split.

#### 3.1.5 Monthly Execution and Ensemble Generation

Stage 1 is executed independently for every combination of target year (2021–2024), target month (January–December), and forecasting scope (4, 8, 12 months). Each execution produces its own partition correspondence table and performance metrics. Across the full temporal range, this yields approximately 144 independent spatial partitions — an ensemble of "views" on how geographic space should be decomposed under varying temporal conditions.

The monthly execution strategy serves two purposes. First, it captures **temporal variability in spatial structure**: the optimal geographic decomposition for February (post-harvest in the Sahel) may differ systematically from that for October (lean season in East Africa). Second, it provides a large sample of partitions for the consensus clustering that follows, improving the statistical stability of the final spatial decomposition.

---

### 3.2 Stage 2: Spatial Weighted Consensus Clustering

The ensemble of ~144 partitions from Stage 1 captures the model's view of spatial heterogeneity under diverse temporal conditions, but individual partitions may be noisy, idiosyncratic, or overfit to the specific training window. Stage 2 synthesises this ensemble into a single stable, consensus spatial decomposition through a four-step process: performance-weighted co-occurrence aggregation, spatial kernel weighting, graph sparsification with spectral analysis, and spectral clustering.

#### 3.2.1 Partition Indexing and Metadata Assembly

All Stage 1 outputs are gathered into a structured experiment directory. Each partition plan is indexed by its year, month, and forecasting scope, and linked to both its correspondence table (polygon → partition assignments) and its performance metrics (class-1 F1 score for the partitioned model and the unpooled baseline). This indexing step creates the metadata foundation for the weighted consensus that follows.

#### 3.2.2 Performance-Weighted Co-Occurrence Similarity

The central idea of the consensus clustering is that **two administrative units should be grouped together if they were consistently co-assigned across many partition plans, especially the well-performing ones**.

For each pair of administrative units (i, j), a co-occurrence similarity is computed as:

$$S_{ij} = \sum_{p \in \mathcal{P}} w_p \cdot \mathbb{1}[\text{partition}_p(i) = \text{partition}_p(j)]$$

where the sum ranges over all partition plans $\mathcal{P}$, the indicator function evaluates whether units i and j were assigned to the same partition in plan p, and $w_p$ is the performance weight of plan p.

The weight $w_p$ is defined as the **logit-transformed F1 improvement** of the partitioned model over the unpooled baseline:

$$w_p = \max\left(0,\; \text{logit}(F1_p^{\text{partitioned}}) - \text{logit}(F1_p^{\text{baseline}})\right)$$

The logit transformation maps F1 scores from the bounded [0, 1] interval to an unbounded scale, amplifying differences in the high-performance regime where marginal gains are most meaningful. Plans where partitioning did not improve upon the baseline receive zero weight and do not influence the consensus.

This weighting scheme ensures that the consensus clustering is shaped primarily by temporal contexts in which spatial partitioning demonstrably mattered — that is, months and scopes where local models substantially outperformed the global model.

#### 3.2.3 Spatial Kernel Weighting

To enforce geographic coherence, the raw co-occurrence similarity is modulated by a spatial proximity kernel:

$$S_{ij}^{\text{final}} = S_{ij} \cdot \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right)$$

where $d_{ij}$ is the great-circle distance between the centroids of units i and j, and $\sigma = 5°$ (approximately 560 km at the equator) controls the spatial bandwidth. This Gaussian kernel downweights co-occurrence evidence between geographically distant units, which may reflect coincidental rather than structurally meaningful co-assignment.

The resulting matrix $S^{\text{final}}$ is normalised to unit maximum and represents a pairwise affinity between all ~5,700 administrative units in the study domain.

#### 3.2.4 KNN Sparsification and Spectral Analysis

The full affinity matrix is dense and computationally expensive to cluster directly. A k-nearest-neighbour (KNN) sparsification step retains only the k = 40 strongest connections for each unit, producing a sparse symmetric graph. This sparsification preserves local neighbourhood structure while dramatically reducing computational complexity.

The sparsified graph is analysed via its **normalised graph Laplacian**:

$$\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$$

where $\mathbf{A}$ is the symmetric adjacency matrix and $\mathbf{D}$ is the diagonal degree matrix. The smallest eigenvalues of $\mathbf{L}$ are computed, and the **eigengap heuristic** — the index of the largest gap between consecutive eigenvalues — determines the optimal number of clusters. This spectral approach discovers the natural community structure of the geographic affinity graph without requiring a priori specification of cluster count.

Connected component analysis of the KNN graph identifies the main connected component and any isolated subgraphs. Units in smaller components are later assigned to the nearest cluster via a nearest-neighbour classifier operating on geographic coordinates.

#### 3.2.5 Spectral Clustering

Using the eigengap-determined cluster count and the KNN affinity matrix as input, **spectral clustering** is applied. The procedure embeds the affinity graph into a low-dimensional eigenspace defined by the leading eigenvectors of the Laplacian, then applies k-means clustering in this spectral embedding. The resulting cluster labels partition the full set of administrative units into a small number of spatially coherent groups (typically 3–10 clusters, depending on the model and temporal subset).

#### 3.2.6 General and Season-Specific Partitions

The consensus clustering is performed in two modes. The **general partition** aggregates all ~144 partition plans regardless of month, capturing the dominant spatial structure across all seasons. **Season-specific partitions** restrict the aggregation to plans from February, June, or October — months corresponding to FEWSNET's three annual projection windows — capturing seasonal variations in spatial heterogeneity. The resulting partition set includes one general and up to three season-specific spatial decompositions per model type.

---

### 3.3 Stage 3: Partitioned Model Evaluation and Comparison

Stage 3 quantifies the predictive value of the consensus spatial clusters by training and evaluating models under two experimental conditions — pooled and partitioned — across the full 48-month evaluation period.

#### 3.3.1 Experimental Design

For each test month (January 2021 through December 2024), two models are trained on identical training data using identical temporal splits:

- **Pooled model**: A single base learner (Random Forest, XGBoost, or Decision Tree) trained on all available training data regardless of spatial cluster membership. This serves as the null hypothesis — the best achievable performance without spatial specialisation.

- **Partitioned model**: A set of cluster-specific base learners, each trained only on the training data from its respective spatial cluster. Clusters with fewer than 50 training samples or with only a single class represented fall back to the pooled model predictions, ensuring that partitioning never degrades performance through data starvation.

Both models are evaluated against the same test-month holdout data, using identical features, temporal lags, and preprocessing. The only methodological difference is the spatial scope of training data.

#### 3.3.2 Class Imbalance Handling

Within each partition, class imbalance between crisis (IPC ≥ 3) and non-crisis observations is addressed via SMOTE (Synthetic Minority Oversampling Technique) with adaptive neighbourhood size. SMOTE is applied independently within each partition's training data, ensuring that synthetic samples reflect partition-specific feature distributions rather than global averages.

#### 3.3.3 Month-Specific Partition Selection

When season-specific partitions are enabled, the framework selects the appropriate partition based on the test month: the February partition for January–April, the June partition for May–August, and the October partition for September–December. The general partition serves as the default when month-specific partitions are unavailable. This selection logic allows the framework to exploit seasonal variation in spatial structure where it exists.

#### 3.3.4 Optional Post-Hoc Contiguity Refinement

Prior to model training, the consensus cluster assignments may undergo an additional round of adjacency-based majority voting refinement, identical in logic to the within-Stage-1 contiguity refinement but applied to the consensus clusters. This step smooths any remaining spatial fragmentation inherited from the spectral clustering, further reducing isolated polygon assignments.

#### 3.3.5 Evaluation Metrics and Aggregation

Performance is evaluated exclusively through **class-1 (crisis) metrics**: precision, recall, and F1 score. This focus reflects the asymmetric cost structure of food security early warning, where failing to predict a crisis (false negative) carries far greater humanitarian cost than a false alarm (false positive). Overall accuracy is recorded but not used as the primary evaluation criterion, as it is dominated by the majority non-crisis class.

Metrics are aggregated at three spatial scales:

- **Full-domain**: Monthly precision, recall, and F1 across all test observations, enabling temporal trend analysis.
- **Country-level**: Metrics aggregated by national boundaries, revealing which countries benefit most from spatial partitioning.
- **Polygon-level**: Per-administrative-unit performance, enabling high-resolution mapping of where partitioned models improve upon or underperform the pooled baseline. The **F1 improvement** ($\Delta F1 = F1^{\text{partitioned}} - F1^{\text{pooled}}$) at each polygon provides a spatially explicit measure of the value of geographic specialisation.

---

## 4. How the Stages Interact as a Unified Framework

The three stages form a tightly coupled methodological system in which each stage both depends on and informs the interpretation of the others.

**Stage 1 provides the empirical basis for spatial structure discovery.** By running the partitioning algorithm across 144 distinct temporal contexts, Stage 1 generates a diverse ensemble of spatial decompositions that collectively reveal which geographic groupings are robust across time and which are artefacts of specific temporal conditions. No single partition from Stage 1 is used directly for final evaluation; rather, the entire ensemble serves as input to Stage 2.

**Stage 2 distils temporal variability into stable spatial structure.** The weighted consensus approach functions as a meta-learning step: it learns which spatial groupings are consistently supported by evidence across multiple temporal windows, with greater weight given to windows where spatial partitioning demonstrably improved prediction. The spectral clustering then projects this high-dimensional co-occurrence information into a low-dimensional partition with interpretable geographic clusters. The eigengap heuristic provides a principled, data-driven determination of cluster count, avoiding arbitrary specification.

**Stage 3 closes the evaluation loop.** By comparing partitioned against pooled models using the Stage 2 consensus clusters, Stage 3 provides an independent, out-of-ensemble evaluation of the spatial structure's predictive value. Critically, Stage 3 uses the raw data and full model pipeline — not the Stage 1 partitioning algorithm — to train its local models. The spatial clusters from Stage 2 serve only as geographic domain definitions. This separation prevents circularity: the partitions are discovered in Stage 1–2 and evaluated in Stage 3 under conditions that do not replicate the discovery process.

The three stages thus implement a **discover–stabilise–evaluate** paradigm:
1. **Discover** latent spatial structure through repeated model-driven partitioning;
2. **Stabilise** this structure through weighted consensus and spectral clustering;
3. **Evaluate** the stabilised structure's predictive utility through controlled comparison.

---

## 5. Key Methodological Design Principles

### 5.1 Polygon Indivisibility and Geographic Interpretability

The framework treats administrative polygons as atomic spatial units throughout all stages. No polygon is ever split across partition branches. This constraint ensures that the resulting spatial decomposition aligns with the administrative boundaries used by humanitarian actors, making the results directly actionable for food security decision-making.

### 5.2 Crisis-Focused Optimisation

All splitting, weighting, and evaluation decisions are oriented toward class-1 (crisis) F1 performance. This asymmetric focus is maintained consistently across Stage 1 (where splits are accepted only if they improve crisis-class F1), Stage 2 (where partition plans are weighted by their crisis-class F1 improvement), and Stage 3 (where the primary comparison metric is crisis-class F1). The framework does not optimise for overall accuracy, which would be dominated by the majority non-crisis class.

### 5.3 Temporal Integrity and Information Boundaries

The rolling-window temporal protocol enforces a strict separation between training and test periods, with an explicit forecast gap matching the operational lead time. No predictor values from the gap period enter training, and temporal lags on time-varying features are adjusted to the forecasting scope. This design ensures that reported performance reflects genuine forecast skill rather than in-sample fit.

### 5.4 Spatial Contiguity as a Regulariser

The majority-voting contiguity refinement, applied after each binary split in Stage 1 and optionally after consensus clustering in Stage 2, serves as a spatial regulariser. It penalises partition assignments that are geographically isolated, smoothing the partition surface and reducing spatial noise. The use of true polygon adjacency (from boundary geometry) rather than distance-based heuristics ensures that this regularisation respects actual geographic connectivity.

### 5.5 Performance-Weighted Consensus

Not all temporal contexts are equally informative about spatial structure. The logit-weighted consensus in Stage 2 upweights partition plans where spatial partitioning produced clear predictive gains, and downweights or excludes plans where it did not. This selective weighting improves the signal-to-noise ratio of the consensus and ensures that the final spatial clusters reflect structural heterogeneity rather than overfitting.

### 5.6 Graceful Degradation

When a spatial cluster has insufficient training data or lacks class diversity, the partitioned model falls back to the pooled model for that cluster. This ensures that spatial partitioning can only help — never hurt — predictive performance relative to the pooled baseline, providing a principled lower bound on the partitioned model's performance.

### 5.7 Base-Learner Agnosticism

The spatial partitioning architecture is decoupled from the choice of base learner. Random Forest, XGBoost, and Decision Tree models share identical spatial partitioning logic, enabling controlled comparison of base learners under the same spatial decomposition. This modularity facilitates ablation studies and ensures that observed performance differences reflect the base learner's capacity rather than differences in spatial strategy.

---

## 6. High-Level Pseudo-Code Abstraction

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1: TEMPORAL PARTITION GENERATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOR each year in [2021 ... 2024]:
  FOR each forecasting_scope in [4-month, 8-month, 12-month]:
    FOR each month in [January ... December]:

      data ← load_panel_data()
      train, test ← rolling_window_split(data, month, scope)
      features ← engineer_features(train, scope)

      // Hierarchical binary spatial partitioning
      partitions ← {root: all_polygons}
      FOR depth = 1 TO max_depth:
        FOR each branch in partitions:
          model ← train_base_learner(branch.train_data)
          errors ← evaluate_on_validation(model, branch.val_data)
          polygon_stats ← aggregate_errors_by_polygon(errors)

          split ← optimise_binary_partition(polygon_stats)
          IF split is statistically significant:
            split ← refine_contiguity(split, adjacency_matrix)
            partitions ← partitions ∪ {split.left, split.right}

      SAVE correspondence_table(polygon → partition)
      SAVE performance_metrics(F1_partitioned, F1_baseline)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2: SPATIAL WEIGHTED CONSENSUS CLUSTERING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

plans ← COLLECT all Stage 1 partitions with metrics

// Weighted co-occurrence similarity
S ← zero_matrix(n_polygons × n_polygons)
FOR each plan p in plans:
  w_p ← max(0, logit(F1_partitioned_p) − logit(F1_baseline_p))
  FOR each pair (i, j) co-assigned in plan p:
    S[i,j] ← S[i,j] + w_p

// Spatial kernel modulation
K ← gaussian_kernel(pairwise_distances, σ = 5°)
S_final ← S ⊙ K        // element-wise product

// Spectral clustering
G ← knn_sparsify(S_final, k = 40)
L ← normalised_laplacian(G)
n_clusters ← eigengap_heuristic(eigenvalues(L))
clusters ← spectral_clustering(G, n_clusters)

SAVE cluster_assignments(polygon → cluster_id)
  // general partition (all months)
  // season-specific partitions (Feb, Jun, Oct subsets)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 3: PARTITIONED MODEL EVALUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

clusters ← LOAD Stage 2 consensus spatial clusters

FOR each test_month in [2021-01 ... 2024-12]:
  FOR each forecasting_scope:
    train, test ← rolling_window_split(data, test_month, scope)

    // Pooled baseline
    model_pooled ← train_base_learner(train)
    y_pred_pooled ← model_pooled.predict(test)

    // Partitioned model
    FOR each cluster c in clusters:
      train_c ← train[cluster_id == c]
      IF |train_c| ≥ 50 AND both classes present:
        train_c ← oversample_minority(train_c)      // SMOTE
        model_c ← train_base_learner(train_c)
      ELSE:
        model_c ← model_pooled                      // fallback

    y_pred_partitioned ← assemble_predictions(models, test)

    RECORD metrics(y_pred_pooled, y_pred_partitioned, y_true)
      // crisis-class F1, precision, recall
      // per-polygon, per-country, full-domain

COMPUTE ΔF1 = F1_partitioned − F1_pooled
  // by polygon, country, month, and overall

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 7. Summary

The GeoRF framework implements a principled approach to geographic model specialisation for food crisis prediction. Rather than imposing spatial structure *a priori* (e.g., by country or agroecological zone), the framework **learns** spatial structure from the data through a discover–stabilise–evaluate pipeline. Stage 1 generates a temporally diverse ensemble of model-driven spatial partitions. Stage 2 fuses this ensemble into stable consensus clusters through performance-weighted spectral clustering, letting the data determine both the cluster boundaries and their optimal number. Stage 3 rigorously evaluates the resulting spatial decomposition through controlled comparison of partitioned and pooled models, providing spatially explicit evidence for where and by how much geographic specialisation improves crisis prediction.

The framework's emphasis on crisis-class optimisation, temporal integrity, polygon-level geographic interpretability, and base-learner agnosticism positions it as a methodologically rigorous tool for operational food security early warning — one that adapts not only to local predictor–outcome relationships but also to seasonal variations in the spatial structure of food crisis risk.
