# Research: GeoDT Branch-Specific Local Tree Interpretability Figure

## Decision: Implement as a standalone exploratory script under `scripts/`

**Rationale**: Existing repository guidance uses `scripts/` for analysis and plotting utilities. A standalone script minimizes brownfield change surface and avoids modifying GeoDT training, prediction dispatch, branch training, or `dt_rules` export behavior.

**Alternatives considered**:
- Modify `app/main_model_DT.py`: rejected because it risks changing existing Stage 1 behavior and root/global `dt_rules` semantics.
- Add reusable package first under `src/vis/`: deferred until implementation shows shared renderer needs.

## Decision: Use matplotlib/scikit-learn plotting for MVP; Graphviz not required

**Rationale**: Repository evidence already supports matplotlib-based plots and sklearn `DecisionTreeClassifier` tree inspection. Graphviz would add an avoidable dependency and operational risk.

**Alternatives considered**:
- Graphviz export: deferred unless matplotlib/sklearn cannot meet readability requirements.
- Text/table rule panels: possible fallback, but current spec asks for shallow tree panels.

## Decision: Preflight-first implementation

**Rationale**: Evidence shows monthly visual archives may not include `checkpoints/` or `space_partitions/`. Running archive characterization before candidate selection prevents false assumptions and avoids misleading figures.

**Alternatives considered**:
- Implement plotting first: rejected because branch artifacts may be unavailable.
- Use `dt_rules_*.csv` as fallback: rejected because those exports are root/global-only under the brownfield evidence.

## Decision: Same-run/source artifact-provider folder may complete a visual archive

**Rationale**: The clarified spec allows selected visual archives to reference matching source run folders when visual archives lack branch artifacts. This reduces blockage from archive packaging while preserving provenance through compatibility evidence.

**Alternatives considered**:
- Require all artifacts inside visual archive: rejected by clarification because packaging gaps should not block a valid same-run diagnostic.

## Decision: Default branch-pair score is top-K split-feature Jaccard distance

**Rationale**: Deterministic feature-set Jaccard distance is simple, auditable, and robust to numeric threshold scale differences. Threshold/direction fields remain supplemental metadata and do not alter default selection.

**Alternatives considered**:
- Include threshold/direction penalty in ranking: rejected for MVP because it adds subjective weighting.
- Manual branch selection: rejected because the figure must not appear cherry-picked.

## Decision: Treat feature-name mismatch as checkpoint unusability

**Rationale**: Publication/report readability requires reliable feature labels. Falling back to generic `feature_<idx>` labels could mislead interpretation.

**Alternatives considered**:
- Plot generic feature names: rejected by spec.
- Guess feature columns from unrelated runs: rejected unless compatibility is explicitly recorded.

## Decision: Metadata JSON is canonical provenance output

**Rationale**: Reproduction mode and auditability require machine-readable provenance. Human-readable audit markdown can be generated alongside JSON, but JSON should be the authoritative record.

**Alternatives considered**:
- Markdown-only summary: rejected because reproduction checks need structured fields.
- CSV-only metadata: insufficient for nested fallback/candidate/checkpoint details.

## Decision: Use synthetic GeoDT-like fixtures before relying on real archives

**Rationale**: Preferred archive availability is unknown. Synthetic fixtures allow deterministic tests for archive discovery, artifact completeness, scoring, readability, failure handling, and reproduction.

**Alternatives considered**:
- Require real archive for all tests: rejected because local artifacts may be absent or incomplete.
