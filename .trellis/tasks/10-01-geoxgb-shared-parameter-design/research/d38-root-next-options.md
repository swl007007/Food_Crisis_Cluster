# D38 root next options: design note (2026-10-02)

**Status:** design discussion only. This is NOT implementation or training approval. No fits were run for this note. A new D38 spec and review are required before any training; nothing expands automatically.

## Starting evidence (D36/D37, E3 matched persistence keys)

- Root vs persistence: group 00 root 1,585 FP vs 0; 01 root 701 TP vs 0; 10 root 3,444 FP vs 7,614; 11 root 9,060 TP / 2,647 FN vs 11,707 TP. Crisis F1 .551516 vs .586406.
- The root's deficit sits mainly in missed persisting crises (11) and stable non-crisis FP (00); its strengths are onset (01) and relief (10). A next design must keep both directions.
- The root already receives `hist_phase_o00` as a feature. Any hypothesis below concerns how finite boosting parameterises persistence (inductive bias), **not** missing persistence information.
- D37 rejects only the fixed 24-month weighting. It does not disprove temporal drift, nor all reweighting or regularisation. Its class-mix diagnostic (`d37_class_mix_diagnostic.json`) does not support a single prior-shift explanation of the FP increase.

## Options considered

| Option | Reading | Decision now |
|---|---|---|
| (A) Fitting-only four-class empirical transition prior as `base_margin` + learned two-way residual | Fewest extra choices; tests one parameterisation hypothesis | **Prioritise for investigation** as the next bounded candidate |
| (B) Explicit change label relative to origin, mapped back to four-class probabilities | Extra encoding, missing-origin fallback model, new class axis | Not now |
| (C) Further root regularisation / weighting sweep | No identified target; adds search on exposed folds | Not now |

## Candidate (A) as currently understood

- Margin = log of an **empirical transition prior** P(target class | origin class), estimated on fitting rows only, with a smoothing rule fixed in advance. It is not exact persistence: its argmax can differ from the origin label, so the margin-only comparator may differ from persistence.
- Missing origin: a fixed fallback (for example the fitting-row class prior), to be specified.
- Trees learn residual departures in both directions (onset and relief).

**Could invalidate it:** trees undo or are dominated by the margin; the prior mixes eras (2013–2016 dominate fitting rows, the same confound as D37); rare-origin cells are thinly supported; margins must be passed at every prediction; a model that collapses to persistence would not be a learned improvement.

## Checks required before a training spec (zero-fit)

1. Rare-origin support: fitting-row counts per origin × target cell, per root.
2. Era mixture of the fitting rows that estimate the prior.
3. Missing-origin rate in fitting and E3 rows, per root.
4. Margin-only comparator on the same E3 keys (argmax of the prior, zero trees), against persistence and the original root.

## Minimum same-key evidence if later approved

On the 21 D34 E3 roots and the same keys: original root, anchored root, persistence and the **mandatory margin-only comparator**; crisis F1 pooled and mean-fold, crisis Brier loss; departure rate from persistence and who is right on disagreements; D36 transition-group TP/FP changes (post-hoc only). C stays descriptive. These dates are exposed development folds, so this is development evidence only, with no further iteration on them.
