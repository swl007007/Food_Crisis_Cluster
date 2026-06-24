# GeoRF Partitioned SHAP Group Heatmap

Values are relative SHAP attribution shares for the partitioned GeoRF model.
They are not causal effects and are not retraining ablation deltas.
Each heatmap cell reports the mean group share of mean absolute SHAP values across the 12 February, June, and October target months in 2021-2024, plus one standard deviation.
SHAP shares are computed over evaluated local-partition samples only; fallback, unmapped, or missing-model samples are excluded from SHAP attribution and counted in the manifest.
