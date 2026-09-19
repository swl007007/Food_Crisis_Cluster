# Design

Add one ETH-specific experiment script beside the aligned-refit workflow. It
reads the four frozen v5 input snapshots, joins raw `fews_ipc` truth one-to-one,
fits the bounded multiclass XGBoost search independently for each temporal fold,
and writes the agreed audit and comparison artifacts to a new run directory.

Reuse the existing v5 fold definitions, keys, predictors, GeoRF predictions, and
FEWS NET timing fields. Keep production Stage 1-3 and all prior run directories
read-only. Suppress 2021-06 explicitly and calculate every comparison on identical
admin-month support.

One script owns fitting, evaluation, and the compact plot; no new framework,
shared abstraction, configuration layer, or legacy GeoXGB integration is needed.
