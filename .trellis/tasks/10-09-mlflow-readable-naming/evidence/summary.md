# Evidence: MLflow readable naming rebuild (2026-10-09)

Lifecycle: the close audit was waived by the user up front (grill decision D18). This file is
delivery evidence, not an audit pass. The new store awaits the user's approval; the old store
is kept until then.

## Stores

| Store | Path |
|---|---|
| Before (experiments `IPCCH`, `IPCCH Summary`) | `~/ipcch-mlflow-backups/20261009-store-before-readable-naming/` (moved whole) |
| First rebuild attempt (stopped by a WSL restart; superseded) | `~/ipcch-mlflow-backups/20261009-rebuild-attempt1-incomplete/` |
| New detailed store before the dashboard write | `~/ipcch-mlflow-backups/20261009-detailed-before-dashboard/` |
| New live store | `~/.local/share/ipcch-mlflow/` (experiments `IPCCH - detailed runs` id 1, `IPCCH - dashboard` id 2) |

## Results

- Import (`import-result.json`): 6 families, 126 records, 20,528 metrics, 1,199 s.
- Deep verify (`verify-detailed.json`): 126 records; every artifact downloaded and hashed, every
  `models.tar` member checked.
- Dashboard (`dashboard-apply.json`): 136 rows (120 + 16 MLP seed means), 5,658 values, 4 NA,
  140 metric names, 74 datasets (66 evaluation + 8 training pools), 68 registered models /
  100 versions / 100 external logged models, 868 inputs; verify passed and the detailed
  experiment is unchanged (126 runs, inventory hash equal before/after).
- Old vs new (`rebuild-check.json`, passed, 0 problems): 20,528 / 20,528 detailed values and
  4,424 / 4,424 Summary values identical under their new names; 330 contrast values and 144
  panel values not in the old Summary (GeoXGB maps to 2024 `combined`) equal their detailed
  source; 760 seed-mean values equal the mean of the three seed rows; 74 dataset names, each
  with one digest; 88 family x arm x window x lead combinations in both stores.
- Tests: 41 pass (`$PY -m unittest discover -s IPCCHMLflow/tests`).

## Browser check (`browser/`, headless Edge, 3 loads per grid)

| Page | First rows | 100-row search payload |
|---|---|---|
| `IPCCH - dashboard` runs | 2.0-2.5 s | 3.64 MB (old `IPCCH Summary`: 0.71 MB) |
| `IPCCH - detailed runs` runs | 1.5-1.7 s | 6.08 MB (old `IPCCH`: 3.82 MB) |

The payload grew (wide rows, readable keys, one description per run); first-paint time is in
the same range as before. Screenshots: dashboard runs, detailed runs, a dashboard run page
(name, datasets, tags, registered model), registered models list, one registered model page
(description), the dashboard's logged models.

Observed in the UI:
- Run pages sort tags by character code, so `_prov.*` tags appear first, not last as stated
  during the grill (MLflow tag names allow only letters, digits, `_ - . / space`).
- Narrow default name columns cut names after the family title; arm and lead show when the
  column is widened or on the run page.

## Fixes found by the live run (each guarded by a test)

- Long family titles did not all start with the short title (truncated names differed).
- Training-pool descriptor carried per-family file paths, so identical content got two
  descriptors; paths moved to `dashboard/row.json`.
- Evaluation descriptor carried the period role, so the 12-month `combined` period of GeoXGB
  maps to 2024 (= the holdout rows) got a second descriptor; the role stays on metric keys.
- `rebuild_check.py` mis-parsed panel keys absent from the old Summary.
