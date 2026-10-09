# Evidence: MLflow readable naming rebuild (2026-10-09)

Lifecycle: the close audit was waived by the user up front (grill decision D18). This file is
delivery evidence, not an audit pass. The user accepted the new store on 2026-10-09 and chose
the follow-ups below.

## Stores

| Store | Path |
|---|---|
| Before (experiments `IPCCH`, `IPCCH Summary`), kept | `~/ipcch-mlflow-backups/20261009-store-before-readable-naming/` (moved whole) |
| Live store (final build) | `~/.local/share/ipcch-mlflow/` (experiments `IPCCH - detailed runs` id 1, `IPCCH - dashboard` id 2) |

Interim builds (an attempt stopped by a WSL restart, the build before the `zz_prov.` prefix,
the build before the tag-order fix) and their pre-dashboard backups were deleted at the user's
request ("delete interim copies, keep the original store").

## Results of the final build (files in this folder)

- Import (`import-result.json`): 6 families, 126 records, 20,528 metrics, 1,679 s.
- Deep verify (`verify-detailed.json`): 126 records; every artifact downloaded and hashed;
  44,203 `models.tar` members checked.
- Dashboard (`dashboard-apply.json`): 136 rows (120 + 16 MLP seed means), 5,658 values, 4 NA,
  140 metric names, 74 datasets (66 evaluation + 8 training pools), 68 registered models /
  100 versions / 100 external logged models, 868 inputs; verify passed and the detailed
  experiment is unchanged (126 runs, inventory hash equal before/after).
- Old vs new (`rebuild-check.json`, passed, 0 problems): 20,528 / 20,528 detailed values and
  4,424 / 4,424 Summary values identical under their new names; 330 contrast values and 144
  panel values not in the old Summary (GeoXGB maps to 2024 `combined`) equal their detailed
  source; 760 seed-mean values equal the mean of the three seed rows; 74 dataset names, each
  with one digest; 88 family x arm x window x lead combinations in both stores.
- Tag order: in all 262 runs of the live DB, no `zz_prov.*` tag is written before a readable
  tag (SQLite rowid order = order on the run page).
- Tests: 41 pass (`$PY -m unittest discover -s IPCCHMLflow/tests`).

## Browser check (`browser/`, headless Edge, 3 loads per grid)

| Page | First rows | 100-row search payload |
|---|---|---|
| `IPCCH - dashboard` runs | 2.4-2.7 s | 3.65 MB (old `IPCCH Summary`: 0.71 MB) |
| `IPCCH - detailed runs` runs | 1.8-2.2 s | 6.09 MB (old `IPCCH`: 3.82 MB) |

The payload grew (wide rows, readable keys, one description per run); first-paint time is in
a similar range. Screenshots: dashboard runs, detailed runs, a dashboard run page (name,
datasets, tags readable-first, registered model), registered models list, one registered model
page (description), the dashboard's logged models. Narrow default name columns cut names after
the family title; the user chose to keep the titles.

## Fixes found during the live builds (each guarded by a test)

- Long family titles did not all start with the short title (truncated names differed).
- Training-pool descriptor carried per-family file paths, so identical content got two
  descriptors; paths moved to `dashboard/row.json`.
- Evaluation descriptor carried the period role, so the 12-month `combined` period of GeoXGB
  maps to 2024 (= the holdout rows) got a second descriptor; the role stays on metric keys.
- `rebuild_check.py` mis-parsed panel keys absent from the old Summary.
- Provenance showed first on run pages: `_` sorts before lowercase letters and the run page
  lists tags in insertion order. Prefix changed to `zz_prov.` (user decision) and every run is
  created with its tags in key order.
