"""Replay first/last nonempty main fold per horizon against frozen saved rows."""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "IPCCHGeoRFExperiment"))
import numpy as np
import pandas as pd
import xgboost
import baseline_runtime as brt
import run_pipeline as runner

RUN = Path("C:/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial/Food_Crisis_Cluster/IPCCHGeoRFExperiment/runs/ipcch-v1-20260920d")
schema = json.loads((RUN / "data/feature_schema.json").read_text())
frame = pd.read_csv(RUN / "data/feature_metadata.csv.gz")
values = np.load(RUN / "data/feature_values.npy")
for index, column in enumerate(schema["feature_columns"]):
    frame[column] = values[:, index]
matrix = SimpleNamespace(frame=frame, metadata_columns=schema["metadata_columns"],
                         feature_columns=schema["feature_columns"])
assignments = pd.read_csv(RUN / "stage1/area_assignments.csv", keep_default_na=False)
valid = pd.read_csv(RUN / "data/target_ledger_valid.csv.gz")
panel = runner.build_stage3_panel(matrix, assignments, valid)
folds = pd.read_csv(RUN / "stage3/folds.csv")
eligible = folds[(folds.period == "main") & (folds.test_rows > 0)].sort_values("target_month")
selected = pd.concat([group.iloc[[0, -1]] for _, group in eligible.groupby("horizon_months")])
saved = pd.read_csv(RUN / "stage3/predictions.csv.gz", keep_default_na=False,
                    float_precision="round_trip")
runtime = brt.BaselineRuntime(
    root=ROOT / "IPCCHGeoRFExperiment/runs/review-baseline/baseline/GeoRFBaseline",
    release_sha256=brt.RELEASE_SHA256, manifest_version="0.1.0-f1-nosmote",
    manifest_source_commit="2dfa121a9398de9a1918ba9c0af34b31ecbb117a",
    payload_files_verified=45, patch_applied=False, patch_diff="",
    pristine_target_sha256="", patched_target_sha256="")
results = []
with (ROOT / "replay-fit-stdout.log").open("w", encoding="utf-8") as log:
    with contextlib.redirect_stdout(log), brt.baseline_imports(runtime):
        helpers, _ = runner._load_stage3_helpers(runtime)
        from src.customize.customize import OutOfRangeImputer
        for row in selected.itertuples():
            target = int(runner.pdata.month_ordinal(int(row.target_month[:4]), int(row.target_month[5:])))
            fold = runner.Fold(row.fold_id, row.period, int(row.horizon_months),
                               target, target - int(row.horizon_months))
            started = time.time()
            outcome = runner.fit_fold(
                fold, panel, helpers,
                lambda: OutOfRangeImputer(strategy="max_plus", multiplier=100.0),
                lambda: xgboost.XGBClassifier(missing=np.nan, **runner.STAGE3_XGB_PARAMS))
            runner.verify_xgb_configuration(outcome["xgb_model"])
            got = outcome["rows"].sort_values("admin_code").reset_index(drop=True)
            expected = saved[saved.fold_id == row.fold_id].sort_values("admin_code").reset_index(drop=True)
            assert np.array_equal(got.admin_code, expected.admin_code)
            differences = {}
            for arm in ("partitioned_rf", "pooled_rf", "xgb"):
                probability = "prob_" + arm
                difference = float(np.max(np.abs(got[probability] - expected[probability].astype(float))))
                assert difference == 0.0, (row.fold_id, probability, difference)
                assert np.array_equal(got["pred_" + arm], expected["pred_" + arm])
                differences[arm] = difference
            result = {"fold_id": row.fold_id, "rows": len(got), "train_rows": outcome["record"]["train_rows"],
                      "max_probability_difference": differences, "xgb_effective_configuration_checked": True,
                      "seconds": round(time.time() - started, 2)}
            results.append(result)
            (ROOT / "replay-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps(result), file=sys.__stdout__, flush=True)
assert len(results) == 8
print("PASS: first/last nonempty main fold in all four horizons reproduced exactly")
