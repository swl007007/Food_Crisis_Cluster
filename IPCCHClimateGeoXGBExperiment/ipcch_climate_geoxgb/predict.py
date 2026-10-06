"""P4 orchestration: rolling Stage3 over every scheduled main and 2026 fold.

Reads ``<run>/prepared`` (re-verified) and the frozen ``<run>/stage1`` maps
(each map's SHA256 must equal its freeze record), writes ``<run>/stage3``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_climate_geoxgb import stage3
from ipcch_climate_geoxgb.artifacts import record_incomplete, sha256_file, write_json
from ipcch_climate_geoxgb.contract import load_experiment_contract, load_feature_schema
from ipcch_climate_geoxgb.errors import TechnicalError
from ipcch_climate_geoxgb.learnmap import environment_identity, load_horizon, verify_prepared
from ipcch_climate_geoxgb.modelstore import ModelStore
from ipcch_climate_geoxgb.stage1 import node_depth

GZ = {"method": "gzip", "mtime": 0}


def load_frozen(stage1_dir: Path, h: int, prepared_manifest_sha256: str, contract_version: str,
                schema_identity: list) -> tuple[dict, dict]:
    """The frozen map for H, bound to this H, this prepared data and the Stage1 winner.

    Rejects a record/CSV copied from another H or run, a map that differs from
    its record, a record that differs from the completed Stage1 summary or its
    selection ledger, and duplicate areas.
    """
    record = json.loads((stage1_dir / f"frozen_h{h:02d}.json").read_text(encoding="utf-8"))
    summary_path = stage1_dir / "stage1-summary.json"
    if not summary_path.is_file():
        raise TechnicalError("Stage1 summary missing: learn-map did not complete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selection_path = stage1_dir / f"h{h:02d}" / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    problems = []
    if record.get("H") != h:
        problems.append(f"record is for H{record.get('H')}")
    if record.get("prepared_manifest_sha256") != prepared_manifest_sha256:
        problems.append("record is bound to different prepared data")
    if record.get("contract_version") != contract_version:
        problems.append("record is bound to a different contract version")
    if record.get("schema") != list(schema_identity):
        problems.append("record is bound to a different feature schema")
    base = summary.get("base_identity", {})
    if base.get("schema") != list(schema_identity) or base.get("prepared_manifest_sha256") != prepared_manifest_sha256:
        problems.append("Stage1 summary identity differs from the current schema/prepared data")
    if summary.get("horizons", {}).get(str(h), {}).get("frozen") != record:
        problems.append("record differs from the Stage1 summary")
    if selection.get("selection", {}).get("winner") != record.get("candidate") or \
            record.get("selection_sha256") != sha256_file(selection_path):
        problems.append("record differs from the selection ledger")
    if record.get("candidate") != f"{record.get('G')}{record.get('L')}":
        problems.append("candidate id differs from its G/L recipe")
    path = stage1_dir / f"frozen_map_h{h:02d}.csv"
    if sha256_file(path) != record.get("map_sha256"):
        problems.append("map bytes differ from the freeze record")
    if problems:
        raise TechnicalError(f"frozen map H{h} rejected: {problems}")
    fmap = pd.read_csv(path, dtype={"node_id": str})
    if fmap["admin_code"].duplicated().any():
        raise TechnicalError(f"frozen map H{h} has duplicate areas")
    for node_id in fmap["node_id"].unique():
        node_depth(node_id)
    return record, dict(zip(fmap["admin_code"].astype(int), fmap["node_id"]))


def run_predict(run_dir: Path) -> dict:
    """Stage3 for all H; any exception leaves a durable INCOMPLETE record (R41)."""
    context: dict = {"stage": "stage3"}
    try:
        return _run_predict(run_dir, context)
    except Exception as error:
        record_incomplete(run_dir, "stage3", context, error)
        raise


def _run_predict(run_dir: Path, context: dict) -> dict:
    started = time.time()
    contract = load_experiment_contract()
    schema = load_feature_schema()
    prepared = run_dir / "prepared"
    bound = verify_prepared(prepared, contract["calendar"]["horizons_months"])
    artifacts = bound["manifest"]["artifacts_sha256"]
    env = environment_identity()
    stage1_dir = run_dir / "stage1"
    out = run_dir / "stage3"
    out.mkdir()
    store = ModelStore(run_dir / "models", out / "model_requests.jsonl")
    calendar = pd.read_csv(prepared / "fold_calendar.csv")
    base_identity = {
        "prepared_manifest_sha256": bound["manifest_sha256"],
        "schema": [schema["schema_version"], schema["ordered_names_sha256"]],
        "availability": bound["manifest"]["availability_policy"]["id"],
        "weights": "unit",
        "env": env,
    }
    summary = {"stage": "P4-stage3", "base_identity": base_identity, "horizons": {}}
    for h in contract["calendar"]["horizons_months"]:
        context.update(H=h, fold_id=None, origin_ord=None)
        frozen, region_of = load_frozen(stage1_dir, h, bound["manifest_sha256"], contract["contract_version"],
                                        [schema["schema_version"], schema["ordered_names_sha256"]])
        keys, X = load_horizon(prepared, h)
        ctx = stage3.HorizonContext(
            h=h, keys=keys, X=X,
            artifact_sha={"X": artifacts[f"X_rich601_h{h:02d}.npy"], "keys": artifacts[f"keys_h{h:02d}.csv.gz"]},
            region_of=region_of, local_enabled=bool(frozen["accepted_split"]),
            gid=frozen["G"], lid=frozen["L"], contract=contract, store=store,
            base_identity={**base_identity, "frozen_map_sha256": frozen["map_sha256"]},
            observed_months=np.unique(keys["target_ord"].to_numpy()),
        )
        hdir = out / f"h{h:02d}"
        hdir.mkdir()
        folds = calendar[calendar["horizon_months"] == h].sort_values(["target_ord", "period"])
        predictions, ledger = [], []
        with open(hdir / "gate_decisions.jsonl", "w", encoding="utf-8", newline="\n") as gate_log:
            for fold in folds.to_dict("records"):
                context.update(fold_id=fold["fold_id"], target_ord=int(fold["target_ord"]),
                               origin_ord=int(fold["origin_ord"]), period=fold["period"])
                result = stage3.run_fold(ctx, fold)
                ledger.append(result["ledger"])
                if result["predictions"] is not None:
                    predictions.append(result["predictions"])
                for decision in result["gate"]:
                    gate_log.write(json.dumps({"fold_id": fold["fold_id"], **decision}, default=str) + "\n")
                if result["pairs"] is not None and len(result["pairs"]):
                    result["pairs"].to_csv(hdir / f"pairs_{fold['fold_id']}.csv.gz", index=False, compression=GZ)
        frame = (pd.concat(predictions, ignore_index=True) if predictions
                 else pd.DataFrame(columns=stage3.PREDICTION_COLUMNS))  # all folds empty: ledger-only H (R47)
        frame.to_csv(hdir / "predictions.csv.gz", index=False, compression=GZ)
        pd.DataFrame(ledger).to_csv(hdir / "fold_ledger.csv", index=False)
        summary["horizons"][str(h)] = {
            "recipe": f"{frozen['G']}{frozen['L']}", "local_enabled": ctx.local_enabled,
            "regions": len(ctx.regions), "folds": len(ledger),
            "scored_folds": sum(1 for r in ledger if r["status"] == "scored"),
            "prediction_rows": int(len(frame)),
            "local_rows": int((frame["route"] == "local").sum()),
            "predictions_sha256": sha256_file(hdir / "predictions.csv.gz"),
        }
    summary["model_store"] = dict(store.counts)
    summary["elapsed_seconds"] = round(time.time() - started, 1)
    digest = write_json(out / "stage3-summary.json", summary)
    return {**summary, "summary_sha256": digest}
