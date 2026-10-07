"""P1 orchestration: all H, all current blocks; keyed outputs and ledgers (design sections 5, 9)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ipcch_yearly_xgb import schedule
from ipcch_yearly_xgb.artifacts import record_incomplete, sha256_file, write_json
from ipcch_yearly_xgb.engine import Engine
from ipcch_yearly_xgb.modelstore import ModelStore

GZ = {"method": "gzip", "mtime": 0}


def run_horizon(out: Path, hz, calendar, contract: dict, store: ModelStore, env: dict) -> dict:
    eng = Engine(hz, contract, store, env)
    hdir = out / f"h{hz.h:02d}"
    hdir.mkdir()
    preds, block_rows, fold_rows = [], [], []
    with open(hdir / "gate_decisions.jsonl", "w", encoding="utf-8", newline="\n") as glog:
        for block in schedule.blocks(calendar, hz.h, eng.first):
            res = eng.run_block(block)
            block_rows.append(res["block"])
            for f in block.folds:
                n = int(len(hz.rows_at(int(f["target_ord"]))))
                fold_rows.append({"fold_id": f["fold_id"], "period": f["period"], "H": hz.h,
                                  "target_ord": int(f["target_ord"]), "block_id": block.block_id,
                                  "eval_keys": n, "status": "scored" if n else "no_valid_target"})
            if res["predictions"] is not None:
                preds.append(res["predictions"])
            for d in res["gates"]:
                glog.write(json.dumps(d, sort_keys=True, default=str) + "\n")
            if res["pairs"] is not None and len(res["pairs"]):
                res["pairs"].to_csv(hdir / f"pairs_{block.block_id}.csv.gz", index=False, compression=GZ)
    frame = pd.concat(preds, ignore_index=True)
    frame.to_csv(hdir / "predictions.csv.gz", index=False, compression=GZ)
    pd.DataFrame(block_rows).to_json(hdir / "block_ledger.json", orient="records", indent=1)
    pd.DataFrame(fold_rows).to_csv(hdir / "fold_ledger.csv", index=False)
    (hdir / "weight_stats.json").write_text(json.dumps(eng.weight_stats, indent=1, sort_keys=True), encoding="utf-8")
    return {"recipe": eng.recipe, "blocks": len(block_rows), "folds": len(fold_rows),
            "scored_folds": sum(1 for f in fold_rows if f["status"] == "scored"), "prediction_rows": int(len(frame)),
            "local_rows": int((frame["route"] == "local").sum()), "diagnostic_rows": int(frame["local_eligible"].sum()),
            "predictions_sha256": sha256_file(hdir / "predictions.csv.gz")}


def run_predict(run_dir: Path, horizons: dict, calendar, contract: dict, env: dict, source_inventory: dict,
                readonly: bool = False, out_name: str = "predict", models_root: Path | None = None) -> dict:
    out = run_dir / out_name
    context = {"stage": out_name}
    try:
        out.mkdir()
        ledger = out / "model_requests.jsonl"
        store = ModelStore(models_root or (run_dir / "models"), ledger, readonly=readonly)
        started = time.time()
        summary = {"stage": out_name, "env": env, "source_inventory": source_inventory, "horizons": {}}
        for h, hz in horizons.items():
            context["H"] = h
            summary["horizons"][str(h)] = run_horizon(out, hz, calendar, contract, store, env)
        summary["store_counts"] = dict(store.counts)
        summary["elapsed_seconds"] = round(time.time() - started, 1)
        digest = write_json(out / "predict-summary.json", summary)
        return {**summary, "summary_sha256": digest}
    except Exception as error:
        record_incomplete(run_dir, out_name, context, error)
        raise
