"""P3 orchestration: Stage1 for every H x (G, L), R44 selection, frozen maps.

Reads ``<run>/prepared`` (artifact hashes re-verified against the manifest)
and writes ``<run>/stage1`` plus the shared ``<run>/models`` store. For each H
the four G root quartets are fitted once on F and shared by L1/L2 (R48); each
of the eight candidates then runs its own search. The complete common S key
set is routed through each candidate's terminal providers, projected, decoded
and scored by pooled exact crisis F1; the R44 winner's terminal membership map
and recipe are frozen for Stage3. Stage1 providers are evidence only: Stage3
refits on its own windows.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_geoxgb import metrics, quartet, stage1
from ipcch_geoxgb.artifacts import sha256_file, write_json
from ipcch_geoxgb.contract import input_path, load_experiment_contract, load_feature_schema, load_inputs
from ipcch_geoxgb.errors import ContractError, TechnicalError
from ipcch_geoxgb.geography import load_adjacency_cache
from ipcch_geoxgb.modelstore import ModelStore, array_digest
from ipcch_geoxgb.preflight import verify_identities
from ipcch_geoxgb.runtime import probe_runtime

QUARTET_CODE = Path(quartet.__file__)


def verify_prepared(prepared: Path) -> dict:
    """Re-hash every prepared artifact against its manifest entry (identity binding)."""
    manifest_path = prepared / "prepared-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, digest in manifest["artifacts_sha256"].items():
        if sha256_file(prepared / name) != digest:
            raise TechnicalError(f"prepared artifact {name} does not match its manifest digest")
    return {"manifest_sha256": sha256_file(manifest_path), "manifest": manifest}


def environment_identity() -> dict:
    runtime = probe_runtime()
    if not runtime["matches_lock"]:
        raise ContractError(f"runtime does not match the lock: {runtime['mismatches']}")
    return {
        "lock": runtime["lock_version"],
        "python": runtime["python"],
        "xgboost": runtime["packages"]["xgboost"],
        "numpy": runtime["packages"]["numpy"],
        "quartet_code_sha256": sha256_file(QUARTET_CODE),
    }


def area_neighbours(inputs: dict) -> dict[int, list[int]]:
    """Shared-boundary neighbours by canonical area ID from the frozen cache."""
    verify_identities({"adjacency_cache": inputs["adjacency_cache"]})
    cache = load_adjacency_cache(input_path(inputs["adjacency_cache"]))
    group = {int(i): int(a) for i, a in cache["polygon_group_mapping"].items()}
    out = {}
    for index, values in cache["adjacency_dict"].items():
        out[group[int(index)]] = sorted(group[int(v)] for v in np.asarray(values).tolist())
    return out


def load_horizon(prepared: Path, h: int) -> tuple[pd.DataFrame, np.ndarray]:
    keys = pd.read_csv(prepared / f"keys_h{h:02d}.csv.gz")
    X = np.load(prepared / f"X_rich561_h{h:02d}.npy", mmap_mode="r")
    if len(keys) != X.shape[0] or X.shape[1] != 561:
        raise TechnicalError(f"H{h}: keys/X shape mismatch")
    return keys, X


def key_digest(frame: pd.DataFrame) -> str:
    return array_digest(frame[["admin_code", "target_ord"]].to_numpy(dtype=np.int64))


def run_learn_map(run_dir: Path) -> dict:
    started = time.time()
    contract = load_experiment_contract()
    schema = load_feature_schema()
    inputs = load_inputs()["inputs"]
    prepared = run_dir / "prepared"
    bound = verify_prepared(prepared)
    env = environment_identity()
    out = run_dir / "stage1"
    out.mkdir()
    store = ModelStore(run_dir / "models", out / "model_requests.jsonl")
    neighbours = area_neighbours(inputs)
    split = pd.read_csv(prepared / "stage1_split.csv.gz")
    Y_COLS = list(quartet.TARGETS)
    base_identity = {
        "prepared_manifest_sha256": bound["manifest_sha256"],
        "schema": [schema["schema_version"], schema["ordered_names_sha256"]],
        "availability": bound["manifest"]["availability_policy"]["id"],
        "weights": "unit",
        "env": env,
    }
    summary = {"stage": "P3-stage1", "base_identity": base_identity, "horizons": {}}

    for h in contract["calendar"]["horizons_months"]:
        keys, X = load_horizon(prepared, h)
        roles = split.rename(columns={"month_ord": "target_ord"})[["admin_code", "target_ord", "split_role"]]
        tagged = keys.reset_index().merge(roles, on=["admin_code", "target_ord"], how="left", validate="one_to_one")
        fit_rows = tagged.loc[tagged.split_role == "fit", "index"].to_numpy()
        val_rows = tagged.loc[tagged.split_role == "validation", "index"].to_numpy()
        fit_keys = keys.iloc[fit_rows].reset_index(drop=True)
        val_keys = keys.iloc[val_rows].reset_index(drop=True)
        X_fit = np.asarray(X[fit_rows])
        X_val = np.asarray(X[val_rows])
        Y_fit = fit_keys[Y_COLS].to_numpy(dtype=np.float64)
        fit_identity = {"fit_keys": key_digest(fit_keys), "X": array_digest(X_fit), "Y": array_digest(Y_fit)}
        hdir = out / f"h{h:02d}"
        hdir.mkdir()
        entries = []
        for gid in contract["model"]["global_recipes"]:
            gparams, grounds = quartet.global_params(contract, gid)
            g_identity = {**base_identity, "scope": "stage1-global", "H": h, "G": gid, "params": gparams,
                          "rounds": grounds, **fit_identity}
            root_quartet, root_use = store.get_or_fit(
                g_identity,
                lambda: quartet.fit_global_quartet(X_fit, Y_fit, gparams, grounds),
                {"stage": "stage1", "H": h, "G": gid, "purpose": "root_global"},
            )
            root = (root_use["identity_sha256"], root_quartet)
            for lid in contract["model"]["local_recipes"]:
                lparams, lrounds = quartet.local_params(contract, lid)
                name = f"{gid}{lid}"

                def fit_local(areas, mask, gid=gid, lid=lid, lparams=lparams, lrounds=lrounds, root=root):
                    child_keys = fit_keys[mask]
                    Xc, Yc = X_fit[mask], Y_fit[mask]
                    identity = {**base_identity, "scope": "stage1-local", "H": h, "G": gid, "L": lid,
                                "params": lparams, "rounds": lrounds,
                                "region_areas": array_digest(np.asarray(areas, dtype=np.int64)),
                                "global_identity": root[0], "global_boosters": root[1].booster_shas(),
                                "fit_keys": key_digest(child_keys), "X": array_digest(Xc), "Y": array_digest(Yc)}
                    q, use = store.get_or_fit(
                        identity,
                        lambda: quartet.continue_local_quartet(root[1], Xc, Yc, lparams, lrounds),
                        {"stage": "stage1", "H": h, "candidate": name, "purpose": "child_local"},
                    )
                    return use["identity_sha256"], q

                result, providers = stage1.search_candidate(
                    fit_keys=fit_keys, val_keys=val_keys, X_fit=X_fit, X_val=X_val, root=root,
                    fit_local=fit_local, neighbours=neighbours, contract=contract,
                )
                entries.append(_score_candidate(hdir, name, gid, lid, contract, val_keys, X_val, result, providers))
        selection = stage1.select_winner(entries)
        ledger = [{k: (str(v) if k == "f1_exact" and v is not None else v) for k, v in e.items()} for e in entries]
        write_json(hdir / "selection.json", {"H": h, "selection": selection, "candidates": ledger})
        if selection["status"] != "selected":
            raise TechnicalError(f"H{h}: selection_unavailable (all candidates NA); no winner frozen")
        winner = next(e for e in entries if e["candidate"] == selection["winner"])
        frozen = _freeze(out, h, winner, neighbours, contract)
        summary["horizons"][str(h)] = {
            "fit_keys": int(len(fit_keys)), "validation_keys": int(len(val_keys)),
            "winner": selection["winner"], "ranking": selection["ranking"], "frozen": frozen,
            "candidates": {e["candidate"]: {"f1": None if e["f1_exact"] is None else str(e["f1_exact"]),
                                            "terminal_regions": e["terminal_regions"],
                                            "accepted_splits": e["accepted_splits"]} for e in entries},
        }
    summary["model_store"] = dict(store.counts)
    summary["elapsed_seconds"] = round(time.time() - started, 1)
    digest = write_json(out / "stage1-summary.json", summary)
    return {**summary, "summary_sha256": digest}


def _score_candidate(hdir, name, gid, lid, contract, val_keys, X_val, result, providers) -> dict:
    """Route all S keys through terminal providers; save keyed evidence; exact F1."""
    area = val_keys["admin_code"].to_numpy()
    node_ids, provider_ids = stage1.terminal_routing(result.terminal, area)
    raw = np.zeros((len(val_keys), 4))
    for digest in sorted(set(provider_ids)):
        rows = provider_ids == digest
        raw[rows] = providers[digest].predict_raw(X_val[rows])
    from ipcch_geoxgb import projection  # noqa: PLC0415

    q_star, phase = projection.project_and_decode(raw)
    counts = metrics.crisis_counts(val_keys["phase_truth"].to_numpy(), phase)
    f1 = metrics.exact_f1(counts)
    cdir = hdir / name
    cdir.mkdir()
    pred = val_keys[["admin_code", "target_ord", "target_month", "phase_truth", "crisis_truth"]].copy()
    pred["node_id"] = node_ids
    pred["provider"] = provider_ids
    for k, q in enumerate(quartet.TARGETS):
        pred[f"{q}_raw"] = raw[:, k]
        pred[f"{q}_star"] = q_star[:, k]
    pred["phase_pred"] = phase
    pred.to_csv(cdir / "s_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    membership = pd.DataFrame(
        [(int(a), n.node_id, n.provider, n.provider_kind) for n in result.terminal for a in n.areas],
        columns=["admin_code", "node_id", "provider", "provider_kind"],
    ).sort_values("admin_code")
    membership.to_csv(cdir / "terminal_map.csv", index=False)
    write_json(cdir / "decisions.json", {"candidate": name, "decisions": result.decisions})
    g = contract["model"]["global_recipes"][gid]
    loc = contract["model"]["local_recipes"][lid]
    return {
        "candidate": name, "g_id": gid, "l_id": lid,
        "f1_exact": f1, "counts": counts,
        "terminal_regions": len(result.terminal),
        "accepted_splits": sum(1 for d in result.decisions if d["outcome"] == "accepted"),
        "scans": sum(1 for d in result.decisions if "m_final" in d),
        "global_rounds": g["rounds"], "local_rounds": loc["appended_rounds"],
        "global_depth": g["max_depth"], "local_depth": loc["max_depth"],
        "map_sha256": hashlib.sha256(membership.to_csv(index=False).encode()).hexdigest(),
    }


def _freeze(out: Path, h: int, winner: dict, neighbours: dict, contract: dict) -> dict:
    """Freeze the winning terminal membership map (not its Stage1 providers)."""
    source = out / f"h{h:02d}" / winner["candidate"] / "terminal_map.csv"
    membership = pd.read_csv(source, dtype={"node_id": str})
    for node_id in membership["node_id"].unique():
        stage1.node_depth(node_id)  # malformed/lost IDs stop here
    frozen = membership[["admin_code", "node_id"]].sort_values("admin_code")
    path = out / f"frozen_map_h{h:02d}.csv"
    frozen.to_csv(path, index=False)
    regions = {}
    for node_id, part in frozen.groupby("node_id"):
        regions[node_id] = stage1.component_diagnostics(part["admin_code"].to_numpy(), neighbours)
    record = {
        "H": h,
        "candidate": winner["candidate"],
        "G": winner["g_id"],
        "L": winner["l_id"],
        "f1_exact": str(winner["f1_exact"]),
        "terminal_regions": winner["terminal_regions"],
        "accepted_split": winner["accepted_splits"] > 0,
        "map_sha256": sha256_file(path),
        "learned_areas": int(len(frozen)),
        "connectivity": regions,
        "note": "membership only; Stage3 refits quartets on its own windows (clarification 2)",
    }
    write_json(out / f"frozen_h{h:02d}.json", record)
    return record
