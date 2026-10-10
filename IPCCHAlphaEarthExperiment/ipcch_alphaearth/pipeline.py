"""Experiment stages over prepared inputs: develop -> freeze-map -> predict (design sections 6-10, 13).

develop:    19 historical B1 quartets (honest OOT providers) + 8 development B0; for each H and development year
            (2021, 2022) rebuild transforms, B2 and three lambda growth paths on residual labels U <= that block's
            fit origin; evaluate nine recipes against B0/B1/B2 with D23 guards; select per H (D13) or B2.
freeze-map: relearn each selected H's map on residual labels through 2022-12 with the selected recipe, freeze
            tree/membership/hashes (MAP_FREEZE.json) before any main G prediction exists.
predict:    4 bridge B1 quartets for the 22 pre-main 2023 slots; then per H in chronological order 11 main + 4
            supplement B1/B0 quartets; annual B2/G coefficients on every eligible saved provider (U <= O) with the
            frozen map, D20 support and exact B2 reuse below two supported leaves.

Every provider key has exactly one B1 provider (schedule.assign_providers); the provider's target is after its
fit origin and absent from its fitting keys. Nothing here selects on main-period outcomes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth import backbone, correction as cr, evaluation as ev, partition as pt, projection, schedule
from ipcch_alphaearth.artifacts import digest, read_json, record_incomplete, sha256_file, write_csv_gz, write_json
from ipcch_alphaearth.errors import ContractError, TechnicalError
from ipcch_alphaearth.months import month_label, parse_month

TARGETS = ("q2", "q3", "q4", "q5")
ARMS = ("B0", "B1", "B2", "G")


@dataclass
class HorizonInputs:
    h: int
    keys: pd.DataFrame  # admin_code, country_key, target_ord, origin_ord, phase_truth, q2..q5, persistence_*
    X296: np.ndarray
    D: np.ndarray  # (n, 128) row descriptors, NaN where not routable
    routable: np.ndarray  # (n,) bool
    core_idx: list  # three X296 column indices (D17 core inputs)

    def __post_init__(self):
        self.target = self.keys["target_ord"].to_numpy(np.int64)
        self.admin = self.keys["admin_code"].to_numpy(np.int64)
        self.Y = self.keys[list(TARGETS)].to_numpy(np.float64)
        self.Z = self.X296[:, self.core_idx]

    def X(self, arm: str) -> np.ndarray:
        if arm == "B0":
            return self.X296
        if arm == "B1":
            return np.hstack([self.X296, self.D])
        raise ValueError(arm)


@dataclass
class Inputs:
    horizons: dict
    features296: list
    features424: list
    admin_codes: np.ndarray  # admins with a descriptor row
    D_admin: np.ndarray  # (n_admin, 128)
    source: dict  # prepared/descriptor identities bound into every fit
    descriptor_names: list = field(default_factory=list)


def recipe_id(budget: int, lam: float) -> str:
    return f"K{budget}_lam{lam:g}"


class Workspace:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def stage(self, name: str) -> Path:
        path = self.run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def complete(self, name: str, payload: dict) -> None:
        write_json(self.run_dir / name / "COMPLETE.json", payload)

    def require(self, name: str) -> None:
        if not (self.run_dir / name / "COMPLETE.json").exists():
            raise ContractError(f"prerequisite stage '{name}' is not COMPLETE; run it explicitly first")


# --------------------------------------------------------------------------- backbone requests


def fit_block(hi: HorizonInputs, block: schedule.Block, arm: str, store: backbone.ModelStore, contract: dict, env: dict,
              source: dict, features: list, purpose: str) -> dict:
    half_life = contract["calendar"]["half_life_months"]
    rows = np.flatnonzero(hi.target <= block.fit_origin)
    X = hi.X(arm)
    w = schedule.decay_weights(hi.target[rows], block.fit_origin, half_life)
    keys = np.column_stack([hi.admin[rows], hi.target[rows]])
    Xf, Yf = np.ascontiguousarray(X[rows]), hi.Y[rows]
    ident = backbone.fit_identity(arm=arm, horizon=hi.h, block_id=block.block_id, role=block.role, fit_origin=block.fit_origin,
                                  rows=rows, keys=keys, X=Xf, Y=Yf, w=w, features=features, contract=contract, env=env, source=source)
    boosters, record = store.get_or_fit(ident, Xf, Yf, w, contract, purpose)
    pred_rows = np.flatnonzero((hi.target >= block.target_lo) & (hi.target <= block.target_hi))
    if (hi.target[pred_rows] <= block.fit_origin).any() or np.isin(pred_rows, rows).any():
        raise ContractError(f"{block.block_id}: a predicted key is inside its own fitting pool")
    raw = backbone.predict_quartet(boosters, X[pred_rows], int(contract["backbone"]["fixed"]["n_jobs"]))
    return {"identity": record["identity_digest"], "rows": pred_rows, "raw": raw, "fit_rows": int(len(rows))}


# --------------------------------------------------------------------------- correction contexts


def provider_frame(providers: dict) -> pd.DataFrame:
    return pd.concat(providers.values(), ignore_index=True) if providers else pd.DataFrame()


def correction_pool(hi: HorizonInputs, prov: pd.DataFrame, fit_origin: int, contract: dict) -> dict:
    """Eligible honest OOT residual rows at a fit origin: provider target U <= O (D29), D7 weights relative to O."""
    sel = prov[prov["target_ord"] <= fit_origin].sort_values("row", kind="mergesort")
    if (sel["provider_fit_origin"] >= sel["target_ord"]).any():
        raise ContractError("a correction row's provider fit origin is not before its target")
    rows = sel["row"].to_numpy(np.int64)
    if len(np.unique(rows)) != len(rows):
        raise ContractError("duplicate provider rows (one provider per key)")
    P = sel[[f"p_{q}" for q in TARGETS]].to_numpy(np.float64)
    w = schedule.decay_weights(hi.target[rows], fit_origin, contract["calendar"]["half_life_months"])
    return {"rows": rows, "P": P, "R": hi.Y[rows] - P, "Z": hi.Z[rows], "w": w, "routable": hi.routable[rows],
            "provider_blocks": sel["provider_block"].to_numpy(dtype=object),
            "provider_identities": sel["provider_identity"].to_numpy(dtype=object), "fit_origin": int(fit_origin)}


def make_pool(hi: HorizonInputs, pool: dict) -> tuple[cr.Transform, pt.Pool]:
    if len(pool["rows"]) == 0:  # callers branch to the declared identity correction first
        raise TechnicalError("make_pool called on an empty correction pool")
    transform = cr.Transform.fit(pool["P"], pool["Z"], pool["w"])
    V = transform.design(pool["P"], pool["Z"])
    return transform, pt.Pool(V, pool["R"], pool["w"], hi.admin[pool["rows"]], hi.target[pool["rows"]], pool["routable"])


def pool_record(hi: HorizonInputs, pool: dict) -> pd.DataFrame:
    rows = pool["rows"]
    return pd.DataFrame({"row": rows, "admin_code": hi.admin[rows], "target_ord": hi.target[rows],
                         "provider_block": pool["provider_blocks"], "provider_identity": pool["provider_identities"],
                         "weight": pool["w"], "routable": pool["routable"].astype(np.int64)})


def project_arm(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return projection.project_and_decode(raw)


def arm_panel(hi: HorizonInputs, rows: np.ndarray, raw: np.ndarray) -> dict:
    star, phase = project_arm(raw)
    return ev.panel(hi.keys["phase_truth"].to_numpy(np.int64)[rows], phase, hi.Y[rows, 1], star[:, 1], raw[:, 1])


def leaf_rows(hi: HorizonInputs, pmap: pt.PartitionMap, rows: np.ndarray) -> np.ndarray:
    return pmap.route(hi.D[rows], hi.routable[rows])


# --------------------------------------------------------------------------- develop


def run_develop(inputs: Inputs, contract: dict, ws: Workspace, store: backbone.ModelStore, env: dict) -> dict:
    out = ws.stage("develop")
    if (out / "COMPLETE.json").exists():
        raise ContractError("develop is COMPLETE; outputs are immutable")
    try:
        return _develop(inputs, contract, ws, out, store, env)
    except Exception as error:
        record_incomplete(ws.run_dir, "develop", {"stage": "develop"}, error)
        raise


def _develop(inputs: Inputs, contract: dict, ws: Workspace, out: Path, store, env) -> dict:
    t0 = time.time()
    c = contract["correction"]
    lambdas, budgets = [float(x) for x in c["regional_lambdas"]], [int(x) for x in c["leaf_budgets"]]
    selections, summary = {}, {}
    for h, hi in sorted(inputs.horizons.items()):
        blocks = schedule.horizon_blocks(contract, h)
        providers, dev_preds = {}, {}
        for b in [b for b in blocks if b.role == "historical"]:
            r1 = fit_block(hi, b, "B1", store, contract, env, inputs.source, inputs.features424, "historical_provider")
            providers[b.block_id] = pd.DataFrame({"row": r1["rows"], "target_ord": hi.target[r1["rows"]], "provider_block": b.block_id,
                                                  "provider_fit_origin": b.fit_origin, "provider_identity": r1["identity"],
                                                  **{f"p_{q}": r1["raw"][:, k] for k, q in enumerate(TARGETS)}})
            if b.development_year is not None:
                r0 = fit_block(hi, b, "B0", store, contract, env, inputs.source, inputs.features296, "development_B0")
                if not np.array_equal(r0["rows"], r1["rows"]):
                    raise ContractError(f"{b.block_id}: B0/B1 prediction keys differ")
                dev_preds[b.development_year] = {"block": b, "rows": r1["rows"], "B1": r1["raw"], "B0": r0["raw"]}
        prov = provider_frame(providers)
        write_csv_gz(out / f"providers_h{h:02d}.csv.gz", prov)
        recipe_records = {recipe_id(k, lam): {"recipe_id": recipe_id(k, lam), "leaf_budget": k, "lambda_region": lam, "years": {}}
                          for lam in lambdas for k in budgets}
        dev_rows_out = []
        for year, dp in sorted(dev_preds.items()):
            b = dp["block"]
            ctx = f"h{h:02d}_dev_{year}"
            pool = correction_pool(hi, prov, b.fit_origin, contract)
            write_csv_gz(out / f"pool_{ctx}.csv.gz", pool_record(hi, pool))
            rows = dp["rows"]
            P_eval, Z_eval = dp["B1"], hi.Z[rows]
            identity = len(pool["rows"]) == 0  # design section 6: zero eligible OOT rows -> declared identity (B2 = B1, G = B2)
            if identity:
                transform, ppool, b2 = None, None, None
                raw_b2 = cr.CorrectionFit.identity().predict(P_eval, Z_eval, np.full(len(rows), None, dtype=object))["b2"]
            else:
                transform, ppool = make_pool(hi, pool)
                active = transform.active()
                b2 = cr.solve_b2(ppool.total, active, float(c["shared_lambda"]), float(c["solver_residual_tolerance"]))
                raw_b2 = cr.correct(P_eval, transform.design(P_eval, Z_eval), b2["beta"])
            panels = {"B0": arm_panel(hi, rows, dp["B0"]), "B1": arm_panel(hi, rows, dp["B1"]), "B2": arm_panel(hi, rows, raw_b2)}
            frame = hi.keys.iloc[rows][["admin_code", "country_key", "target_ord", "phase_truth"]].copy()
            frame["context"] = ctx
            for arm, raw in (("B0", dp["B0"]), ("B1", dp["B1"]), ("B2", raw_b2)):
                star, phase = project_arm(raw)
                for k, q in enumerate(TARGETS):
                    frame[f"{arm}_{q}_raw"] = raw[:, k]
                frame[f"{arm}_phase"] = phase
            ctx_record = {"context": ctx, "fit_origin": month_label(b.fit_origin), "pool_rows": int(len(pool["rows"])),
                          "identity_empty_pool": identity,
                          "transform": None if identity else transform.record(), "beta_b2": None if identity else b2["beta"].tolist(),
                          "b2_objective": None if identity else b2["objective"], "lambdas": {}}
            if identity:
                g_panel = panels["B2"]
                guards = ev.guard_check(g_panel, {"B1": panels["B1"], "B2": panels["B2"], "B0": panels["B0"]}, contract)
                for lam in lambdas:
                    for k in budgets:
                        rid = recipe_id(k, lam)
                        recipe_records[rid]["years"][year] = {"gain": ev.f1_gain(g_panel, panels["B2"]), "guards_pass": guards["pass"],
                                                              "guards_failed": guards["failed"], "n_leaves": 1, "depth": 0,
                                                              "g_kind": "identity_empty_pool", "map": None, "g": ev.flat(g_panel)}
                        frame[f"G_{rid}_phase"] = frame["B2_phase"]
                        frame[f"G_{rid}_route"] = "identity_empty_pool"
            for lam in ([] if identity else lambdas):
                search = pt.Search(ppool, inputs.D_admin, inputs.admin_codes, active, contract, lam, inputs.descriptor_names)
                states = search.run(max(budgets), int(c["max_depth"]))
                write_csv_gz(out / f"candidates_{ctx}_lam{lam:g}.csv.gz", search.ledger_frame())
                ctx_record["lambdas"][f"{lam:g}"] = [{"step": s["step"], "objective": s["objective"], "map": s["map"].record(),
                                                      "fit": state_fit_record(s)} for s in states]
                for k in budgets:
                    st = pt.state_at_budget(states, k)
                    rid = recipe_id(k, lam)
                    fit = recipe_fit(transform, b2, st)
                    leaf_of = leaf_rows(hi, st["map"], rows)
                    pred = fit.predict(P_eval, Z_eval, leaf_of)
                    g_panel = arm_panel(hi, rows, pred["g"])
                    gain = ev.f1_gain(g_panel, panels["B2"])
                    guards = ev.guard_check(g_panel, {"B1": panels["B1"], "B2": panels["B2"], "B0": panels["B0"]}, contract)
                    recipe_records[rid]["years"][year] = {"gain": gain, "guards_pass": guards["pass"], "guards_failed": guards["failed"],
                                                          "n_leaves": st["map"].n_leaves, "depth": st["map"].depth if st["map"].n_leaves > 1 else 0,
                                                          "g_kind": fit.g_kind, "map": st["map"].record(), "g": ev.flat(g_panel)}
                    _, gphase = project_arm(pred["g"])
                    frame[f"G_{rid}_phase"] = gphase
                    frame[f"G_{rid}_route"] = pred["route"]
            ctx_record["panels"] = {arm: ev.flat(p) for arm, p in panels.items()}
            write_json(out / f"context_{ctx}.json", ctx_record)
            dev_rows_out.append(frame)
        write_csv_gz(out / f"development_predictions_h{h:02d}.csv.gz", pd.concat(dev_rows_out, ignore_index=True))
        sel = ev.select_recipe(list(recipe_records.values()))
        selections[h] = sel
        write_json(out / f"selection_h{h:02d}.json", _jsonable(sel))
        summary[str(h)] = {"selected": sel["selected"], "providers": int(len(prov))}
    ledger = pd.DataFrame(store.ledger)
    write_csv_gz(out / "request_ledger.csv.gz", ledger)
    result = {"stage": "develop", "selections": {str(h): s["selected"] for h, s in selections.items()}, "summary": summary,
              "fits": int((ledger["action"] == "fit").sum()) if len(ledger) else 0, "elapsed_seconds": round(time.time() - t0, 1)}
    ws.complete("develop", result)
    return result


def state_fit_record(state: dict) -> dict:
    f = state["fit"]
    if f["kind"] != "joint":
        return {"kind": "b2", "objective": float(f["objective"])}
    return {"kind": "joint", "beta": np.asarray(f["beta"]).tolist(), "u": np.asarray(f["u"]).tolist(), "leaf_ids": list(f["leaf_ids"]),
            "objective": float(f["objective"]), "objective_q": np.asarray(f["objective_q"]).tolist()}


def recipe_fit(transform: cr.Transform, b2: dict, state: dict) -> cr.CorrectionFit:
    if state["map"].n_leaves < 2 or state["fit"]["kind"] != "joint":
        return cr.CorrectionFit(transform, b2["beta"], b2["objective"], "b2_reuse")
    f = state["fit"]
    return cr.CorrectionFit(transform, b2["beta"], b2["objective"], "joint", np.asarray(f["beta"]), list(f["leaf_ids"]),
                            np.asarray(f["u"]), float(f["objective"]))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Fraction):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# --------------------------------------------------------------------------- freeze map


def run_freeze_map(inputs: Inputs, contract: dict, ws: Workspace) -> dict:
    ws.require("develop")
    out = ws.stage("maps")
    if (out / "MAP_FREEZE.json").exists():
        raise ContractError("maps are frozen; never relearned")
    if (ws.run_dir / "predict").exists():
        raise ContractError("predict outputs exist before the map freeze")
    c = contract["correction"]
    freeze = parse_month(contract["calendar"]["map_freeze_cutoff"])
    record = {"stage": "map_freeze", "cutoff": contract["calendar"]["map_freeze_cutoff"], "horizons": {}}
    for h, hi in sorted(inputs.horizons.items()):
        sel = read_json(ws.run_dir / "develop" / f"selection_h{h:02d}.json")
        prov = pd.read_csv(ws.run_dir / "develop" / f"providers_h{h:02d}.csv.gz", float_precision="round_trip")
        pool = correction_pool(hi, prov, freeze, contract) if sel["selected"] != "B2" else None
        if sel["selected"] == "B2":
            rec = {"selected": "B2", "map": None, "fallback": "no eligible recipe in development (D13)"}
        elif len(pool["rows"]) == 0:
            rec = {"selected": sel["selected"], "map": None, "fallback": "empty final-map pool: exact B2 (never reopen selection)"}
        else:
            if (hi.target[pool["rows"]] > freeze).any():
                raise ContractError("final map pool contains a post-cutoff target")
            transform, ppool = make_pool(hi, pool)
            search = pt.Search(ppool, inputs.D_admin, inputs.admin_codes, transform.active(), contract, float(sel["lambda_region"]),
                               inputs.descriptor_names)
            states = search.run(int(sel["leaf_budget"]), int(c["max_depth"]))
            st = pt.state_at_budget(states, int(sel["leaf_budget"]))
            write_csv_gz(out / f"final_candidates_h{h:02d}.csv.gz", search.ledger_frame())
            write_csv_gz(out / f"final_pool_h{h:02d}.csv.gz", pool_record(hi, pool))
            pmap = st["map"]
            routable_admin = np.isfinite(inputs.D_admin).all(axis=1)
            membership = pd.DataFrame({"admin_code": inputs.admin_codes, "leaf": pmap.route(inputs.D_admin, routable_admin)})
            write_csv_gz(out / f"membership_h{h:02d}.csv.gz", membership)
            rec = {"selected": sel["selected"], "leaf_budget": sel["leaf_budget"], "lambda_region": sel["lambda_region"],
                   "map": pmap.record(), "objective_path": [{"step": s["step"], "objective": s["objective"], "n_leaves": s["map"].n_leaves} for s in states],
                   "fallback": "selected recipe produced no final split: exact B2 (never reopen selection)" if pmap.n_leaves < 2 else None,
                   "membership_sha256": sha256_file(out / f"membership_h{h:02d}.csv.gz")}
        write_json(out / f"final_map_h{h:02d}.json", rec)
        record["horizons"][str(h)] = {**rec, "file_sha256": sha256_file(out / f"final_map_h{h:02d}.json"),
                                      "selection_sha256": sha256_file(ws.run_dir / "develop" / f"selection_h{h:02d}.json")}
    record["statement"] = "maps and recipes frozen before any main/supplement G prediction; 2023+ outcomes never redraw maps"
    record["frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(out / "MAP_FREEZE.json", record)
    ws.complete("maps", {"map_freeze_sha256": sha256_file(out / "MAP_FREEZE.json")})
    return record


def verify_map_freeze(ws: Workspace, horizons) -> dict:
    """Check the recorded chain maps/COMPLETE -> MAP_FREEZE -> final_map / selection / membership before any use."""
    maps, dev = ws.run_dir / "maps", ws.run_dir / "develop"
    done = read_json(maps / "COMPLETE.json")
    if sha256_file(maps / "MAP_FREEZE.json") != done.get("map_freeze_sha256"):
        raise ContractError("MAP_FREEZE.json differs from the hash pinned in maps/COMPLETE.json")
    freeze = read_json(maps / "MAP_FREEZE.json")
    for h in horizons:
        entry = freeze["horizons"].get(str(h))
        if entry is None:
            raise ContractError(f"MAP_FREEZE has no entry for H{h}")
        if sha256_file(maps / f"final_map_h{h:02d}.json") != entry["file_sha256"]:
            raise ContractError(f"final_map_h{h:02d}.json differs from its frozen hash")
        if sha256_file(dev / f"selection_h{h:02d}.json") != entry["selection_sha256"]:
            raise ContractError(f"selection_h{h:02d}.json differs from the selection bound at the freeze")
        if entry.get("membership_sha256") and sha256_file(maps / f"membership_h{h:02d}.csv.gz") != entry["membership_sha256"]:
            raise ContractError(f"membership_h{h:02d}.csv.gz differs from its frozen hash")
    return freeze


def load_frozen_map(ws: Workspace, h: int, descriptor_names: list):
    verify_map_freeze(ws, [h])
    rec = read_json(ws.run_dir / "maps" / f"final_map_h{h:02d}.json")
    if rec["map"] is None:
        return rec, None
    return rec, pt.PartitionMap.from_record(rec["map"], descriptor_names)


# --------------------------------------------------------------------------- predict (bridge + main + supplement)


def run_predict(inputs: Inputs, contract: dict, ws: Workspace, store: backbone.ModelStore, env: dict) -> dict:
    ws.require("maps")
    out = ws.stage("predict")
    if (out / "COMPLETE.json").exists():
        raise ContractError("predict is COMPLETE; outputs are immutable")
    try:
        return _predict(inputs, contract, ws, out, store, env)
    except Exception as error:
        record_incomplete(ws.run_dir, "predict", {"stage": "predict"}, error)
        raise


def _predict(inputs: Inputs, contract: dict, ws: Workspace, out: Path, store, env) -> dict:
    t0 = time.time()
    summary = {}
    for h, hi in sorted(inputs.horizons.items()):
        rec, pmap = load_frozen_map(ws, h, inputs.descriptor_names)
        lam = float(rec["lambda_region"]) if pmap is not None else None
        prov = pd.read_csv(ws.run_dir / "develop" / f"providers_h{h:02d}.csv.gz", float_precision="round_trip")
        blocks = schedule.horizon_blocks(contract, h)
        bridge = [b for b in blocks if b.role == "bridge"]
        for b in bridge:
            r1 = fit_block(hi, b, "B1", store, contract, env, inputs.source, inputs.features424, "bridge_provider")
            prov = pd.concat([prov, _provider_rows(hi, b, r1)], ignore_index=True)
        outputs, contexts = [], []
        for b in sorted([b for b in blocks if b.role in ("main", "supplement")], key=lambda x: x.fit_origin):
            r1 = fit_block(hi, b, "B1", store, contract, env, inputs.source, inputs.features424, f"{b.role}_B1")
            r0 = fit_block(hi, b, "B0", store, contract, env, inputs.source, inputs.features296, f"{b.role}_B0")
            if not np.array_equal(r0["rows"], r1["rows"]):
                raise ContractError(f"{b.block_id}: B0/B1 prediction keys differ")
            pool = correction_pool(hi, prov, b.fit_origin, contract)
            cfit, support = context_fit(hi, pool, pmap, lam, contract)
            rows = r1["rows"]
            leaf_of = leaf_rows(hi, pmap, rows) if pmap is not None else np.full(len(rows), None, dtype=object)
            pred = cfit.predict(r1["raw"], hi.Z[rows], leaf_of)
            frame = keyed_outputs(hi, rows, {"B0": r0["raw"], "B1": r1["raw"], "B2": pred["b2"], "G": pred["g"]}, b)
            frame["G_route"] = pred["route"]
            frame["B1_identity"], frame["B0_identity"] = r1["identity"], r0["identity"]
            outputs.append(frame)
            ctx = f"h{h:02d}_{b.role}_{b.target_lo // 12}"
            write_csv_gz(out / f"pool_{ctx}.csv.gz", pool_record(hi, pool))
            contexts.append({"context": ctx, "block": b.record(), "fit_origin": month_label(b.fit_origin), "pool_rows": int(len(pool["rows"])),
                             "correction": cfit.record(), "support": support, "map_frozen": pmap is not None})
            prov = pd.concat([prov, _provider_rows(hi, b, r1)], ignore_index=True)
        write_csv_gz(out / f"predictions_h{h:02d}.csv.gz", pd.concat(outputs, ignore_index=True))
        write_csv_gz(out / f"providers_all_h{h:02d}.csv.gz", prov)
        write_json(out / f"contexts_h{h:02d}.json", contexts)
        summary[str(h)] = {"rows": int(sum(len(o) for o in outputs)), "contexts": len(contexts)}
    ledger = pd.DataFrame(store.ledger)
    write_csv_gz(out / "request_ledger.csv.gz", ledger)
    result = {"stage": "predict", "summary": summary, "elapsed_seconds": round(time.time() - t0, 1)}
    ws.complete("predict", result)
    return result


def _provider_rows(hi: HorizonInputs, b: schedule.Block, r1: dict) -> pd.DataFrame:
    return pd.DataFrame({"row": r1["rows"], "target_ord": hi.target[r1["rows"]], "provider_block": b.block_id,
                         "provider_fit_origin": b.fit_origin, "provider_identity": r1["identity"],
                         **{f"p_{q}": r1["raw"][:, k] for k, q in enumerate(TARGETS)}})


def context_fit(hi: HorizonInputs, pool: dict, pmap, lam, contract: dict):
    """Annual B2/G fit for one context; zero eligible OOT rows is the declared identity correction (B2 = B1, G = B2)."""
    if len(pool["rows"]) == 0:
        return cr.CorrectionFit.identity(), {"reason": "identity_empty_pool"}
    transform, ppool = make_pool(hi, pool)
    return annual_fit(hi, pool, transform, ppool, pmap, lam, contract)


def annual_fit(hi: HorizonInputs, pool: dict, transform: cr.Transform, ppool: pt.Pool, pmap, lam, contract: dict):
    c = contract["correction"]
    if pmap is None or pmap.n_leaves < 2:
        b2 = cr.solve_b2(ppool.total, transform.active(), float(c["shared_lambda"]), float(c["solver_residual_tolerance"]))
        return cr.CorrectionFit(transform, b2["beta"], b2["objective"], "b2_reuse"), {"reason": "no frozen partition (B2)"}
    leaf_of = pmap.route(hi.D[pool["rows"]], pool["routable"])
    fitres = pt.fit_frozen_map(ppool, transform.active(), pmap, leaf_of, contract, lam)
    b2 = fitres["b2"]
    if fitres["g_kind"] == "b2_reuse":
        return cr.CorrectionFit(transform, b2["beta"], b2["objective"], "b2_reuse",
                                unsupported_leaves=[l for l in pmap.leaves if l not in fitres["supported"]]), fitres["support"]
    return cr.CorrectionFit(transform, b2["beta"], b2["objective"], "joint", fitres["beta_g"], fitres["supported"], fitres["u"],
                            fitres["g_objective"], [l for l in pmap.leaves if l not in fitres["supported"]]), fitres["support"]


def keyed_outputs(hi: HorizonInputs, rows: np.ndarray, raws: dict, block: schedule.Block) -> pd.DataFrame:
    keep = ["admin_code", "country_key", "target_ord", "target_month", "origin_month", "phase_truth", "q2", "q3", "q4", "q5",
            "persistence_available", "persistence_phase", "persistence_q3", "persistence_source_month", "persistence_age_months"]
    frame = hi.keys.iloc[rows][[c for c in keep if c in hi.keys.columns]].reset_index(drop=True)
    frame.insert(0, "row", rows)
    frame["period"] = "supplement" if block.role == "supplement" else "main"
    frame["block_id"] = block.block_id
    frame["routable"] = hi.routable[rows].astype(np.int64)
    for arm, raw in raws.items():
        star, phase = project_arm(raw)
        for k, q in enumerate(TARGETS):
            frame[f"{arm}_{q}_raw"] = raw[:, k]
            frame[f"{arm}_{q}_star"] = star[:, k]
        frame[f"{arm}_phase"] = phase
    return frame
