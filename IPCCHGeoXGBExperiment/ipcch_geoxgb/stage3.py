"""Stage3: rolling folds with frozen maps, R37 historical gate replay and paired cohorts.

Adapted from ``FEWSNETGeoXGBExperiment/src/experiment/stage3.py`` (``gate_decision``,
``run_fold``/``_run_arm`` date-major historical replay, shared-root local fits,
global fallback when local support fails) with the accepted IPCCH contract:
H in {1,3,6,12}; inclusive 36-month target window [O-35, O] (R23); up to six
latest observed target months U < O from the complete valid ledger, V = U - H
(R37); pooled confusion counts; R28/R29 support plus >= 3 successful supported
local dates; strict exact gain > 0.01 (R16); quartet atomic routing; unmapped
areas and maps without an accepted split route to the same-fold global (R36,
clarification 2); empty current folds are ledger-only (R47); an empty required
global pool stops the run (R40); technical errors stop (R41).

Model identities bind the verified prepared artifacts (X/keys SHA256 from the
manifest) plus the ordered fitting row positions and (area, month) keys, the
window, H, recipe/params/seed, schema, availability, environment and code; a
local identity also binds the region members and its global quartet. Gate
decisions are recomputed at every O and never cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pandas as pd

from ipcch_geoxgb import metrics, projection, quartet, schedule
from ipcch_geoxgb.errors import ContractError, TechnicalError
from ipcch_geoxgb.modelstore import ModelStore, array_digest, target_digests
from ipcch_geoxgb.stage1 import meets, support

STAGE3_GAIN = Fraction(1, 100)


@dataclass
class HorizonContext:
    h: int
    keys: pd.DataFrame  # one row per valid key at this H, aligned with X rows
    X: np.ndarray
    artifact_sha: dict  # {"X": ..., "keys": ...} from the verified prepared manifest
    region_of: dict  # area -> terminal node id (learned areas only)
    local_enabled: bool  # the frozen map has at least one accepted split
    gid: str
    lid: str
    contract: dict
    store: ModelStore
    base_identity: dict
    observed_months: np.ndarray  # all distinct valid target months (all areas)
    rows_by_month: dict = field(default_factory=dict)
    regions: dict = field(default_factory=dict)

    def __post_init__(self):
        months = self.keys["target_ord"].to_numpy()
        order = np.argsort(months, kind="mergesort")
        for m in np.unique(months):
            self.rows_by_month[int(m)] = order[months[order] == m]
        nodes = {}
        for area, node in self.region_of.items():
            nodes.setdefault(node, []).append(int(area))
        self.regions = {n: np.array(sorted(a), dtype=np.int64) for n, a in sorted(nodes.items())}
        self.gparams, self.grounds = quartet.global_params(self.contract, self.gid)
        self.lparams, self.lrounds = quartet.local_params(self.contract, self.lid)
        self.Y = self.keys[list(quartet.TARGETS)].to_numpy(dtype=np.float64)
        self.area = self.keys["admin_code"].to_numpy(dtype=np.int64)

    def rows_at(self, month: int) -> np.ndarray:
        return self.rows_by_month.get(int(month), np.zeros(0, dtype=np.int64))

    def window_rows(self, origin: int) -> np.ndarray:
        lo, hi = schedule.training_window(origin, self.contract["calendar"]["rolling_window_calendar_months"])
        parts = [self.rows_at(m) for m in range(lo, hi + 1)]
        return np.sort(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)

    def _identity(self, scope: str, origin: int, rows: np.ndarray) -> dict:
        lo, hi = schedule.training_window(origin, self.contract["calendar"]["rolling_window_calendar_months"])
        keys = self.keys.iloc[rows][["admin_code", "target_ord"]].to_numpy(dtype=np.int64)
        return {
            **self.base_identity,
            "scope": scope,
            "H": self.h,
            "targets": list(quartet.TARGETS),
            "fitting_origin": origin,
            "window": [lo, hi],
            "fit_rows": array_digest(np.asarray(rows, dtype=np.int64)),
            "n_rows": int(len(rows)),
            "y_sha256": target_digests(self.Y[rows]),
            "fit_keys": array_digest(keys),
            "X_artifact_sha256": self.artifact_sha["X"],
            "keys_artifact_sha256": self.artifact_sha["keys"],
        }

    def global_quartet(self, origin: int, purpose: dict) -> tuple[str, quartet.Quartet, dict]:
        rows = self.window_rows(origin)
        if len(rows) == 0:
            raise ContractError(
                f"R40 stop: empty required global pool at H{self.h}, fitting origin {origin} (incomplete run)"
            )
        identity = {**self._identity("stage3-global", origin, rows), "G": self.gid, "params": self.gparams,
                    "rounds": self.grounds}
        q, use = self.store.get_or_fit(
            identity,
            lambda: quartet.fit_global_quartet(self.X[rows], self.Y[rows], self.gparams, self.grounds),
            {"stage": "stage3", "H": self.h, "fitting_origin": origin, "purpose": "global", **purpose},
        )
        info = {**support(self.keys.iloc[rows]), "constant_targets": {t: q.records[t]["constant"] for t in quartet.TARGETS}}
        return use["identity_sha256"], q, info

    def local_quartet(self, origin: int, node: str, global_ref: tuple[str, quartet.Quartet], purpose: dict):
        """Region quartet on the region's window rows, or (None, support) if unsupported."""
        window = self.window_rows(origin)
        rows = window[np.isin(self.area[window], self.regions[node])]
        sup = support(self.keys.iloc[rows])
        if not meets(sup, self.contract["support"]["local_fit"]):
            return None, sup
        identity = {**self._identity("stage3-local", origin, rows), "G": self.gid, "L": self.lid,
                    "params": self.lparams, "rounds": self.lrounds,
                    "region_node": node, "region_areas": array_digest(self.regions[node]),
                    "global_identity": global_ref[0], "global_boosters": global_ref[1].booster_shas()}
        q, use = self.store.get_or_fit(
            identity,
            lambda: quartet.continue_local_quartet(global_ref[1], self.X[rows], self.Y[rows], self.lparams, self.lrounds),
            {"stage": "stage3", "H": self.h, "fitting_origin": origin, "region": node, "purpose": "local", **purpose},
        )
        return (use["identity_sha256"], q), sup


def predict(q: quartet.Quartet, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = q.predict_raw(X)
    q_star, phase = projection.project_and_decode(raw)
    return raw, q_star, phase


def gate_decision(pairs: pd.DataFrame, contract: dict) -> dict:
    """R28/R29/R37/R16 on one region's pooled historical validation keys."""
    floor = contract["support"]["validation"]
    n = int(len(pairs))
    crisis = int((pairs["phase_truth"] >= 3).sum()) if n else 0
    dated = pairs.groupby("validation_month")["local_fit_ok"].any() if n else pd.Series(dtype=bool)
    record = {
        "keys": n,
        "areas": int(pairs["admin_code"].nunique()) if n else 0,
        "target_months": int(len(dated)),
        "crisis_keys": crisis,
        "noncrisis_keys": n - crisis,
        "local_fit_dates": int(dated.sum()) if n else 0,
    }
    short = [k for k, v in floor.items() if record[k] < v]
    if record["local_fit_dates"] < contract["support"]["stage3_min_successful_local_dates"]:
        short.append("local_fit_dates")
    counts_g = metrics.crisis_counts(pairs["phase_truth"], pairs["phase_global"]) if n else None
    counts_l = metrics.crisis_counts(pairs["phase_truth"], pairs["phase_local_routed"]) if n else None
    if n:
        f_g, f_l = metrics.exact_f1(counts_g), metrics.exact_f1(counts_l)
        record.update(counts_global=counts_g, counts_local=counts_l,
                      f1_global=None if f_g is None else str(f_g), f1_local=None if f_l is None else str(f_l))
    if short:
        return {**record, "enabled": False, "reason": "gate_support:" + "+".join(short)}
    passed, why = metrics.gain_passes(counts_l, counts_g, STAGE3_GAIN)
    return {**record, "enabled": bool(passed), "reason": "" if passed else why}


def run_fold(ctx: HorizonContext, fold: dict) -> dict:
    """One (H, T) fold: lawful gate replay at O = T - H, current fits, keyed predictions."""
    h = ctx.h
    target = int(fold["target_ord"])
    origin = target - h
    if int(fold["origin_ord"]) != origin:
        raise TechnicalError("fold origin is not T - H")
    eval_rows = ctx.rows_at(target)
    ledger = {"fold_id": fold["fold_id"], "period": fold["period"], "H": h, "target_ord": target,
              "origin_ord": origin, "eval_keys": int(len(eval_rows))}
    if len(eval_rows) == 0:
        return {"ledger": {**ledger, "status": "no_valid_target"}, "predictions": None, "gate": [], "pairs": None}

    tag = {"fold": fold["fold_id"]}
    g_ref_digest, g_cur, g_info = ctx.global_quartet(origin, {**tag, "use": "current"})
    raw_g, star_g, phase_g = predict(g_cur, ctx.X[eval_rows])
    raw, star, phase = raw_g.copy(), star_g.copy(), phase_g.copy()
    area = ctx.area[eval_rows]
    node = np.array([ctx.region_of.get(int(a), "") for a in area], dtype=object)
    route = np.where(node == "", "unmapped_area_global", "global_only_no_accepted_split").astype(object)
    provider = np.full(len(eval_rows), g_ref_digest, dtype=object)
    gate_records, pair_frames = [], []
    dates = schedule.historical_gate_dates(ctx.observed_months, origin, ctx.contract["calendar"]["historical_gate_max_dates"])

    if ctx.local_enabled:
        for u in dates:  # date-major: each historical global is loaded once
            v = int(u) - h
            gv_digest, gv, _ = ctx.global_quartet(v, {**tag, "use": "gate", "gate_month": int(u)})
            u_rows = ctx.rows_at(u)
            _, _, phase_u = predict(gv, ctx.X[u_rows])
            for region in ctx.regions:
                in_region = np.isin(ctx.area[u_rows], ctx.regions[region])
                if not in_region.any():
                    continue
                val = u_rows[in_region]
                try:
                    local_ref, sup = ctx.local_quartet(v, region, (gv_digest, gv), {**tag, "use": "gate", "gate_month": int(u)})
                except Exception as error:
                    error.add_note(f"stage3 H{h} {fold['fold_id']} gate month {int(u)} region {region}")
                    raise
                if local_ref is not None:
                    _, _, phase_l = predict(local_ref[1], ctx.X[val])
                    ok, local_digest = True, local_ref[0]
                else:
                    phase_l, ok, local_digest = phase_u[in_region], False, ""
                pair_frames.append(pd.DataFrame({
                    "region": region, "admin_code": ctx.area[val], "validation_month": int(u),
                    "internal_origin": v, "phase_truth": ctx.keys["phase_truth"].to_numpy()[val],
                    "phase_global": phase_u[in_region], "phase_local_routed": phase_l,
                    "local_fit_ok": ok, "global_identity": gv_digest, "local_identity": local_digest,
                    "local_fit_keys": sup["keys"], "local_fit_areas": sup["areas"],
                    "local_fit_months": sup["target_months"],
                }))
        pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame(
            columns=["region", "admin_code", "validation_month", "phase_truth", "phase_global",
                     "phase_local_routed", "local_fit_ok"])
        for region in ctx.regions:
            rows_te = node == region
            decision = {"region": region, "areas_in_map": int(len(ctx.regions[region])), "test_keys": int(rows_te.sum()),
                        **gate_decision(pairs[pairs["region"] == region], ctx.contract)}
            if not rows_te.any():
                decision.update(route="no_current_keys")
            elif decision["enabled"]:
                try:
                    local_ref, sup = ctx.local_quartet(origin, region, (g_ref_digest, g_cur), {**tag, "use": "current"})
                except Exception as error:
                    error.add_note(f"stage3 H{h} {fold['fold_id']} current region {region}")
                    raise
                decision["current_fit_support"] = sup
                if local_ref is not None:
                    r, s, p = predict(local_ref[1], ctx.X[eval_rows[rows_te]])
                    raw[rows_te], star[rows_te], phase[rows_te] = r, s, p
                    route[rows_te], provider[rows_te] = "local", local_ref[0]
                    decision.update(route="local", local_identity=local_ref[0])
                else:
                    route[rows_te] = "global_fallback:current_fit_support"
                    decision.update(route="global_fallback", reason="current_fit_support")
            else:
                route[rows_te] = f"global_fallback:{decision['reason']}"
                decision.update(route="global_fallback")
            gate_records.append(decision)
    else:
        pairs = None

    k = ctx.keys.iloc[eval_rows]
    pred = pd.DataFrame({
        "admin_code": area, "target_month": k["target_month"].to_numpy(), "target_ord": target,
        "horizon_months": h, "origin_ord": origin, "fold_id": fold["fold_id"], "period": fold["period"],
        "country_key": k["country_key"].to_numpy(),
        "phase_truth": k["phase_truth"].to_numpy(), "crisis_truth": k["crisis_truth"].to_numpy(),
        "q3_truth": k["q3"].to_numpy(), "region": node, "route": route, "provider": provider,
        "global_identity": g_ref_digest,
    })
    for j, t in enumerate(quartet.TARGETS):
        pred[f"geo_{t}_raw"], pred[f"geo_{t}_star"] = raw[:, j], star[:, j]
        pred[f"pool_{t}_raw"], pred[f"pool_{t}_star"] = raw_g[:, j], star_g[:, j]
    pred["geo_phase"], pred["pool_phase"] = phase, phase_g
    for col in ("persistence_available", "persistence_phase", "persistence_q3",
                "persistence_source_month", "persistence_age_months"):
        pred[col] = k[col].to_numpy()
    ledger.update(status="scored", gate_dates=[int(u) for u in dates], current_global=g_ref_digest,
                  current_global_support=g_info,
                  local_regions=int(sum(1 for d in gate_records if d.get("route") == "local")))
    return {"ledger": ledger, "predictions": pred, "gate": gate_records, "pairs": pairs}
