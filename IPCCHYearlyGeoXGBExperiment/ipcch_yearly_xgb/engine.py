"""Annual fixed-map GeoXGB engine (design sections 2-5, 9).

Per H and current block (period x target year): one weighted global quartet P
at the block origin O on all valid rows with t <= O; for each mapped region
with block evaluation keys and count support at O, a fresh local continuation
L of its own P root (ungated diagnostic). The gate is decided once per region
at O from the latest six globally observed target months U < O, each scored
by the annual historical pair of its own fit origin (fit_origin(H, U)); a
region without historical fitting support keeps its validation keys with the
historical pooled prediction on the local side. G uses L only when the gate
and current support pass; otherwise G equals P exactly. Gate decision logic
is adapted from ipcch_geoxgb/stage3.py ``gate_decision`` at 6798df2 (the
comparator is the matched annual pooled quartet).

Fitting identities bind protocol, fit-source digest, runtime, H, recipe and
parameters, fit origin and annual anchor, exact ordered rows/keys, X/keys
artifact digests, per-target y digests and weight digests; local identities
add region membership, map digest and the parent global booster digests.
Request purpose (current/historical) is ledgered separately.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np
import pandas as pd

from ipcch_yearly_xgb import metrics, projection, quartet, schedule
from ipcch_yearly_xgb.errors import ContractError, TechnicalError
from ipcch_yearly_xgb.modelstore import ModelStore, array_digest, target_digests
from ipcch_yearly_xgb.sources import Horizon, meets, support

TARGETS = quartet.TARGETS
GAIN = Fraction(1, 100)
PROTOCOL_ID = "ipcch-yearly-xgb-protocol-v1.0"


def gate_decision(pairs: pd.DataFrame, contract: dict) -> dict:
    """Validation support + strict exact crisis-F1 gain of L-routed over pooled (unweighted)."""
    floor = contract["support"]["validation"]
    n = int(len(pairs))
    crisis = int((pairs["phase_truth"] >= 3).sum()) if n else 0
    dated = pairs.groupby("validation_month")["local_fit_ok"].any() if n else pd.Series(dtype=bool)
    record = {"keys": n, "areas": int(pairs["admin_code"].nunique()) if n else 0, "target_months": int(len(dated)),
              "crisis_keys": crisis, "noncrisis_keys": n - crisis,
              "local_fit_dates": int(dated.sum()) if n else 0,
              "distinct_local_models": int(pairs.loc[pairs["local_fit_ok"], "local_provider"].nunique()) if n else 0,
              "distinct_global_models": int(pairs["pool_provider"].nunique()) if n else 0}
    short = [k for k, v in floor.items() if record[k] < v]
    if record["local_fit_dates"] < contract["support"]["min_successful_local_dates"]:
        short.append("local_fit_dates")
    if n:
        cp = metrics.crisis_counts(pairs["phase_truth"], pairs["phase_pool"])
        cl = metrics.crisis_counts(pairs["phase_truth"], pairs["phase_local_routed"])
        fp, fl = metrics.exact_f1(cp), metrics.exact_f1(cl)
        record.update(counts_pool=cp, counts_local=cl, f1_pool=None if fp is None else str(fp),
                      f1_local=None if fl is None else str(fl))
    if short:
        return {**record, "historical_support": False, "enabled": False, "reason": "gate_support:" + "+".join(short)}
    passed, why = metrics.gain_passes(record["counts_local"], record["counts_pool"], GAIN)
    return {**record, "historical_support": True, "enabled": bool(passed), "reason": "" if passed else why}


def _decode(raw: np.ndarray):
    return projection.project_and_decode(raw)


class Engine:
    """Model requests for one H (fit or exact reuse through the store)."""

    def __init__(self, hz: Horizon, contract: dict, store: ModelStore, env: dict):
        self.hz, self.c, self.store, self.env = hz, contract, store, env
        self.recipe = contract["recipes"][str(hz.h)]
        self.gparams, self.grounds = quartet.global_params(contract, self.recipe[:2])
        self.lparams, self.lrounds = quartet.local_params(contract, self.recipe[2:])
        self.first = schedule.parse_month(contract["protocol"]["first_main_target"][str(hz.h)])
        self.half_life = contract["protocol"]["half_life_months"]
        self._globals: dict = {}
        self._locals: dict = {}
        self.weight_stats: dict = {}

    def fit_origin(self, u: int) -> int:
        return schedule.fit_origin(self.hz.h, self.first, int(u))

    def _base(self, origin: int, rows: np.ndarray) -> tuple[dict, np.ndarray]:
        hz = self.hz
        w = schedule.decay_weights(hz.t[rows], origin, self.half_life)
        keys = np.column_stack([hz.area[rows], hz.t[rows]]).astype(np.int64)
        ident = {
            "protocol": PROTOCOL_ID, "contract": self.c["contract_version"], "env": self.env, "H": hz.h,
            "recipe": self.recipe, "fit_origin": int(origin), "annual_anchor": int(origin + hz.h),
            "fit_pool": "t<=fit_origin, no lower bound",
            "fit_rows": array_digest(np.asarray(rows, dtype=np.int64)), "n_rows": int(len(rows)),
            "fit_keys": array_digest(keys), "y_sha256": target_digests(hz.Y[rows]),
            "X_artifact_sha256": hz.x_sha256, "keys_artifact_sha256": hz.keys_sha256,
            "schema": hz.lineage.get("schema"), "prepared_manifest_sha256": hz.lineage.get("prepared_manifest_sha256"),
            "weights": {"formula": "0.5**((O-t)/24)", "half_life_months": self.half_life,
                        "protocol_sha256": hashlib.sha256(np.ascontiguousarray(w, np.float64).tobytes()).hexdigest(),
                        "effective_float32_sha256": hashlib.sha256(
                            np.ascontiguousarray(w.astype(np.float32)).tobytes()).hexdigest()},
        }
        return ident, w

    def global_quartet(self, origin: int, use: dict):
        hz = self.hz
        rows = hz.pool_rows(origin)
        if len(rows) == 0:
            raise ContractError(f"empty required global pool: H{hz.h} origin {schedule.month_label(origin)}")
        ident, w = self._base(origin, rows)
        ident = {**ident, "scope": "yearly-global", "G": self.recipe[:2], "params": self.gparams,
                 "rounds": self.grounds}
        q, entry = self.store.get_or_fit(
            ident, lambda: quartet.fit_global_quartet(np.asarray(hz.X[rows]), hz.Y[rows], w, self.gparams, self.grounds),
            {"H": hz.h, "role": "global", "fit_origin": int(origin), **use})
        self.weight_stats[entry["identity_sha256"]] = {"n": int(len(rows)), "sum": float(w.sum()),
                                                       "ess": float(w.sum() ** 2 / (w ** 2).sum())}
        return entry["identity_sha256"], q, rows, w

    def local_support(self, origin: int, node: str) -> tuple[bool, dict]:
        rows = self.hz.pool_rows(origin)
        sub = rows[self.hz.region_mask(rows, node)]
        sup = support(self.hz.keys.iloc[sub])
        return meets(sup, self.c["support"]["local_fit"]), sup

    def local_quartet(self, origin: int, node: str, gref: tuple, use: dict):
        """Fresh continuation of the matching global root on the region's exact subset and weights."""
        hz = self.hz
        g_digest, gq, grows, gw = gref
        mask = hz.region_mask(grows, node)
        rows, w = grows[mask], gw[mask]
        ok, sup = self.local_support(origin, node)
        if not ok:
            return None, sup
        ident, w_check = self._base(origin, rows)
        if not np.array_equal(w_check, w):
            raise TechnicalError("regional weights differ from the indexed global weights")
        ident = {**ident, "scope": "yearly-local", "G": self.recipe[:2], "L": self.recipe[2:],
                 "params": self.lparams, "rounds": self.lrounds, "region_node": node,
                 "region_areas": array_digest(hz.regions[node]), "map_sha256": hz.map_sha256,
                 "global_identity": g_digest, "global_boosters": gq.booster_shas()}
        q, entry = self.store.get_or_fit(
            ident, lambda: quartet.continue_local_quartet(gq, np.asarray(hz.X[rows]), hz.Y[rows], w, self.lparams,
                                                          self.lrounds),
            {"H": hz.h, "role": "local", "fit_origin": int(origin), "region": node, **use})
        self.weight_stats[entry["identity_sha256"]] = {"n": int(len(rows)), "sum": float(w.sum()),
                                                       "ess": float(w.sum() ** 2 / (w ** 2).sum())}
        return (entry["identity_sha256"], q), sup

    def _gref(self, origin: int, use: dict):
        if origin not in self._globals:
            self._globals[origin] = self.global_quartet(origin, use)
        else:
            self.store._log({"identity_sha256": self._globals[origin][0], "status": "memory_reuse", "H": self.hz.h,
                             "role": "global", "fit_origin": int(origin), **use})
            self.store.counts["requests"] += 1
            self.store.counts["hits"] += 1
        return self._globals[origin]

    def _lref(self, origin: int, node: str, use: dict):
        key = (origin, node)
        if key not in self._locals:
            self._locals[key] = self.local_quartet(origin, node, self._gref(origin, use), use)
        else:
            ref, _ = self._locals[key]
            if ref is not None:
                self.store._log({"identity_sha256": ref[0], "status": "memory_reuse", "H": self.hz.h, "role": "local",
                                 "fit_origin": int(origin), "region": node, **use})
                self.store.counts["requests"] += 1
                self.store.counts["hits"] += 1
        return self._locals[key]

    # ------------------------------------------------------------------ one current block

    def run_block(self, block: schedule.Block) -> dict:
        hz, c = self.hz, self.c
        eval_rows = np.concatenate([hz.rows_at(int(f["target_ord"])) for f in block.folds]) \
            if block.folds else np.zeros(0, dtype=np.int64)
        eval_rows = np.sort(eval_rows)
        O = block.origin
        info = {"block_id": block.block_id, "H": hz.h, "period": block.period, "year": block.year,
                "anchor_ord": block.anchor, "anchor": schedule.month_label(block.anchor), "fit_origin_ord": O,
                "fit_origin": schedule.month_label(O), "eval_keys": int(len(eval_rows)),
                "folds": [f["fold_id"] for f in block.folds]}
        if len(eval_rows) == 0:
            return {"block": {**info, "status": "no_valid_target"}, "predictions": None, "gates": [], "pairs": None}
        tag = {"block": block.block_id}
        # ---- historical gate replay (frozen at O)
        dates = schedule.gate_dates(hz.observed_months, O, c["protocol"]["historical_gate_max_dates"])
        frames = []
        for u in dates:
            v = self.fit_origin(int(u))
            if not v < int(u) < O:
                raise TechnicalError(f"historical origin {v} not before validation month {u} before O {O}")
            use = {**tag, "use": "gate", "gate_month": int(u)}
            g_digest, gq, _, _ = self._gref(v, use)
            u_rows = hz.rows_at(int(u))
            raw_p = gq.predict_raw(np.asarray(hz.X[u_rows]))
            star_p, phase_p = _decode(raw_p)
            for node in hz.regions:
                inr = hz.region_mask(u_rows, node)
                if not inr.any():
                    continue
                val = u_rows[inr]
                lref, sup = self._lref(v, node, use)
                if lref is not None:
                    raw_l = lref[1].predict_raw(np.asarray(hz.X[val]))
                    ok, lprov = True, lref[0]
                else:  # support fallback: the local side IS the historical pooled prediction
                    raw_l, ok, lprov = raw_p[inr], False, ""
                star_l, phase_l = _decode(raw_l)
                fr = {"region": node, "row": val, "admin_code": hz.area[val], "validation_month": int(u),
                      "historical_fit_origin": v, "phase_truth": hz.keys["phase_truth"].to_numpy()[val],
                      "phase_pool": phase_p[inr], "phase_local_routed": phase_l, "local_fit_ok": ok,
                      "pool_provider": g_digest, "local_provider": lprov, "local_fit_keys": sup["keys"],
                      "local_fit_areas": sup["areas"], "local_fit_months": sup["target_months"]}
                for j, q in enumerate(TARGETS):
                    fr[f"pool_{q}_raw"] = raw_p[inr, j]
                    fr[f"local_routed_{q}_raw"] = raw_l[:, j]
                frames.append(pd.DataFrame(fr))
        pairs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["region", "row", "admin_code", "validation_month", "phase_truth", "phase_pool",
                     "phase_local_routed", "local_fit_ok", "pool_provider", "local_provider"])
        # ---- current models
        use = {**tag, "use": "current"}
        g_digest, gq, grows, gw = self._gref(O, use)
        X_e = np.asarray(hz.X[eval_rows])
        raw_p = gq.predict_raw(X_e)
        star_p, phase_p = _decode(raw_p)
        n = len(eval_rows)
        raw_l = np.full((n, 4), np.nan)
        node = hz.node[eval_rows]
        route = np.where(node == "", "unmapped_area_pool", "pending").astype(object)
        l_prov = np.full(n, "", dtype=object)
        gates = []
        for region in hz.regions:
            rte = node == region
            d = {"block_id": block.block_id, "region": region, "fit_origin_ord": O,
                 "gate_dates": [int(u) for u in dates], "test_keys": int(rte.sum()),
                 **gate_decision(pairs[pairs["region"] == region], c)}
            if rte.any():
                lref, sup = self._lref(O, region, use)
                d["current_fit_support"], d["current_fit_supported"] = sup, lref is not None
                if lref is not None:  # ungated diagnostic, whatever the gate says
                    raw_l[rte] = lref[1].predict_raw(X_e[rte])
                    l_prov[rte] = lref[0]
                    d["local_provider"] = lref[0]
                if d["enabled"] and lref is not None:
                    route[rte], d["route"] = "local", "local"
                elif d["enabled"]:
                    route[rte], d["route"] = "pool_fallback:current_fit_support", "pool_fallback"
                else:
                    route[rte], d["route"] = f"pool_fallback:{d['reason']}", "pool_fallback"
            else:
                d["route"] = "no_current_keys"
            gates.append(d)
        if (route == "pending").any():
            raise TechnicalError("a mapped key was left without a route")
        has_l = ~np.isnan(raw_l).any(axis=1)
        star_l = np.full((n, 4), np.nan)
        phase_l = np.zeros(n, dtype=np.int64)
        if has_l.any():
            star_l[has_l], phase_l[has_l] = _decode(raw_l[has_l])
        use_l = route == "local"
        raw_g = np.where(use_l[:, None], raw_l, raw_p)
        star_g = np.where(use_l[:, None], star_l, star_p)
        phase_g = np.where(use_l, phase_l, phase_p)
        k = hz.keys.iloc[eval_rows]
        fold_of = {int(f["target_ord"]): (f["fold_id"], f["period"]) for f in block.folds}
        pred = pd.DataFrame({
            "row": eval_rows, "admin_code": hz.area[eval_rows], "target_month": k["target_month"].to_numpy(),
            "target_ord": hz.t[eval_rows], "horizon_months": hz.h,
            "fold_id": [fold_of[int(t)][0] for t in hz.t[eval_rows]], "period": block.period,
            "block_id": block.block_id, "annual_anchor_ord": block.anchor, "fit_origin_ord": O,
            "row_origin_ord": k["origin_ord"].to_numpy(), "country_key": k["country_key"].to_numpy(),
            "phase_truth": k["phase_truth"].to_numpy(), "crisis_truth": k["crisis_truth"].to_numpy(),
            **{f"{q}_truth": k[q].to_numpy() for q in TARGETS},
            "region": node, "route": route, "local_eligible": has_l.astype(int),
            "pool_provider": g_digest, "local_provider": l_prov})
        for arm, raw, star in (("pool", raw_p, star_p), ("local", raw_l, star_l), ("geo", raw_g, star_g)):
            for j, q in enumerate(TARGETS):
                pred[f"{arm}_{q}_raw"], pred[f"{arm}_{q}_star"] = raw[:, j], star[:, j]
        pred["pool_phase"], pred["local_phase"], pred["geo_phase"] = phase_p, phase_l, phase_g
        for col in ("persistence_available", "persistence_phase", "persistence_q3", "persistence_source_month",
                    "persistence_age_months"):
            pred[col] = k[col].to_numpy()
        info.update(status="scored", gate_dates=[int(u) for u in dates],
                    historical_origins=sorted({self.fit_origin(int(u)) for u in dates}),
                    pool_provider=g_digest, global_fit_rows=int(len(grows)), weight_sum=float(gw.sum()),
                    weight_ess=float(gw.sum() ** 2 / (gw ** 2).sum()),
                    adopted_regions=int(sum(1 for d in gates if d["route"] == "local")),
                    diagnostic_keys=int(has_l.sum()))
        return {"block": info, "predictions": pred, "gates": gates, "pairs": pairs}
