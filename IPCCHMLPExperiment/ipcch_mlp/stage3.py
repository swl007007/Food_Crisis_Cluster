"""P2: rolling Stage3 with frozen maps, MLP historical gate and ungated regional diagnostic.

Adapted from ipcch_geoxgb/stage3.py at 6798df2 (``run_fold``, ``gate_decision``):
same calendar (O = T - H, window [O-35, O]), latest six observed U < O with
V = U - H, pooled confusion counts, validation support plus >= 3 successful
supported regional dates, strict exact gain > 1/100, atomic quartet routing,
empty folds ledger-only, empty required global pool stops. Changes (PRD R8-R13,
R21): the reference and fallback is P = B + pooled residual; the regional side
is L = B + regional residual on the same frozen B; every mapped region with
current keys and current fitting support gets L predictions (diagnostic),
whatever its gate; G uses L only where the gate passes, otherwise P.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_mlp import metrics, preprocess, projection, sources
from ipcch_mlp.artifacts import sha256_file, write_json
from ipcch_mlp.errors import TechnicalError
from ipcch_mlp.quartets import TARGETS, Engine

GZ = {"method": "gzip", "mtime": 0}
GAIN = Fraction(1, 100)
ARMS = ("base", "pool", "local", "geo")


def gate_decision(pairs: pd.DataFrame, contract: dict) -> dict:
    """Validation support + strict exact gain of L-routed over P on pooled historical keys."""
    floor = contract["support"]["validation"]
    n = int(len(pairs))
    crisis = int((pairs["phase_truth"] >= 3).sum()) if n else 0
    dated = pairs.groupby("validation_month")["local_fit_ok"].any() if n else pd.Series(dtype=bool)
    record = {"keys": n, "areas": int(pairs["admin_code"].nunique()) if n else 0, "target_months": int(len(dated)),
              "crisis_keys": crisis, "noncrisis_keys": n - crisis, "local_fit_dates": int(dated.sum()) if n else 0}
    short = [k for k, v in floor.items() if record[k] < v]
    if record["local_fit_dates"] < contract["support"]["stage3_min_successful_local_dates"]:
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


def _decode(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return projection.project_and_decode(raw)


class Stage3:
    def __init__(self, engine: Engine, contract: dict, hz: sources.Horizon, replicate: int, recipe: str):
        self.e, self.c, self.hz, self.rep = engine, contract, hz, replicate
        self.gid, self.rid = recipe[:2], recipe[2:]
        self.win = contract["calendar"]["rolling_window_calendar_months"]
        self.batch = contract["training"]["inference_batch_size"]

    def _global(self, origin: int, use: dict):
        gf = self.e.global_fit(self.hz, "stage3", self.rep, self.gid, origin, self.hz.window_rows(origin, self.win), use)
        P = self.e.residual_fit(self.hz, gf, self.rid, None, use)
        return gf, P

    def _regional_supported(self, origin: int, node: str) -> tuple[bool, dict]:
        rows = self.hz.region_rows(self.hz.window_rows(origin, self.win), node)
        sup = sources.support(self.hz.keys.iloc[rows])
        return sources.meets(sup, self.c["support"]["local_fit"]), sup

    def run_fold(self, fold: dict) -> dict:
        hz, h = self.hz, self.hz.h
        target = int(fold["target_ord"])
        origin = target - h
        if int(fold["origin_ord"]) != origin:
            raise TechnicalError("fold origin is not T - H")
        eval_rows = hz.rows_at(target)
        ledger = {"fold_id": fold["fold_id"], "period": fold["period"], "H": h, "replicate": self.rep,
                  "target_ord": target, "origin_ord": origin, "eval_keys": int(len(eval_rows))}
        if len(eval_rows) == 0:
            return {"ledger": {**ledger, "status": "no_valid_target"}, "predictions": None, "gate": [], "pairs": None}
        tag = {"stage": "stage3", "H": h, "replicate": self.rep, "fold": fold["fold_id"]}
        gf, P = self._global(origin, {**tag, "use": "current", "fitting_origin": origin})
        Xe = self.e.inputs(hz, gf, eval_rows)
        B_e = gf.B.predict(Xe, self.e.device, self.batch)
        P_e = P.predict(Xe, self.e.device, self.batch)
        n = len(eval_rows)
        L_e = np.full((n, 4), np.nan)
        node = hz.node[eval_rows]
        route = np.where(node == "", "unmapped_area_pool", "pool_fallback:pending").astype(object)
        l_provider = np.full(n, "", dtype=object)
        gate_records, pair_frames = [], []
        dates = sources.historical_gate_dates(hz.observed_months, origin, self.c["calendar"]["historical_gate_max_dates"])
        for u in dates:
            v = int(u) - h
            gv, Pv = self._global(v, {**tag, "use": "gate", "gate_month": int(u), "fitting_origin": v})
            u_rows = hz.rows_at(int(u))
            Xu = self.e.inputs(hz, gv, u_rows)
            Bu = gv.B.predict(Xu, self.e.device, self.batch)
            Pu = Pv.predict(Xu, self.e.device, self.batch)
            pool_raw = Bu + Pu
            pool_star, pool_phase = _decode(pool_raw)
            for region in hz.regions:
                inr = np.isin(hz.area[u_rows], hz.regions[region])
                if not inr.any():
                    continue
                val = u_rows[inr]
                ok, sup = self._regional_supported(v, region)
                if ok:
                    Lq = self.e.residual_fit(hz, gv, self.rid, region,
                                             {**tag, "use": "gate", "gate_month": int(u), "fitting_origin": v})
                    Lu = Lq.predict(Xu[inr], self.e.device, self.batch)
                    routed_raw = Bu[inr] + Lu
                    provider = Lq.provider()
                else:  # support fallback: the local-routed side IS that date's P
                    Lu = np.full((int(inr.sum()), 4), np.nan)
                    routed_raw = pool_raw[inr]
                    provider = ""
                r_star, r_phase = _decode(routed_raw)
                frame = {"region": region, "admin_code": hz.area[val], "row": val, "validation_month": int(u),
                         "internal_origin": v, "phase_truth": hz.keys["phase_truth"].to_numpy()[val],
                         "phase_pool": pool_phase[inr], "phase_local_routed": r_phase, "local_fit_ok": ok,
                         "base_provider": gv.B.provider(), "pres_provider": Pv.provider(), "lres_provider": provider,
                         "local_fit_keys": sup["keys"], "local_fit_areas": sup["areas"],
                         "local_fit_months": sup["target_months"]}
                for j, q in enumerate(TARGETS):
                    frame[f"base_{q}"] = Bu[inr, j]
                    frame[f"pres_{q}"] = Pu[inr, j]
                    frame[f"lres_{q}"] = Lu[:, j]
                pair_frames.append(pd.DataFrame(frame))
        pairs = (pd.concat(pair_frames, ignore_index=True) if pair_frames else
                 pd.DataFrame(columns=["region", "admin_code", "validation_month", "phase_truth", "phase_pool",
                                       "phase_local_routed", "local_fit_ok"]))
        for region in hz.regions:
            rows_te = node == region
            decision = {"region": region, "areas_in_map": int(len(hz.regions[region])), "test_keys": int(rows_te.sum()),
                        **gate_decision(pairs[pairs["region"] == region], self.c)}
            if rows_te.any():
                ok, sup = self._regional_supported(origin, region)
                decision["current_fit_support"] = sup
                decision["current_fit_supported"] = bool(ok)
                if ok:  # ungated diagnostic: fit whatever the gate says (R21)
                    Lq = self.e.residual_fit(hz, gf, self.rid, region, {**tag, "use": "current", "fitting_origin": origin})
                    L_e[rows_te] = Lq.predict(Xe[rows_te], self.e.device, self.batch)
                    l_provider[rows_te] = Lq.provider()
                    decision["local_provider"] = Lq.provider()
                if decision["enabled"] and ok:
                    route[rows_te] = "local"
                    decision["route"] = "local"
                elif decision["enabled"]:
                    route[rows_te] = "pool_fallback:current_fit_support"
                    decision["route"] = "pool_fallback"
                else:
                    route[rows_te] = f"pool_fallback:{decision['reason']}"
                    decision["route"] = "pool_fallback"
            else:
                decision["route"] = "no_current_keys"
            gate_records.append(decision)
        if (route == "pool_fallback:pending").any():
            raise TechnicalError("a mapped key was left without a route")
        k = hz.keys.iloc[eval_rows]
        base_star, base_phase = _decode(B_e)
        pool_raw = B_e + P_e
        pool_star, pool_phase = _decode(pool_raw)
        has_l = ~np.isnan(L_e).any(axis=1)
        local_raw = np.where(has_l[:, None], B_e + np.nan_to_num(L_e), np.nan)
        local_star = np.full((n, 4), np.nan)
        local_phase = np.zeros(n, dtype=np.int64)
        if has_l.any():
            local_star[has_l], local_phase[has_l] = _decode(local_raw[has_l])
        use_l = route == "local"
        geo_raw = np.where(use_l[:, None], local_raw, pool_raw)
        geo_star = np.where(use_l[:, None], local_star, pool_star)
        geo_phase = np.where(use_l, local_phase, pool_phase)
        pred = pd.DataFrame({
            "row": eval_rows, "admin_code": hz.area[eval_rows], "target_month": k["target_month"].to_numpy(),
            "target_ord": target, "horizon_months": h, "origin_ord": origin, "fold_id": fold["fold_id"],
            "period": fold["period"], "replicate": self.rep, "country_key": k["country_key"].to_numpy(),
            "phase_truth": k["phase_truth"].to_numpy(), "crisis_truth": k["crisis_truth"].to_numpy(),
            "q3_truth": k["q3"].to_numpy(), "region": node, "route": route, "local_eligible": has_l.astype(int),
            "base_provider": gf.B.provider(), "pres_provider": P.provider(), "lres_provider": l_provider,
        })
        for j, q in enumerate(TARGETS):
            pred[f"base_{q}_raw"], pred[f"pres_{q}"], pred[f"lres_{q}"] = B_e[:, j], P_e[:, j], L_e[:, j]
        for arm, raw, star in (("base", B_e, base_star), ("pool", pool_raw, pool_star),
                               ("local", local_raw, local_star), ("geo", geo_raw, geo_star)):
            for j, q in enumerate(TARGETS):
                if arm != "base":
                    pred[f"{arm}_{q}_raw"] = raw[:, j]
                pred[f"{arm}_{q}_star"] = star[:, j]
        pred["base_phase"], pred["pool_phase"], pred["local_phase"], pred["geo_phase"] = (
            base_phase, pool_phase, local_phase, geo_phase)
        for col in ("persistence_available", "persistence_phase", "persistence_q3", "persistence_source_month",
                    "persistence_age_months"):
            pred[col] = k[col].to_numpy()
        ledger.update(status="scored", gate_dates=[int(u) for u in dates], transform_sha256=gf.transform_sha256,
                      base_provider=gf.B.provider(), pres_provider=P.provider(),
                      local_routed_regions=int(sum(1 for d in gate_records if d.get("route") == "local")),
                      unseen_missingness_rows=preprocess.unseen_missingness(gf.transform, hz.raw(eval_rows))["rows_affected"])
        return {"ledger": ledger, "predictions": pred, "gate": gate_records, "pairs": pairs}


def run_stage3(run_dir: Path, engine: Engine, contract: dict, horizons: dict, calendar: pd.DataFrame,
               winners: dict) -> dict:
    out = run_dir / "stage3"
    out.mkdir()
    summary = {"stage": "P2-stage3", "recipes": winners, "replicates": {}}
    for rep in contract["replicates"]:
        rdir = out / f"rep{rep}"
        rdir.mkdir()
        fits_before = engine.store.counts["fits"]
        rsum = {}
        for h, hz in horizons.items():
            s3 = Stage3(engine, contract, hz, rep, winners[str(h)])
            hdir = rdir / f"h{h:02d}"
            hdir.mkdir()
            preds, ledger = [], []
            folds = calendar[calendar["horizon_months"] == h].sort_values(["target_ord", "period"])
            with open(hdir / "gate_decisions.jsonl", "w", encoding="utf-8", newline="\n") as glog:
                for fold in folds.to_dict("records"):
                    res = s3.run_fold(fold)
                    ledger.append(res["ledger"])
                    if res["predictions"] is not None:
                        preds.append(res["predictions"])
                    for d in res["gate"]:
                        glog.write(json.dumps({"fold_id": fold["fold_id"], **d}, sort_keys=True, default=str) + "\n")
                    if res["pairs"] is not None and len(res["pairs"]):
                        res["pairs"].to_csv(hdir / f"pairs_{fold['fold_id']}.csv.gz", index=False, compression=GZ)
            frame = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
            frame.to_csv(hdir / "predictions.csv.gz", index=False, compression=GZ)
            pd.DataFrame(ledger).to_csv(hdir / "fold_ledger.csv", index=False)
            rsum[str(h)] = {"recipe": winners[str(h)], "folds": len(ledger),
                            "scored_folds": sum(1 for r in ledger if r["status"] == "scored"),
                            "prediction_rows": int(len(frame)),
                            "local_rows": int((frame["route"] == "local").sum()) if len(frame) else 0,
                            "local_eligible_rows": int(frame["local_eligible"].sum()) if len(frame) else 0,
                            "predictions_sha256": sha256_file(hdir / "predictions.csv.gz")}
        rsum["new_scalar_fits"] = engine.store.counts["fits"] - fits_before
        summary["replicates"][str(rep)] = rsum
    summary["store_counts"] = dict(engine.store.counts)
    write_json(out / "stage3-summary.json", summary)
    return summary
