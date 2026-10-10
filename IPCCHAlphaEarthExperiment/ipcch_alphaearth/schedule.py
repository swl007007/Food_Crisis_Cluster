"""Annual block calendar, unique B1 OOT providers and the no-fit request enumeration (design section 6, D3-D7, D27).

Every block fits at its earliest *scheduled* target minus H (independent of observed labels). Blocks never
use an origin before 2018-01. Each H/admin/target key with a B1 prediction has exactly one provider block:
historical blocks through target 2022, the bridge for pre-main 2023 targets, then the formal main/supplement
B1 blocks. Correction pools at fit origin O are provider rows with target U <= O (D29 inclusive), weighted
2**(-(O - U)/24). Nothing here fits a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ipcch_alphaearth.errors import ContractError
from ipcch_alphaearth.months import month_label, parse_month


@dataclass(frozen=True)
class Block:
    block_id: str
    horizon: int
    role: str  # historical | bridge | main | supplement
    target_lo: int
    target_hi: int
    fit_origin: int
    arms: tuple  # ("B1",) or ("B1", "B0")
    development_year: int | None

    def record(self) -> dict:
        out = asdict(self)
        out["arms"] = list(self.arms)
        out.update(target_lo_month=month_label(self.target_lo), target_hi_month=month_label(self.target_hi),
                   fit_origin_month=month_label(self.fit_origin), slots=self.target_hi - self.target_lo + 1)
        return out


def decay_weights(target_ord, fit_origin: int, half_life: float) -> np.ndarray:
    """Unnormalized D7 weights ``2**(-(O - U)/half_life)``; a label after the origin is a contract error."""
    age = int(fit_origin) - np.asarray(target_ord, dtype=np.int64)
    if (age < 0).any():
        raise ContractError("training label after the fit origin")
    return np.power(2.0, -age.astype(np.float64) / float(half_life))


def n_eff(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    s2 = float(np.sum(w * w))
    return float(np.sum(w)) ** 2 / s2 if s2 > 0 else 0.0


def horizon_blocks(contract: dict, horizon: int) -> list[Block]:
    cal = contract["calendar"]
    h = int(horizon)
    first_origin = parse_month(cal["first_origin"])
    main_first_origin = parse_month(cal["main_first_origin"])
    main_first = parse_month(cal["main_first_target"][str(h)])
    main_last = parse_month(cal["main_last_target"])
    dev_years = set(cal["development_target_years"])
    freeze = parse_month(cal["map_freeze_cutoff"])
    supp = [parse_month(m) for m in cal["supplement_target_months"]]
    blocks: list[Block] = []
    first_target = first_origin + h
    for year in range(first_target // 12, freeze // 12 + 1):
        lo = max(year * 12, first_target)
        hi = min(year * 12 + 11, freeze)
        if lo > hi:
            continue
        dev = year if year in dev_years else None
        blocks.append(Block(f"h{h:02d}_hist_{year}", h, "historical", lo, hi, lo - h, ("B1", "B0") if dev else ("B1",), dev))
    bridge_lo, bridge_hi = freeze + 1, main_first - 1
    if bridge_lo <= bridge_hi:
        if bridge_hi // 12 != bridge_lo // 12:
            raise ContractError(f"H{h}: bridge spans more than one target year")
        blocks.append(Block(f"h{h:02d}_bridge_{bridge_lo // 12}", h, "bridge", bridge_lo, bridge_hi, bridge_lo - h, ("B1",), None))
    for year in range(main_first // 12, main_last // 12 + 1):
        lo = max(year * 12, main_first)
        hi = min(year * 12 + 11, main_last)
        if lo <= hi:
            blocks.append(Block(f"h{h:02d}_main_{year}", h, "main", lo, hi, lo - h, ("B1", "B0"), None))
    if supp:
        if supp != list(range(supp[0], supp[-1] + 1)):
            raise ContractError("supplement months must be contiguous")
        blocks.append(Block(f"h{h:02d}_supp_{supp[0] // 12}", h, "supplement", supp[0], supp[-1], supp[0] - h, ("B1", "B0"), None))
    # structural checks
    for b in blocks:
        if b.fit_origin < first_origin:
            raise ContractError(f"{b.block_id}: fit origin {month_label(b.fit_origin)} precedes {cal['first_origin']}")
        if b.role in ("main", "supplement") and b.fit_origin < main_first_origin:
            raise ContractError(f"{b.block_id}: main fit origin precedes {cal['main_first_origin']}")
        if b.role in ("historical", "bridge") and b.target_lo - h >= main_first_origin:
            raise ContractError(f"{b.block_id}: pre-main block has a main-period origin")
    ranges = sorted((b.target_lo, b.target_hi) for b in blocks)
    for (a_lo, a_hi), (b_lo, _b_hi) in zip(ranges, ranges[1:]):
        if b_lo <= a_hi:
            raise ContractError(f"H{h}: overlapping block target ranges (non-unique providers)")
    return blocks


def all_blocks(contract: dict) -> list[Block]:
    out = [b for h in contract["calendar"]["horizons_months"] for b in horizon_blocks(contract, h)]
    check_fit_ceiling(contract, out)
    return out


def check_fit_ceiling(contract: dict, blocks: list[Block]) -> dict:
    cal = contract["calendar"]
    b1 = sum("B1" in b.arms for b in blocks)
    b0 = sum("B0" in b.arms for b in blocks)
    bridge_slots = sum(b.target_hi - b.target_lo + 1 for b in blocks if b.role == "bridge")
    counts = {"B1_quartets": b1, "B0_quartets": b0, "quartets": b1 + b0, "scalar_fits": 4 * (b1 + b0)}
    if counts != cal["fit_ceiling"]:
        raise ContractError(f"calendar fit enumeration {counts} differs from the frozen ceiling {cal['fit_ceiling']}")
    if bridge_slots != cal["bridge_slots_total"]:
        raise ContractError(f"bridge has {bridge_slots} H/month slots, contract says {cal['bridge_slots_total']}")
    return {**counts, "bridge_slots": bridge_slots}


def fold_calendar(contract: dict, keys_by_h: dict | None = None) -> pd.DataFrame:
    """122 main + supplement H x target-month folds; empty folds kept with status ``no_valid_target``."""
    cal = contract["calendar"]
    main_last = parse_month(cal["main_last_target"])
    rows = []
    for h in cal["horizons_months"]:
        for t in range(parse_month(cal["main_first_target"][str(h)]), main_last + 1):
            rows.append(("main", h, t))
        for m in cal["supplement_target_months"]:
            rows.append(("supplement", h, parse_month(m)))
    frame = pd.DataFrame(rows, columns=["period", "horizon", "target_ord"])
    frame["origin_ord"] = frame["target_ord"] - frame["horizon"]
    frame["target_month"] = month_label(frame["target_ord"].to_numpy())
    frame["origin_month"] = month_label(frame["origin_ord"].to_numpy())
    main = frame[frame["period"] == "main"]
    if len(main) != cal["main_fold_total"]:
        raise ContractError(f"main calendar has {len(main)} folds, contract says {cal['main_fold_total']}")
    for h, n in cal["main_fold_counts"].items():
        if int((main["horizon"] == int(h)).sum()) != n:
            raise ContractError(f"H{h} main fold count differs from {n}")
    if (main["origin_ord"] < parse_month(cal["main_first_origin"])).any():
        raise ContractError("a main fold origin precedes the main boundary")
    if keys_by_h is not None:
        counts = []
        for row in frame.itertuples(index=False):
            k = keys_by_h[row.horizon]
            counts.append(int((k["target_ord"].to_numpy() == row.target_ord).sum()))
        frame["eval_keys"] = counts
        frame["status"] = np.where(frame["eval_keys"] > 0, "scheduled", "no_valid_target")
    return frame


def assign_providers(keys: pd.DataFrame, blocks: list[Block]) -> pd.DataFrame:
    """One B1 provider block per key (or none for targets outside every block, e.g. pre-2018 origins)."""
    t = keys["target_ord"].to_numpy(dtype=np.int64)
    provider = np.full(len(keys), "", dtype=object)
    fit_origin = np.full(len(keys), -1, dtype=np.int64)
    role = np.full(len(keys), "", dtype=object)
    hits = np.zeros(len(keys), dtype=np.int64)
    for b in blocks:
        mask = (t >= b.target_lo) & (t <= b.target_hi)
        provider[mask] = b.block_id
        fit_origin[mask] = b.fit_origin
        role[mask] = b.role
        hits += mask
    if (hits > 1).any():
        raise ContractError("a key has more than one provider block")
    has = hits == 1
    if (fit_origin[has] >= t[has]).any():
        raise ContractError("a provider's fit origin is not before its predicted target")
    return pd.DataFrame({"provider_block": provider, "provider_fit_origin": fit_origin, "provider_role": role,
                         "has_provider": has.astype(np.int64)})


def pool_summary(target_ord: np.ndarray, admin: np.ndarray, mask: np.ndarray, fit_origin: int, half_life: float) -> dict:
    t = target_ord[mask]
    if len(t) == 0:
        return {"rows": 0, "admins": 0, "target_months": 0, "sum_w": 0.0, "n_eff": 0.0, "first_target": "", "last_target": "",
                "largest_month_weight_share": None, "empty": True}
    w = decay_weights(t, fit_origin, half_life)
    month_w = pd.Series(w).groupby(t).sum()
    return {"rows": int(len(t)), "admins": int(np.unique(admin[mask]).size), "target_months": int(np.unique(t).size),
            "sum_w": float(w.sum()), "n_eff": n_eff(w), "first_target": month_label(int(t.min())),
            "last_target": month_label(int(t.max())), "largest_month_weight_share": float(month_w.max() / w.sum()),
            "empty": False}


def correction_contexts(contract: dict, blocks: list[Block]) -> list[dict]:
    """Every correction/map fit: development (each dev block origin), final map (freeze cutoff), main/supp blocks."""
    freeze = parse_month(contract["calendar"]["map_freeze_cutoff"])
    out = []
    for h in contract["calendar"]["horizons_months"]:
        hb = [b for b in blocks if b.horizon == h]
        for b in hb:
            if b.development_year is not None:
                out.append({"context": f"h{h:02d}_dev_{b.development_year}", "horizon": h, "kind": "development",
                            "fit_origin": b.fit_origin, "evaluate_block": b.block_id})
        out.append({"context": f"h{h:02d}_final_map", "horizon": h, "kind": "final_map", "fit_origin": freeze, "evaluate_block": None})
        for b in hb:
            if b.role in ("main", "supplement"):
                out.append({"context": f"h{h:02d}_{b.role}_{b.target_lo // 12}", "horizon": h, "kind": b.role,
                            "fit_origin": b.fit_origin, "evaluate_block": b.block_id})
    return out


def enumerate_requests(contract: dict, keys_by_h: dict) -> dict:
    """No-fit enumeration: backbone requests with actual pools, providers, correction pools and fold support."""
    blocks = all_blocks(contract)
    half_life = contract["calendar"]["half_life_months"]
    requests, providers, corrections = [], {}, []
    for h in contract["calendar"]["horizons_months"]:
        keys = keys_by_h[h]
        t = keys["target_ord"].to_numpy(dtype=np.int64)
        admin = keys["admin_code"].to_numpy(dtype=np.int64)
        hb = [b for b in blocks if b.horizon == h]
        prov = assign_providers(keys, hb)
        providers[h] = prov
        for b in hb:
            fit_mask = t <= b.fit_origin
            pred_mask = (t >= b.target_lo) & (t <= b.target_hi)
            pool = pool_summary(t, admin, fit_mask, b.fit_origin, half_life)
            for arm in b.arms:
                requests.append({**b.record(), "arm": arm, "request_id": f"{arm}_{b.block_id}", "scalar_fits": 4,
                                 "predict_rows": int(pred_mask.sum()), "predict_admins": int(np.unique(admin[pred_mask]).size),
                                 **{f"fit_{k}": v for k, v in pool.items()}})
        has = prov["has_provider"].to_numpy() == 1
        for ctx in [c for c in correction_contexts(contract, blocks) if c["horizon"] == h]:
            mask = has & (t <= ctx["fit_origin"])
            pool = pool_summary(t, admin, mask, ctx["fit_origin"], half_life)
            roles = prov.loc[mask, "provider_role"].value_counts().to_dict()
            corrections.append({**ctx, "fit_origin_month": month_label(ctx["fit_origin"]), "provider_roles": roles,
                                **{f"pool_{k}": v for k, v in pool.items()}})
    req = pd.DataFrame(requests)
    empty = req[req["fit_empty"]]
    summary = check_fit_ceiling(contract, blocks)
    summary.update(empty_backbone_pools=empty["request_id"].tolist(),
                   empty_correction_pools=[c["context"] for c in corrections if c["pool_empty"]])
    return {"blocks": [b.record() for b in blocks], "requests": req, "providers": providers,
            "corrections": pd.DataFrame(corrections), "summary": summary}
