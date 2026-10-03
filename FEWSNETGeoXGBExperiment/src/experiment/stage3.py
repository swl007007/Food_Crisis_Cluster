"""Stage 3 engine shared by G screening, development folds and the final evaluation.

One external fold = (H, target T, origin O = T - H). Every booster is fitted at its own
origin on labels in [origin-59, origin): the external global at O, and, for the
historical gate, one global per internal origin V = U - H for the six most recent
globally observed label months U < O (plan section 5, D13, D18).

Arms (plan section 6):
  pooled       the global booster of the fold (P-XGB)
  shared       map regions; local = continue the SAME global (G prefix frozen) with L
  independent  map regions; local = fresh G on the region's own rows, then L

A region's local route is enabled only if, on its pooled gate validation keys, the
local-routed predictions beat the global predictions by a strict exact macro-F1 gain
> .01 with the gate support floors met; a date whose local fit fails keeps its rows
with the global prediction. A current local fit failing its support floor falls back
to the global even when the gate passed. Unmapped areas use the global.

Interruption task (design D2/G2/G4): a ``ScenarioPanel`` replaces the frozen snapshot with
release-aware views for one availability scenario k and training strategy A/B. Its fitting
pools are the lawful (weighted, grouped) views at each origin, support counts original keys,
internal gate fits and gate rows inherit the outer forecast's hidden cycles, target rows are a
prediction cohort that needs no label, and the gate metric is crisis F1 (undefined -> not
enabled). The frozen ``Panel`` keeps its behaviour; both implement the same small interface.
"""
from __future__ import annotations

import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment import plan
from src.metrics import fourclass
from src.model import native_xgb as nx

LOCAL = "local_model"


def mi(label: str) -> int:
    year, month = str(label).split("-")
    return int(year) * 12 + int(month) - 1


def ml(index: int) -> str:
    return f"{int(index) // 12:04d}-{int(index) % 12 + 1:02d}"


class Panel:
    """One horizon's origin-aligned snapshot as dense arrays (features already frozen)."""

    def __init__(self, path, features, horizon):
        snap = pd.read_parquet(path)
        if not (snap["horizon"] == horizon).all() or list(snap.columns[-len(features):]) != list(features):
            raise ValueError("snapshot horizon or feature order mismatch")
        snap = snap.sort_values(["area", "target_month"]).reset_index(drop=True)
        self.horizon = horizon
        self.features = list(features)
        self.X = nx.clean(snap[self.features].to_numpy(dtype=float))
        self.y = snap["class_code"].to_numpy(dtype=np.int64)
        self.area = snap["area"].to_numpy(dtype=np.int64)
        self.month = snap["target_month"].to_numpy(dtype=np.int64)
        self.country = snap["country"].to_numpy()
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    weight = None          # the frozen snapshot fits unweighted rows
    gate_metric = "macro"  # D13 as frozen

    def fit_pool(self, origin: int, excluded=frozenset()):
        if excluded:
            raise ValueError("the frozen snapshot has no release masks")
        return self, self.window(origin)

    def target_rows(self, target: int):
        rows = self.at(target)
        if len(rows) == 0:
            raise ValueError(f"{ml(target)}: no labelled target rows")
        return self, rows

    def gate_rows(self, month: int, outer_origin: int, excluded=frozenset()):
        return self, self.at(month)

    def outer_exclusions(self, origin: int) -> frozenset:
        return frozenset()

    def gate_months(self, origin: int):
        return None        # the prepared schedule's gate dates

    def store_identity(self, pool, rows, excluded, origin) -> dict:
        return {}

    def window(self, origin: int) -> np.ndarray:
        return np.where((self.month >= origin - plan.WINDOW) & (self.month < origin))[0]

    def at(self, month: int) -> np.ndarray:
        return np.where(self.month == month)[0]

    def support(self, rows) -> dict:
        return nx.support(self.y[rows], self.area[rows], self.month[rows])

    def keys_sha(self, rows) -> str:
        return nx.keys_sha(self.area[rows], self.month[rows])


class FitPool:
    """One lawful scenario fitting pool: variant rows grouped by original key (design G4)."""

    def __init__(self, frame: pd.DataFrame, features):
        self.X = nx.clean(frame[list(features)].to_numpy(dtype=float))
        self.y = frame["class_code"].to_numpy(dtype=np.int64)
        self.area = frame["area"].to_numpy(dtype=np.int64)
        self.month = frame["target_month"].to_numpy(dtype=np.int64)
        self.key = frame["orig_key"].to_numpy(dtype=np.int64)
        self.variant = frame["variant_k"].to_numpy(dtype=np.int64)
        w = frame["weight"].to_numpy(dtype=float)
        self.weight = None if np.all(w == 1.0) else w   # strategy A fits unweighted, as before

    def support(self, rows) -> dict:
        """Support of the ORIGINAL keys among ``rows``; variant copies never add support."""
        rows = np.asarray(rows, dtype=np.int64)
        _, first = np.unique(self.key[rows], return_index=True)
        orig = rows[first]
        return nx.support(self.y[orig], self.area[orig], self.month[orig])

    def keys_sha(self, rows) -> str:
        keys = np.column_stack([self.area[rows], self.month[rows], self.variant[rows]]).astype(np.int64)
        return hashlib.sha256(np.ascontiguousarray(keys).tobytes()).hexdigest()


class EvalRows:
    """Prediction or gate rows of a scenario: features, optional truth (NaN = none), metadata."""

    def __init__(self, view: pd.DataFrame, features, truth, meta_columns=()):
        self.X = nx.clean(view[list(features)].to_numpy(dtype=float))
        self.area = view["area"].to_numpy(dtype=np.int64)
        self.month = view["target_month"].to_numpy(dtype=np.int64)
        self.y = np.asarray(truth)
        self.meta = view[list(meta_columns)].reset_index(drop=True) if meta_columns else None


class ScenarioPanel:
    """Release-aware replacement of ``Panel`` for one horizon, scenario k and strategy A/B.

    Built on ``availability.Availability``; implements the engine interface (fit_pool,
    target_rows, gate_rows, outer_exclusions, gate_months, store_identity, gate_metric)."""

    gate_metric = "crisis"
    META = ("country", "scenario_k", "persistence_class_code", "persistence_source_month", "persistence_age")

    def __init__(self, availability, k: int, strategy: str, prediction_areas=None):
        self.av = availability
        self.k = int(k)
        self.strategy = strategy
        self.horizon = availability.horizon
        self.features = list(availability.features)
        areas = availability.scaffold.areas if prediction_areas is None else prediction_areas
        self.prediction_areas = np.array(sorted(int(a) for a in areas), dtype=np.int64)
        led = availability.ledger
        self.identity = {
            "kind": "availability_scenario", "scenario_k": self.k, "strategy": strategy, "horizon": self.horizon,
            "due_rule": led.due_rule,
            "ledger_sha256": hashlib.sha256(led.frame[["cycle_id", "product", "country", "reference_month",
                                                      "release_date", "evidence"]].to_csv(index=False).encode()).hexdigest(),
            "alignment_sha256": hashlib.sha256(json.dumps(availability.alignment, sort_keys=True,
                                                          default=str).encode()).hexdigest(),
            "features_sha256": hashlib.sha256(json.dumps(self.features).encode()).hexdigest(),
            "availability_inputs_sha256": availability.inputs_sha256}
        self.sha256 = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self._pools = {}

    def outer_exclusions(self, origin: int) -> frozenset:
        return self.av.ledger.hidden(origin, self.k)

    def gate_months(self, origin: int) -> list:
        return self.av.gate_dates(origin, self.k)

    def fit_pool(self, origin: int, excluded=frozenset()):
        key = (int(origin), frozenset(excluded))
        if key not in self._pools:
            frame, _ = self.av.fitting_view(origin, self.k, self.strategy, excluded)
            self._pools[key] = FitPool(frame, self.features)
        pool = self._pools[key]
        return pool, np.arange(len(pool.y))

    def store_identity(self, pool, rows, excluded, origin) -> dict:
        w = None if pool.weight is None else pool.weight[rows]
        masked = frozenset(excluded) | self.av.ledger.hidden(origin, self.k)
        return {"scenario_sha256": self.sha256, "scenario_k": self.k, "strategy": self.strategy,
                "excluded_months": [ml(m) for m in sorted(excluded)],
                "masked_months": [ml(m) for m in sorted(masked)],
                "features_sha256": hashlib.sha256(np.ascontiguousarray(pool.X[rows]).tobytes()).hexdigest(),
                "weights_sha256": None if w is None else hashlib.sha256(w.astype(np.float64).tobytes()).hexdigest(),
                "labels_sha256": hashlib.sha256(np.ascontiguousarray(pool.y[rows], dtype=np.int64).tobytes()).hexdigest(),
                "original_keys": pool.support(rows)["rows"]}

    def target_rows(self, target: int):
        """Prediction cohort at T: every prediction area, labelled or not. Truth comes only from
        explicitly supplied evaluator truth (NaN otherwise, e.g. before the final-truth release)."""
        n = len(self.prediction_areas)
        view = self.av.prediction_view(self.prediction_areas, np.full(n, target), self.k)
        truth = self.av.truth_view(self.prediction_areas, np.full(n, target))["truth_code"].to_numpy()
        return EvalRows(view, self.features, truth, self.META), np.arange(n)

    def gate_rows(self, month: int, outer_origin: int, excluded=frozenset()):
        """Gate rows at U: keys whose truth is lawful at the outer cutoff (evaluator-only), with
        features of the internal forecast at V = U - H under the same k and inherited exclusions."""
        lawful = self.av.lawful_labels(month, outer_origin, excluded)
        areas = lawful["area"].to_numpy(dtype=np.int64)
        view = self.av.prediction_view(areas, np.full(len(areas), month), self.k, excluded)
        truth = lawful["class_code"].to_numpy(dtype=np.int64)
        return EvalRows(view, self.features, truth), np.arange(len(areas))


class GlobalStore:
    """Exact-identity cache of global boosters: (H, origin, G) on the panel's window keys.

    A stored booster is reused only if its record matches the horizon, origin, G
    parameters, snapshot hash and the window's fitting-key digest; fits are deterministic,
    so concurrent writers of one key produce identical bytes (atomic replace). The in-memory
    memo and, for scenario panels, the stored path are keyed by the full identity digest, so
    one store never mixes strategies, scenarios, outer masks, features or weights."""

    def __init__(self, root):
        self.root = Path(root)
        self.memo = {}

    def path(self, horizon, origin, g):
        return self.root / f"h{horizon}" / g / f"O{ml(origin)}"

    def get(self, panel, origin: int, g: str, fit_if_missing=True, excluded=frozenset()):
        pool, rows = panel.fit_pool(origin, excluded)
        # "g_config_params" is the REQUESTED G configuration (identity); the fit record's
        # "params" are the resolved xgb.train parameters (provenance). They used to share one
        # key, so the fit record overwrote the identity field and every disk reopen failed.
        identity = {"horizon": panel.horizon, "origin_month": ml(origin), "g_config": g,
                    "g_config_params": plan.G_CONFIGS[g], "snapshot_sha256": panel.sha256,
                    "fit_keys_sha256": pool.keys_sha(rows),
                    "fit_label_months": [ml(origin - plan.WINDOW), ml(origin - 1)]}
        extra = panel.store_identity(pool, rows, excluded, origin)
        identity.update(extra)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        key = (panel.horizon, origin, g, digest)
        if key in self.memo:
            return self.memo[key]
        base = self.path(panel.horizon, origin, g)
        if extra:
            base = base.with_name(f"{base.name}_{digest[:16]}")
        if base.with_suffix(".json").is_file():
            record = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
            if any(record.get(k) != v for k, v in json.loads(json.dumps(identity, default=str)).items()):
                raise RuntimeError(f"stored global {base} does not match its identity")
            payload = base.with_suffix(".ubj").read_bytes()
            if hashlib.sha256(payload).hexdigest() != record["booster_sha256"]:
                raise RuntimeError(f"stored global {base} bytes changed")
            booster = nx.from_raw(payload)
        else:
            if not fit_if_missing:
                raise RuntimeError(f"global {base} is not in the store")
            if len(rows) == 0:
                raise RuntimeError(f"empty global fitting pool at {ml(origin)}")
            weight = None if pool.weight is None else pool.weight[rows]
            booster, fit = nx.fit_global(pool.X[rows], pool.y[rows], plan.G_CONFIGS[g], sample_weight=weight)
            record = {**fit, **identity, "fit_support": pool.support(rows)}
            if record["params"] != fit["params"] or set(identity) & set(fit):
                raise RuntimeError("identity and fit record fields collide")
            base.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(base.with_suffix(".ubj"), nx.raw(booster))
            _atomic_write(base.with_suffix(".json"), json.dumps(record, indent=1, default=str).encode())
        self.memo[key] = (booster, record)
        return booster, record


def _atomic_write(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    try:
        os.replace(tmp, path)
    except PermissionError:
        # Windows refuses to replace a file another worker holds open. The fits are
        # deterministic, so an existing destination must already carry these exact bytes.
        tmp.unlink()
        if not path.is_file() or path.read_bytes() != payload:
            raise


def fit_local(arm, global_booster, panel, rows, g, local):
    """(booster, record) for one region: shared continues the global; independent
    fits a fresh G on the region rows and then appends L to it. ``panel`` is any fitting
    pool (frozen panel or scenario pool); its row weights, if any, reach every fit."""
    X, y = panel.X[rows], panel.y[rows]
    w = None if getattr(panel, "weight", None) is None else panel.weight[rows]
    if arm == "shared":
        parent, parent_record = global_booster, None
    elif arm == "independent":
        parent, parent_record = nx.fit_global(X, y, plan.G_CONFIGS[g], sample_weight=w)
    else:
        raise ValueError(arm)
    booster, record = nx.continue_booster(parent, X, y, plan.L_CONFIGS[local], sample_weight=w)
    record.update(arm=arm, fit_keys_sha256=panel.keys_sha(rows), fit_support=panel.support(rows),
                  independent_prefix=parent_record)
    return booster, record


def exact_f1(y, pred) -> Fraction:
    return fourclass.macro_f1_exact(np.asarray(y, dtype=np.int64), np.asarray(pred, dtype=np.int64))


def crisis_f1_or_none(y, pred):
    """Exact crisis F1 with the reporting convention: None (undefined) when 2TP + FP + FN = 0."""
    c = fourclass.crisis_counts(np.asarray(y, dtype=np.int64), np.asarray(pred, dtype=np.int64))
    denominator = 2 * c["tp"] + c["fp"] + c["fn"]
    return Fraction(2 * c["tp"], denominator) if denominator else None


def gate_decision(pairs: pd.DataFrame, metric: str = "macro") -> dict:
    """D13 on one region's pooled gate keys: support floors, then strict gain > .01.

    ``metric`` "macro" (frozen) or "crisis" (interruption task: crisis F1, and an undefined
    crisis F1 on either side cannot enable the local)."""
    floor = plan.STAGE3_GATE_SUPPORT
    rows = int(len(pairs))
    dated = pairs.groupby("validation_month")["local_fit_ok"].any() if rows else pd.Series(dtype=bool)
    record = {"rows": rows, "areas": int(pairs["area"].nunique()) if rows else 0,
              "dates": int(len(dated)), "local_fit_dates": int(dated.sum()) if rows else 0}
    undefined = False
    if rows and metric == "macro":
        f_global = exact_f1(pairs["y_true"], pairs["y_global"])
        f_local = exact_f1(pairs["y_true"], pairs["y_local_routed"])
        record.update(macro_f1_global=float(f_global), macro_f1_local=float(f_local),
                      gain=str(f_local - f_global), gain_float=float(f_local - f_global))
    elif rows and metric == "crisis":
        f_global = crisis_f1_or_none(pairs["y_true"], pairs["y_global"])
        f_local = crisis_f1_or_none(pairs["y_true"], pairs["y_local_routed"])
        undefined = f_global is None or f_local is None
        record.update(metric="crisis_f1", crisis_f1_global=None if f_global is None else float(f_global),
                      crisis_f1_local=None if f_local is None else float(f_local))
        if not undefined:
            record.update(gain=str(f_local - f_global), gain_float=float(f_local - f_global))
    elif metric not in ("macro", "crisis"):
        raise ValueError(metric)
    short = [k for k, v in floor.items() if record[k] < v]
    if short:
        record.update(enabled=False, reason=f"gate_support:{'+'.join(short)}")
    elif undefined:
        record.update(enabled=False, reason="gate_metric_undefined")
    elif not (Fraction(record["gain"]) > plan.STAGE3_GAIN):
        record.update(enabled=False, reason="gate_gain_not_above_0.01")
    else:
        record.update(enabled=True, reason="")
    return record


def run_fold(panel: Panel, store: GlobalStore, fold: dict, g: str, arms: list, keep_boosters=False):
    """All requested arms of one external fold.

    ``arms``: list of dicts {"arm": "pooled"|"shared"|"independent", "label": str,
    "local": L or None, "route": "learned_map"|"null_consensus"|"no_prior_candidates"|None,
    "cluster_of": {area: cluster} or None, "map_id": str or None}.
    Returns {label: result} with predictions, gate tables/pairs, routes and model records.
    """
    horizon = panel.horizon
    target = mi(fold["target_month"])
    origin = target - horizon
    if mi(fold["origin_month"]) != origin:
        raise ValueError("fold origin is not T - H")
    gate_months = [mi(x["validation_month"]) for x in fold.get("gate", [])]
    lawful = panel.gate_months(origin)
    if lawful is not None:
        if "gate" in fold and gate_months != lawful:
            raise ValueError("fold gate dates differ from the release-aware gate dates")
        gate_months = lawful
    src, test = panel.target_rows(target)
    g_booster, g_record = store.get(panel, origin, g)
    p_global = nx.proba(g_booster, src.X[test])
    out = {}
    for spec in arms:
        out[spec["label"]] = _run_arm(panel, store, spec, g, origin, (src, test), g_booster, g_record,
                                      p_global, gate_months, keep_boosters)
    return out


def _frame(panel, rows, proba, cluster, route, horizon, origin):
    frame = pd.DataFrame({"area": panel.area[rows], "target_month": [ml(m) for m in panel.month[rows]],
                          "origin_month": ml(origin), "horizon": horizon,
                          "y_true_code": panel.y[rows], "y_pred_code": fourclass.argmax_codes(proba),
                          "cluster_id": cluster, "route": route})
    for k, label in enumerate(fourclass.CLASS_LABELS):
        frame[f"p_{label}"] = proba[:, k]
    meta = getattr(panel, "meta", None)   # scenario cohort: persistence source/age, scenario k
    if meta is not None:
        frame = pd.concat([frame, meta.iloc[rows].reset_index(drop=True)], axis=1)
    return frame


def _run_arm(panel, store, spec, g, origin, target, g_booster, g_record, p_global, gate_months, keep):
    horizon = panel.horizon
    src, test = target
    outer = panel.outer_exclusions(origin)
    result = {"spec": {k: v for k, v in spec.items() if k != "cluster_of"},
              "global": {"booster_sha256": g_record["booster_sha256"], "origin_month": ml(origin),
                         "fit_keys_sha256": g_record["fit_keys_sha256"]},
              "gate": [], "gate_pairs": None, "locals": {}, "boosters": {}}
    if spec["arm"] == "pooled" or spec.get("route") != "learned_map":
        reason = "pooled_arm" if spec["arm"] == "pooled" else f"{spec.get('route')}_pooled"
        result["predictions"] = _frame(src, test, p_global, -1, reason, horizon, origin)
        return result
    cluster_of = spec["cluster_of"]
    cluster_test = np.array([cluster_of.get(int(a), -1) for a in src.area[test]])
    clusters = sorted(c for c in set(cluster_test.tolist()) if c >= 0)
    members = {c: np.array(sorted(a for a, k in cluster_of.items() if k == c)) for c in clusters}
    local = spec["local"]
    arm = spec["arm"]

    # Historical gate: date-major so each internal global is loaded once.
    pair_frames = []
    for u in gate_months:
        v = u - horizon
        gv, gv_record = store.get(panel, v, g, excluded=outer)
        ev, rows_u = panel.gate_rows(u, origin, outer)
        p_u = nx.proba(gv, ev.X[rows_u]) if len(rows_u) else np.zeros((0, 4))
        pool, win = panel.fit_pool(v, outer)
        for c in clusters:
            in_c = np.isin(ev.area[rows_u], members[c])
            val = rows_u[in_c]
            if len(val) == 0:
                continue
            fit_rows = win[np.isin(pool.area[win], members[c])]
            sup = pool.support(fit_rows)
            y_global = fourclass.argmax_codes(p_u[in_c])
            if nx.meets(sup, plan.FIT_SUPPORT):
                booster, record = fit_local(arm, gv, pool, fit_rows, g, local)
                y_local = fourclass.argmax_codes(nx.proba(booster, ev.X[val]))
                ok, booster_sha = True, record["booster_sha256"]
            else:
                y_local, ok, booster_sha = y_global, False, None
            pair_frames.append(pd.DataFrame({
                "cluster_id": c, "area": ev.area[val], "validation_month": ml(u),
                "internal_origin": ml(v), "y_true": ev.y[val], "y_global": y_global,
                "y_local_routed": y_local, "local_fit_ok": ok,
                "global_sha256": gv_record["booster_sha256"], "local_sha256": booster_sha,
                "local_fit_rows": sup["rows"], "local_fit_areas": sup["areas"],
                "local_fit_dates": sup["dates"], "local_fit_classes": sup["classes"]}))
    pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame(
        columns=["cluster_id", "area", "validation_month", "internal_origin", "y_true", "y_global",
                 "y_local_routed", "local_fit_ok"])
    result["gate_pairs"] = pairs

    proba = p_global.copy()
    route = np.full(len(test), "unmapped_area_global", dtype=object)
    pool_o, win_o = panel.fit_pool(origin)
    for c in clusters:
        rows_te = cluster_test == c
        decision = {"cluster_id": int(c), "areas_in_map": int(len(members[c])),
                    "test_rows": int(rows_te.sum()),
                    **gate_decision(pairs[pairs["cluster_id"] == c], panel.gate_metric)}
        fit_rows = win_o[np.isin(pool_o.area[win_o], members[c])]
        sup = pool_o.support(fit_rows)
        decision["current_fit_support"] = sup
        if decision["enabled"] and nx.meets(sup, plan.FIT_SUPPORT):
            booster, record = fit_local(arm, g_booster, pool_o, fit_rows, g, local)
            proba[rows_te] = nx.proba(booster, src.X[test[rows_te]])
            route[rows_te] = LOCAL
            decision.update(route=LOCAL, local_sha256=record["booster_sha256"])
            result["locals"][str(c)] = record
            if keep:
                result["boosters"][f"local_{c}"] = nx.raw(booster)
        else:
            why = decision["reason"] if not decision["enabled"] else "current_fit_support"
            route[rows_te] = f"global_fallback:{why}"
            decision.update(route="global_fallback", reason=why)
        result["gate"].append(decision)
    result["predictions"] = _frame(src, test, proba, cluster_test, route, horizon, origin)
    return result


def unmapped_gate(cluster_of: dict, observations: pd.DataFrame, evaluated: pd.DataFrame | None = None) -> dict:
    """Inherited coverage gate: unmapped share of ALL labelled panel rows <= 2%. The share
    among the evaluated target keys is disclosed, never gated on."""
    pct = float(100 * (~observations["area"].isin(list(cluster_of))).mean())
    record = {"gate_population": "all labelled panel rows (release definition)",
              "unmapped_pct_gate": pct, "threshold_pct": 2.0, "passed": pct <= 2.0}
    if evaluated is not None:
        record["unmapped_pct_evaluated_targets_disclosed"] = float(
            100 * (~evaluated["area"].isin(list(cluster_of))).mean())
    return record
