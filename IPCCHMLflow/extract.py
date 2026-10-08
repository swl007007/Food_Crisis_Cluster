"""Explicit report -> evaluation-view extraction for the six known IPCCH source schemas.

Only already-saved values are read; nothing is recomputed from predictions
except cohort key-set digests (sorted admin_code|target_ord), which identify
compatible samples and are checked against the reported cohort sizes.
Undefined/absent scores are recorded as NA entries (path + reason), never as
metrics. Metric namespaces: ``{period}.{cohort}[.{group}].{metric}``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

HORIZONS = ("1", "3", "6", "12")
PERIODS = ("main", "supplementary")
REQUESTED = ("binary.accuracy", "binary.precision", "binary.recall", "binary.f1", "binary.f2",
             "four_class.accuracy", "four_class.macro_f1", "q3_r2_projected", "q3_r2_raw")
P6_PARENT = "p6_geoxgb/p6-formal-20261004b"


class SourceConflict(RuntimeError):
    """Missing or contradictory source content: stop for reconciliation."""


def finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


@dataclass
class Child:
    family: str
    source_run_id: str
    H: str
    arm: str
    seed: str
    arm_kind: str
    metrics: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    na: list = field(default_factory=list)
    panels: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.family}/{self.source_run_id}/H{self.H}/{self.arm}/seed{self.seed}"

    def add(self, name: str, value, src: str) -> None:
        if name in self.metrics and self.metrics[name] != float(value):
            raise SourceConflict(f"{self.key}: metric {name} given two different values")
        if finite(value):
            self.metrics[name] = float(value)
            self.provenance[name] = src
        else:
            self.na.append({"metric": name, "source_path": src, "reason": "null/undefined in source"})

    def note(self, namespace: str, reason: str) -> None:
        self.na.append({"metric": namespace, "source_path": None, "reason": reason})


def _get(d: dict, dotted: str):
    if isinstance(d, dict) and dotted in d:  # flat panels (climate compare-p6) use dotted keys
        return d[dotted]
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def add_panel(c: Child, ns: str, panel, src: str) -> None:
    if not isinstance(panel, dict):
        c.note(ns, "panel absent in source report")
        return
    c.panels[ns] = {"source_path": src, "panel": panel}
    if "n" in panel:
        c.add(f"{ns}.n", panel["n"], f"{src}.n")
    reasons = {**{f"binary.{k}": v for k, v in (_get(panel, "binary.na_reasons") or {}).items()},
               **{f"four_class.{k}": v for k, v in (_get(panel, "four_class.na_reasons") or {}).items()}}
    for k in REQUESTED:
        v = _get(panel, k)
        if finite(v):
            c.add(f"{ns}.{k}", v, f"{src}.{k}")
        else:
            reason = reasons.get(k) or panel.get(f"{k}_na_reason") or "null/undefined in source"
            c.na.append({"metric": f"{ns}.{k}", "source_path": f"{src}.{k}", "reason": reason})
    for k, v in (_get(panel, "binary.counts") or {}).items():
        c.add(f"{ns}.binary.count.{k}", v, f"{src}.binary.counts.{k}")


def add_flat(c: Child, ns: str, d, src: str) -> None:
    if not isinstance(d, dict):
        c.note(ns, "delta absent in source report")
        return
    for k, v in d.items():
        if finite(v):
            c.add(f"{ns}.{k}", v, f"{src}.{k}")
        else:
            c.na.append({"metric": f"{ns}.{k}", "source_path": f"{src}.{k}", "reason": "undefined delta in source"})


def add_boot(c: Child, ns: str, rec, src: str) -> None:
    if not isinstance(rec, dict):
        c.note(ns, "bootstrap record absent in source report")
        return
    c.panels[ns] = {"source_path": src, "bootstrap": {k: v for k, v in rec.items() if k != "country_counts"}}
    c.add(f"{ns}.point_delta", rec.get("point_delta"), f"{src}.point_delta")
    for k in ("K", "draws", "defined_draws", "undefined_draws", "seed"):
        if k in rec:
            c.add(f"{ns}.{k}", rec[k], f"{src}.{k}")
    iv = rec.get("interval")
    if isinstance(iv, list) and len(iv) == 2:
        c.add(f"{ns}.ci_lower", iv[0], f"{src}.interval[0]")
        c.add(f"{ns}.ci_upper", iv[1], f"{src}.interval[1]")
    else:
        c.na.append({"metric": f"{ns}.ci", "source_path": f"{src}.interval", "reason": rec.get("na_reason") or "no interval"})
    c.tags[f"bootstrap.{ns}"] = (f"country cluster bootstrap, draws={rec.get('draws')}, rng seed={rec.get('seed')}, "
                                 "pointwise, conditional on saved predictions")


def add_tree(c: Child, ns: str, obj, src: str, depth: int = 0) -> None:
    """Numeric leaves of a coverage/route dict (lists skipped; they stay in the report artifact)."""
    if depth > 3 or not isinstance(obj, dict):
        return
    for k, v in obj.items():
        name = re.sub(r"[^\w\-. ]", "_", str(k))  # MLflow-safe; original key kept in the provenance path
        if isinstance(v, dict):
            add_tree(c, f"{ns}.{name}", v, f"{src}.{k}", depth + 1)
        elif finite(v):
            c.add(f"{ns}.{name}", v, f"{src}.{k}")


# ------------------------------------------------------------ cohort key digests

def key_digest(frame: pd.DataFrame) -> tuple[str, int]:
    keys = sorted(f"{int(a)}|{int(t)}" for a, t in zip(frame["admin_code"], frame["target_ord"]))
    return hashlib.sha256("\n".join(keys).encode()).hexdigest(), len(keys)


def read_pred(path: Path, cols: list) -> pd.DataFrame:
    if not path.is_file():
        raise SourceConflict(f"prediction file missing: {path}")
    return pd.read_csv(path, usecols=lambda c: c in cols, dtype={"region": str}, keep_default_na=True)


def cohort_tag(c: Child, ns: str, frame: pd.DataFrame) -> None:
    digest, n = key_digest(frame)
    c.tags[f"cohort_keys.{ns}"] = digest
    reported = c.metrics.get(f"{ns}.n")
    if reported is not None and int(reported) != n:
        raise SourceConflict(f"{c.key}: cohort {ns} has {n} keys in predictions but report n={reported}")
    c.metrics.setdefault(f"{ns}.n", float(n))
    c.provenance.setdefault(f"{ns}.n", "predictions key count")


# ------------------------------------------------------------ family extractors

def _p6_children(fam: dict, rep: dict, root: Path) -> list:
    out = []
    seed = fam.get("model_seed", "42")
    cmp_ = json.loads((root / fam["comparison"]).read_text()) if fam.get("comparison") else None
    comb = json.loads((root / fam["combined"]).read_text()) if fam.get("combined") else None
    for H in HORIZONS:
        hrep = rep["horizons"].get(H)
        if hrep is None:
            raise SourceConflict(f"{fam['family']}: H{H} missing from report")
        pred = read_pred(root / fam["predictions"].format(H=int(H)),
                         ["admin_code", "target_ord", "target_month", "period", "persistence_available"])
        for arm, kind in fam["arms"].items():
            c = Child(fam["family"], fam["source_run_id"], H, arm, "none" if kind == "persistence_baseline" else seed, kind)
            for period in PERIODS:
                e = hrep.get(period)
                if e is None:
                    c.note(period, "period absent in source report")
                    continue
                src = f"report.horizons.{H}.{period}"
                sub = pred[pred["period"] == period]
                if arm in ("pool", "geo"):
                    ea = e["E_all"]
                    c.add(f"{period}.E_all.n", ea.get("n"), f"{src}.E_all.n")
                    if ea.get("status") == "scored":
                        add_panel(c, f"{period}.E_all", ea.get(arm), f"{src}.E_all.{arm}")
                    else:
                        c.note(f"{period}.E_all", ea.get("na_reason") or ea.get("status") or "empty cohort")
                    cohort_tag(c, f"{period}.E_all", sub)
                if arm in ("geo", "persistence"):
                    ep = e["E_persist"]
                    c.add(f"{period}.E_persist.n", ep.get("n"), f"{src}.E_persist.n")
                    if ep.get("status") == "scored":
                        add_panel(c, f"{period}.E_persist", ep.get(arm), f"{src}.E_persist.{arm}")
                    else:
                        c.note(f"{period}.E_persist", ep.get("na_reason") or ep.get("status") or "empty cohort")
                    cohort_tag(c, f"{period}.E_persist", sub[sub["persistence_available"] == 1])
                if arm == "pool":
                    c.note(f"{period}.E_persist", "pool arm is not reported on E_persist in this source format")
                if arm == "persistence":
                    c.note(f"{period}.E_all", "persistence is only defined on persistence-available keys (E_persist)")
                if arm == "geo":
                    if e["E_all"].get("status") == "scored":
                        add_flat(c, f"{period}.E_all.delta.geo_minus_pool", e["E_all"].get("delta_geo_minus_pool"),
                                 f"{src}.E_all.delta_geo_minus_pool")
                    if e["E_persist"].get("status") == "scored":
                        add_flat(c, f"{period}.E_persist.delta.geo_minus_persistence",
                                 e["E_persist"].get("delta_geo_minus_persistence"), f"{src}.E_persist.delta_geo_minus_persistence")
                    boot = e.get("bootstrap") or {}
                    if "geo_vs_pool_E_all" in boot:
                        add_boot(c, f"{period}.E_all.contrast.geo_vs_pool", boot["geo_vs_pool_E_all"], f"{src}.bootstrap.geo_vs_pool_E_all")
                    if "geo_vs_persistence_E_persist" in boot:
                        add_boot(c, f"{period}.E_persist.contrast.geo_vs_persistence", boot["geo_vs_persistence_E_persist"],
                                 f"{src}.bootstrap.geo_vs_persistence_E_persist")
                    add_tree(c, f"{period}.coverage", e.get("coverage"), f"{src}.coverage")
                    add_tree(c, f"{period}.routes", e.get("routes"), f"{src}.routes")
                if cmp_ is not None and arm in ("pool", "geo"):
                    ce = cmp_["horizons"][H].get(period)
                    if ce is not None and "E_all" in ce:
                        csrc = f"compare-p6.horizons.{H}.{period}"
                        old = "old_geo" if arm == "geo" else "old_pool"
                        add_panel(c, f"{period}.E_all.matched_old.p6{arm}", ce["E_all"]["panels"].get(old), f"{csrc}.E_all.panels.{old}")
                        add_flat(c, f"{period}.E_all.delta.new_minus_old_{arm}", ce["E_all"].get(f"delta_new_minus_old_{arm}"),
                                 f"{csrc}.E_all.delta_new_minus_old_{arm}")
                        b = (ce.get("bootstrap") or {}).get(f"new_minus_old_{arm}_E_all")
                        if b is not None:
                            add_boot(c, f"{period}.E_all.contrast.new_minus_old_{arm}", b, f"{csrc}.bootstrap.new_minus_old_{arm}_E_all")
                        if arm == "geo":
                            add_panel(c, f"{period}.E_persist.matched_old.p6geo", ce["E_persist"]["panels"].get("old_geo"),
                                      f"{csrc}.E_persist.panels.old_geo")
                            add_flat(c, f"{period}.E_persist.matched_old.delta.p6geo_minus_persistence",
                                     ce["E_persist"].get("delta_old_geo_minus_persistence"),
                                     f"{csrc}.E_persist.delta_old_geo_minus_persistence")
                            b = (ce.get("bootstrap") or {}).get("old_geo_minus_persistence_E_persist")
                            if b is not None:
                                add_boot(c, f"{period}.E_persist.matched_old.contrast.p6geo_minus_persistence", b,
                                         f"{csrc}.bootstrap.old_geo_minus_persistence_E_persist")
                            add_tree(c, f"{period}.label_flips_vs_p6", ce.get("label_flips"), f"{csrc}.label_flips")
                        c.tags["comparator_parent"] = P6_PARENT
            if comb is not None:
                hc = comb["horizons"].get(H)
                if hc is None:
                    c.note("combined", f"H{H} absent from combined-and-matched-report.json")
                else:
                    for blk, name in (("combined", "combined"), ("2025", "y2025"), ("2026", "y2026")):
                        be = hc.get(blk)
                        if be is None:
                            c.note(name, "block absent in combined report")
                            continue
                        bsrc = f"combined.horizons.{H}.{blk}"
                        if blk == "combined":
                            bsub = pred
                        else:
                            bsub = pred[pred["target_month"].astype(str).str[:4] == blk]
                        if arm in ("pool", "geo"):
                            ea = be["E_all"]
                            c.add(f"{name}.E_all.n", ea.get("n"), f"{bsrc}.E_all.n")
                            if ea.get("status") == "scored":
                                add_panel(c, f"{name}.E_all", ea.get(arm), f"{bsrc}.E_all.{arm}")
                                add_panel(c, f"{name}.E_all.matched_old.p6{arm}", (be.get("old_split_matched") or {}).get(arm),
                                          f"{bsrc}.old_split_matched.{arm}")
                                c.tags["comparator_parent"] = P6_PARENT
                            else:
                                c.note(f"{name}.E_all", ea.get("status") or "empty cohort")
                            cohort_tag(c, f"{name}.E_all", bsub)
                        if arm in ("geo", "persistence"):
                            ep = be["E_persist"]
                            c.add(f"{name}.E_persist.n", ep.get("n"), f"{bsrc}.E_persist.n")
                            if ep.get("status") == "scored":
                                add_panel(c, f"{name}.E_persist", ep.get(arm), f"{bsrc}.E_persist.{arm}")
                            else:
                                c.note(f"{name}.E_persist", ep.get("status") or "empty cohort")
                            cohort_tag(c, f"{name}.E_persist", bsub[bsub["persistence_available"] == 1])
                        if arm == "geo" and be["E_all"].get("status") == "scored":
                            add_flat(c, f"{name}.E_all.delta.geo_minus_pool", be["E_all"].get("delta_geo_minus_pool"),
                                     f"{bsrc}.E_all.delta_geo_minus_pool")
                            add_flat(c, f"{name}.E_all.delta.new_minus_old_geo", be.get("delta_new_minus_old_geo"),
                                     f"{bsrc}.delta_new_minus_old_geo")
                            for k in ("local_rows", "unmapped_rows"):
                                c.add(f"{name}.{k}", be.get(k), f"{bsrc}.{k}")
            out.append(c)
    return out


def _diag(c: Child, period: str, e: dict, src: str, arm: str, comparator: str | None) -> None:
    d = e.get("ungated_local_diagnostic")
    if not isinstance(d, dict):
        c.note(f"{period}.local_eligible", "diagnostic absent")
        return
    c.add(f"{period}.local_eligible.n_keys", d.get("keys"), f"{src}.ungated_local_diagnostic.keys")
    allp = d.get("all")
    if isinstance(allp, dict):
        add_panel(c, f"{period}.local_eligible", allp["panels"].get(arm), f"{src}.ungated_local_diagnostic.all.panels.{arm}")
        if comparator:
            add_panel(c, f"{period}.local_eligible.comparator.{comparator[1]}", allp["panels"].get(comparator[0]),
                      f"{src}.ungated_local_diagnostic.all.panels.{comparator[0]}")
        if arm == "local":
            add_flat(c, f"{period}.local_eligible.delta.local_minus_pool", allp.get("local_minus_pool"),
                     f"{src}.ungated_local_diagnostic.all.local_minus_pool")
    if arm in ("local", "pool"):
        for g, ge in (d.get("by_gate") or {}).items():
            add_panel(c, f"{period}.local_eligible.{g}", (ge.get("panels") or {}).get(arm),
                      f"{src}.ungated_local_diagnostic.by_gate.{g}.panels.{arm}")
            if arm == "local":
                add_flat(c, f"{period}.local_eligible.{g}.delta.local_minus_pool", ge.get("local_minus_pool"),
                         f"{src}.ungated_local_diagnostic.by_gate.{g}.local_minus_pool")


def _mlp_children(fam: dict, rep: dict, root: Path) -> list:
    out = []
    comp = {"geo": ("xgbgeo", "p6geo"), "pool": ("xgbpool", "p6pool")}
    owner = {"geo_minus_pool": "geo", "pool_minus_base": "pool", "base_minus_xgbpool": "base", "geo_minus_xgbgeo": "geo"}
    persist_ref: dict = {}
    for seed in fam["seeds"]:
        for H in HORIZONS:
            pred = read_pred(root / fam["predictions"].format(seed=seed, H=int(H)),
                             ["admin_code", "target_ord", "period", "persistence_available", "local_eligible"])
            for arm, kind in fam["arms"].items():
                if arm == "persistence":
                    continue
                c = Child(fam["family"], fam["source_run_id"], H, arm, seed, kind)
                for period in PERIODS:
                    e = rep["replicates"][seed][H].get(period)
                    if e is None:
                        c.note(period, "period absent in source report")
                        continue
                    src = f"report.replicates.{seed}.{H}.{period}"
                    sub = pred[pred["period"] == period]
                    if arm != "local":
                        for coh, frame in (("E_all", sub), ("E_persist", sub[sub["persistence_available"] == 1])):
                            ce = e[coh]
                            c.add(f"{period}.{coh}.n", ce.get("n"), f"{src}.{coh}.n")
                            if ce.get("status") == "scored":
                                add_panel(c, f"{period}.{coh}", ce["panels"].get(arm), f"{src}.{coh}.panels.{arm}")
                                if arm in comp:
                                    add_panel(c, f"{period}.{coh}.comparator.{comp[arm][1]}", ce["panels"].get(comp[arm][0]),
                                              f"{src}.{coh}.panels.{comp[arm][0]}")
                                    c.tags["comparator_parent"] = P6_PARENT
                            else:
                                c.note(f"{period}.{coh}", ce.get("status") or "empty cohort")
                            cohort_tag(c, f"{period}.{coh}", frame)
                        for name, d in (e["E_all"].get("deltas") or {}).items():
                            if owner.get(name) == arm:
                                add_flat(c, f"{period}.E_all.delta.{name}", d, f"{src}.E_all.deltas.{name}")
                        d = (e["E_persist"].get("deltas") or {}).get(f"{arm}_minus_persistence")
                        if d is not None:
                            add_flat(c, f"{period}.E_persist.delta.{arm}_minus_persistence", d,
                                     f"{src}.E_persist.deltas.{arm}_minus_persistence")
                        for kind_ in ("raw", "star"):
                            v = (e.get("q3_mse") or {}).get(f"{arm}_{kind_}")
                            if v is not None:
                                c.add(f"{period}.E_all.q3_mse.{kind_}", v, f"{src}.q3_mse.{arm}_{kind_}")
                    else:
                        c.note(f"{period}.E_all", "local predictions exist only on L-eligible keys (local_eligible cohort)")
                    _diag(c, period, e, src, arm, comp.get(arm))
                    cohort_tag(c, f"{period}.local_eligible", sub[sub["local_eligible"] == 1])
                    if arm == "geo":
                        boot = e.get("bootstrap") or {}
                        for bname, ns in (("geo_minus_pool_E_all", "E_all.contrast.geo_minus_pool"),
                                          ("geo_minus_persistence_E_persist", "E_persist.contrast.geo_minus_persistence")):
                            if bname in boot:
                                add_boot(c, f"{period}.{ns}", boot[bname], f"{src}.bootstrap.{bname}")
                        add_tree(c, f"{period}.coverage", e.get("coverage"), f"{src}.coverage")
                        add_tree(c, f"{period}.label_flips_geo_vs_pool", e.get("label_flips_geo_vs_pool"),
                                 f"{src}.label_flips_geo_vs_pool")
                out.append(c)
            # persistence: identical across seeds (verified), one view per H
            for period in PERIODS:
                ce = rep["replicates"][seed][H][period]["E_persist"]
                p = ce["panels"].get("persistence") if ce.get("status") == "scored" else None
                sub = pred[(pred["period"] == period) & (pred["persistence_available"] == 1)]
                ident = (json.dumps(p, sort_keys=True), key_digest(sub))
                ref = persist_ref.setdefault((H, period), ident)
                if ref != ident:
                    raise SourceConflict(f"MLP persistence H{H} {period} differs between seeds")
    for H in HORIZONS:
        c = Child(fam["family"], fam["source_run_id"], H, "persistence", "none", "persistence_baseline")
        pred = read_pred(root / fam["predictions"].format(seed=fam["seeds"][0], H=int(H)),
                         ["admin_code", "target_ord", "period", "persistence_available"])
        for period in PERIODS:
            ce = rep["replicates"][fam["seeds"][0]][H][period]["E_persist"]
            src = f"report.replicates.{fam['seeds'][0]}.{H}.{period}.E_persist"
            c.add(f"{period}.E_persist.n", ce.get("n"), f"{src}.n")
            if ce.get("status") == "scored":
                add_panel(c, f"{period}.E_persist", ce["panels"].get("persistence"), f"{src}.panels.persistence")
            cohort_tag(c, f"{period}.E_persist", pred[(pred["period"] == period) & (pred["persistence_available"] == 1)])
            c.note(f"{period}.E_all", "persistence is only defined on persistence-available keys (E_persist)")
        c.tags["dedup"] = "identical persistence panel and keys verified across seeds 42/43/44; one view per H"
        out.append(c)
    return out


def _yearly_children(fam: dict, rep: dict, root: Path) -> list:
    out = []
    comp = {"geo": ("p6geo", "p6geo"), "pool": ("p6pool", "p6pool")}
    owner = {"geo_minus_pool": "geo", "pool_minus_p6pool": "pool", "geo_minus_p6geo": "geo"}
    seed = fam.get("model_seed", "42")
    for H in HORIZONS:
        pred = read_pred(root / fam["predictions"].format(H=int(H)),
                         ["admin_code", "target_ord", "period", "persistence_available", "local_eligible"])
        for arm, kind in fam["arms"].items():
            c = Child(fam["family"], fam["source_run_id"], H, arm, "none" if kind == "persistence_baseline" else seed, kind)
            for period in PERIODS:
                e = rep["horizons"][H].get(period)
                if e is None:
                    c.note(period, "period absent in source report")
                    continue
                src = f"report.horizons.{H}.{period}"
                sub = pred[pred["period"] == period]
                lp = sub[(sub["local_eligible"] == 1) & (sub["persistence_available"] == 1)]
                cohorts = (("E_all", sub), ("E_persist", sub[sub["persistence_available"] == 1]))
                for coh, frame in cohorts:
                    if arm == "local" or (arm == "persistence" and coh == "E_all"):
                        c.note(f"{period}.{coh}", "arm not defined on this cohort in the source report")
                        continue
                    ce = e[coh]
                    c.add(f"{period}.{coh}.n", ce.get("n"), f"{src}.{coh}.n")
                    if ce.get("status") == "scored":
                        add_panel(c, f"{period}.{coh}", ce["panels"].get(arm), f"{src}.{coh}.panels.{arm}")
                        if arm in comp:
                            add_panel(c, f"{period}.{coh}.comparator.{comp[arm][1]}", ce["panels"].get(comp[arm][0]),
                                      f"{src}.{coh}.panels.{comp[arm][0]}")
                            c.tags["comparator_parent"] = P6_PARENT
                    else:
                        c.note(f"{period}.{coh}", ce.get("status") or "empty cohort")
                    cohort_tag(c, f"{period}.{coh}", frame)
                for name, d in (e["E_all"].get("deltas") or {}).items():
                    if owner.get(name) == arm:
                        add_flat(c, f"{period}.E_all.delta.{name}", d, f"{src}.E_all.deltas.{name}")
                for name, d in (e["E_persist"].get("deltas") or {}).items():
                    a = name.split("_minus_")[0]
                    if a == arm:
                        add_flat(c, f"{period}.E_persist.delta.{name}", d, f"{src}.E_persist.deltas.{name}")
                    elif comp.get(arm, (None,))[0] == a:
                        add_flat(c, f"{period}.E_persist.comparator_delta.{name}", d, f"{src}.E_persist.deltas.{name}")
                if arm != "persistence":
                    _diag(c, period, e, src, arm, comp.get(arm))
                    cohort_tag(c, f"{period}.local_eligible", sub[sub["local_eligible"] == 1])
                m = e.get("local_persistence_matched") or {}
                c.add(f"{period}.local_persist_matched.n_keys", m.get("keys"), f"{src}.local_persistence_matched.keys")
                if arm in ("local", "pool", "geo", "persistence") and isinstance(m.get("panels"), dict):
                    add_panel(c, f"{period}.local_persist_matched", m["panels"].get(arm),
                              f"{src}.local_persistence_matched.panels.{arm}")
                    for name, d in (m.get("deltas") or {}).items():
                        if name.split("_minus_")[0] == arm:
                            add_flat(c, f"{period}.local_persist_matched.delta.{name}", d,
                                     f"{src}.local_persistence_matched.deltas.{name}")
                    if len(lp):
                        digest, n = key_digest(lp)
                        c.tags[f"cohort_keys.{period}.local_persist_matched"] = digest
                        if m.get("keys") is not None and int(m["keys"]) != n:
                            raise SourceConflict(f"{c.key}: local_persist_matched key count {n} != report {m['keys']}")
                if arm == "geo":
                    boot = e.get("bootstrap") or {}
                    for bname, ns in (("geo_minus_pool_E_all", "E_all.contrast.geo_minus_pool"),
                                      ("geo_minus_persistence_E_persist", "E_persist.contrast.geo_minus_persistence")):
                        if bname in boot:
                            add_boot(c, f"{period}.{ns}", boot[bname], f"{src}.bootstrap.{bname}")
                    add_tree(c, f"{period}.coverage", e.get("coverage"), f"{src}.coverage")
                    add_tree(c, f"{period}.label_flips_geo_vs_pool", e.get("label_flips_geo_vs_pool"),
                             f"{src}.label_flips_geo_vs_pool")
            out.append(c)
    return out


def _window_children(fam: dict, rep: dict, root: Path) -> list:
    out = []
    seed = fam.get("model_seed", "42")
    for H in HORIZONS:
        h = int(H)
        cells = [x for x in rep["cells"] if x["H"] == h]
        aggs = [x for x in rep["aggregates"] if x["H"] == h]
        if not cells or not aggs:
            raise SourceConflict(f"window: H{H} has no cells/aggregates")
        frames = []
        for cell in cells:
            f = read_pred(root / fam["predictions"].format(H=h, U=cell["U"]),
                          ["admin_code", "target_ord", "region", "base_local_ok", "exp_local_ok"])
            f["U"] = cell["U"]
            frames.append(f)
        allp = pd.concat(frames, ignore_index=True)
        allp["region"] = allp["region"].fillna("")
        bl = allp["base_local_ok"].astype(str).str.lower().isin(["true", "1"])
        el = allp["exp_local_ok"].astype(str).str.lower().isin(["true", "1"])
        cohorts = {"all": allp, "mapped": allp[allp["region"] != ""], "common_local_support": allp[bl & el],
                   "new_local_support": allp[~bl & el]}
        for arm, kind in fam["arms"].items():
            c = Child(fam["family"], fam["source_run_id"], H, arm, seed, kind)
            for a in aggs:
                ns = f"selected_dates.{a['cohort']}"
                c.add(f"{ns}.n", a.get("n"), f"summary.aggregates[H={h},cohort={a['cohort']}].n")
                add_panel(c, ns, a["panels"].get(arm), f"summary.aggregates[H={h},cohort={a['cohort']}].panels.{arm}")
                cohort_tag(c, ns, cohorts[a["cohort"]])
            for cell in cells:
                ns = f"selected_date_{cell['U']}.all"
                add_panel(c, ns, cell["panels"].get(arm), f"summary.cells[H={h},U={cell['U']}].panels.{arm}")
                cohort_tag(c, ns, allp[allp["U"] == cell["U"]])
                fit_key = "new_fit_n" if arm.startswith("exp") else "old_fit_n"
                c.add(f"selected_date_{cell['U']}.fit_n", cell.get(fit_key), f"summary.cells[H={h},U={cell['U']}].{fit_key}")
            c.tags["scope_note"] = ("selected dates only; local arms fall back to the corresponding global outside local "
                                    "support; only common_local_support compares supported local fits")
            out.append(c)
    return out


EXTRACTORS = {"p6": _p6_children, "mlp": _mlp_children, "yearly": _yearly_children, "window": _window_children}


def parent_metrics(fam: dict, rep: dict) -> dict:
    out = {}
    if fam["format"] == "mlp":
        for k, v in (rep.get("seed_mean_range_descriptive") or {}).items():
            for metric, stats in v.items():
                for s, x in stats.items():
                    if finite(x):
                        out[f"seed_summary.{k}.{metric}.{s}"] = float(x)
    return out


def extract(fam: dict, root: Path) -> tuple[list, dict, str]:
    report_path = root / fam["report"]
    if not report_path.is_file():
        raise SourceConflict(f"{fam['family']}: report missing at {report_path}")
    raw = report_path.read_bytes()
    rep = json.loads(raw)
    children = EXTRACTORS[fam["format"]](fam, rep, root)
    return children, parent_metrics(fam, rep), hashlib.sha256(raw).hexdigest()
