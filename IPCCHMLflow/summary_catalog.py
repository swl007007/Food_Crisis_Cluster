"""`IPCCH - dashboard`: one wide row per family x arm x lead (x seed), external model catalog,
evaluation and training-pool dataset Inputs.

  plan       read-only: project the detailed records into dashboard rows, datasets and models;
             freeze counts and a plan fingerprint
  inventory  snapshot of the detailed experiment (proves apply leaves it unchanged)
  apply      write ONLY dashboard/catalog objects for one frozen plan (needs a fresh backup)
  verify     read back every row, input, model and version against the plan

Scores are copied from the detailed records; prediction rows are read only to bind evaluation
keys and truth, never to recompute a score. The only derived values are the MLP seed means
(mean of three saved values) and nothing else. Models are catalog descriptors (not loadable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import import_runs  # noqa: E402
import naming  # noqa: E402
from extract import SourceConflict, key_digest  # noqa: E402
from import_runs import NOTE, PROV, T_KEY, T_STATUS  # noqa: E402

import pandas as pd  # noqa: E402

VERSION = "ipcch-dashboard-v1"
DASHBOARD_EXPERIMENT = "IPCCH - dashboard"
LEAVES = ("binary.accuracy", "binary.precision", "binary.recall", "binary.f1", "binary.f2",
          "four_class.accuracy", "four_class.macro_f1", "share_phase3plus_r2", "n_rows")
ROLES = ("primary", "holdout", "combined", "selected_months")
COHORTS = ("all_scored", "persistence_available", "regional_model_fitted",
           "regional_model_fitted_and_persistence_available", "in_partition_map",
           "regional_model_fitted_both_windows", "regional_model_fitted_full_history_only")
PANEL = re.compile(rf"^({'|'.join(ROLES)})\.({'|'.join(COHORTS)})$")
TRUTH_COLUMNS = ("phase_truth", "q3_truth")
EXPECTED = {"rows": 136, "registered_models": 68, "model_versions": 100}
FP_TAG, ROW_STATUS = f"{PROV}dashboard_plan_fingerprint", f"{PROV}dashboard_status"
EXTERNAL_NOTE = ("External catalog descriptor of a historical multi-fold recipe imported from saved artifacts. "
                 "Not loadable, not retrained, no inference wrapper. Weights stay in the family run's models.tar; "
                 "its member manifest describes the files, which are not exclusive to this arm.")


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Store:
    """Local artifact access for the detailed experiment (mlflow-artifacts proxy or file URIs)."""

    def __init__(self, store: Path):
        self.root = store / "artifacts"

    def path(self, artifact_uri: str, rel: str) -> Path:
        if artifact_uri.startswith("mlflow-artifacts:/"):
            return self.root / artifact_uri.split("mlflow-artifacts:/", 1)[1].lstrip("/") / rel
        if artifact_uri.startswith("file://"):
            return Path(artifact_uri[len("file://"):]) / rel
        raise SourceConflict(f"unsupported artifact URI {artifact_uri}")


# ------------------------------------------------------------------ plan

def cohort(pred: pd.DataFrame, fmt: str, period: str, name: str) -> pd.DataFrame:
    """Rows of one original (period, cohort) namespace; same rules as extract.py."""
    if fmt == "window":
        region = pred["region"].fillna("")
        bl = pred["base_local_ok"].astype(str).str.lower().isin(["true", "1"])
        el = pred["exp_local_ok"].astype(str).str.lower().isin(["true", "1"])
        return {"all": pred, "mapped": pred[region != ""], "common_local_support": pred[bl & el],
                "new_local_support": pred[~bl & el]}[name]
    sub = pred if period == "combined" else pred[pred["period"] == period]
    pa = sub["persistence_available"].astype(int) == 1
    if name == "E_all":
        return sub
    if name == "E_persist":
        return sub[pa]
    le = sub["local_eligible"].astype(int) == 1
    return sub[le] if name == "local_eligible" else sub[le & pa]


def eval_descriptor(family: str, H: str, period: str, old_cohort: str, frame: pd.DataFrame, truth: str,
                    dates) -> dict:
    lines = sorted(f"{int(a)}|{int(t)}|{p}|{q}" for a, t, p, q in
                   zip(frame["admin_code"], frame["target_ord"], frame["phase_truth"], frame["q3_truth"]))
    keys, n = key_digest(frame)
    months = sorted(set(frame["target_month"].astype(str)))
    role = "selected_months" if period == "selected_dates" else naming.period_role(family, period)
    d = {"schema": "ipcch-evaluation-dataset-v2", "context": "evaluation", "lead_months": int(H),
         "period_role": role, "period": naming.span_label(family, H, period, dates),
         "cohort": naming.COHORTS[old_cohort], "cohort_definition": naming.COHORT_MEANING[naming.COHORTS[old_cohort]],
         "num_rows": n, "keys_sha256": keys, "truth_sha256": sha("\n".join(lines).encode()),
         "truth_columns": list(TRUTH_COLUMNS), "truth_definition": truth,
         "key_definition": "one row per admin_code|target_ord; truth text as saved in the predictions",
         "target_month_min": months[0] if months else None, "target_month_max": months[-1] if months else None}
    if naming.COHORTS[old_cohort] in naming.FAMILY_DEPENDENT:
        d["rows_depend_on"] = naming.FAMILIES[family]["short"]
    if dates:
        d["selected_target_months"] = list(dates)
    return d


def check_span(family: str, H: str, period: str, old_cohort: str, d: dict) -> None:
    """The period named in the dataset must cover the cohort's months (exactly, for all rows)."""
    if period == "selected_dates":
        months = sorted(set(d.get("selected_target_months") or []))
        if d["num_rows"] and not (months and d["target_month_min"] >= months[0] and d["target_month_max"] <= months[-1]):
            raise SourceConflict(f"{family} H{H}: selected months {months} do not cover {d['target_month_min']}..")
        return
    lo, hi = naming.period_span(family, period, H).split("..")
    if d["num_rows"] and (d["target_month_min"] < lo or d["target_month_max"] > hi):
        raise SourceConflict(f"{family} H{H} {period}: rows {d['target_month_min']}..{d['target_month_max']} "
                             f"outside named span {lo}..{hi}")
    if old_cohort in ("E_all", "all") and (d["target_month_min"], d["target_month_max"]) != (lo, hi):
        raise SourceConflict(f"{family} H{H} {period}: scored months {d['target_month_min']}.."
                             f"{d['target_month_max']} differ from the named span {lo}..{hi}")


def contrast_values(view: dict, role: str, coh: str, a: str, b: str) -> dict:
    """Crisis-F1 contrast own arm minus b, from the saved bootstrap (with interval) or the delta block."""
    m, out = view["metrics"], {}
    boot = f"{role}.{coh}.bootstrap.{a}_minus_{b}.binary.f1"
    if f"{boot}.delta" in m:
        out[""] = f"{boot}.delta"
        for k in ("ci_low", "ci_high"):
            if f"{boot}.{k}" in m:
                out[f".{k}"] = f"{boot}.{k}"
    elif f"{role}.{coh}.delta.{a}_minus_{b}.binary.f1" in m:
        out[""] = f"{role}.{coh}.delta.{a}_minus_{b}.binary.f1"
    return out


def build_plan(client, cfg: dict, store: Path) -> dict:
    st = Store(store)
    exp = client.get_experiment_by_name(cfg["experiment"])
    if exp is None:
        raise SourceConflict(f"experiment {cfg['experiment']} not found")
    runs = client.search_runs([exp.experiment_id], max_results=5000)
    for r in runs:
        if r.data.tags.get(T_STATUS) != "complete":
            raise SourceConflict(f"{r.data.tags.get(T_KEY)}: detailed record not complete -- stop")
    parents = {r.data.tags[f"{PROV}original_family"]: r for r in runs if r.data.tags.get("record_kind") == "family"}
    children = sorted((r for r in runs if r.data.tags.get("record_kind") == "evaluation"),
                      key=lambda r: r.data.tags[T_KEY])
    fams = {f["family"]: f for f in cfg["families"]}
    pred_cache, manifests, rows, models, datasets = {}, {}, [], [], {}

    def manifest(family: str) -> dict:
        if family not in manifests:
            parent = parents[family]
            out = {}
            for name in ("include", "excluded", "shared"):
                for x in json.loads(st.path(parent.info.artifact_uri, f"manifests/{name}.json").read_text()):
                    out[x["path"]] = x["sha256"]
            manifests[family] = out
        return manifests[family]

    def predictions(family: str, rel: str) -> pd.DataFrame:
        parent = parents[family]
        if (family, rel) not in pred_cache:
            p = st.path(parent.info.artifact_uri, f"source/{rel}")
            if manifest(family).get(rel) != sha_file(p):
                raise SourceConflict(f"{rel}: archived predictions differ from the family manifest -- stop")
            pred_cache[(family, rel)] = pd.read_csv(p, dtype=str, keep_default_na=False)
        return pred_cache[(family, rel)]

    def add_dataset(desc: dict, name: str) -> dict:
        full = sha(canon(desc))
        rec = datasets.setdefault(name, {"name": name, "digest": full[:32], "full_sha256": full, "descriptor": desc})
        if rec["full_sha256"] != full:
            raise SourceConflict(f"dataset name {name!r} would mean two different contents -- stop")
        return rec

    def training_dataset(family: str, H: str) -> dict:
        pool = naming.TRAINING_POOL[family]
        files = manifest(pool["from"])
        x, k = pool["x"].format(H=int(H)), pool["keys"].format(H=int(H))
        if x not in files or k not in files:
            raise SourceConflict(f"{family} H{H}: training pool files {x} / {k} not in {pool['from']} manifests")
        desc = {"schema": "ipcch-training-pool-v1", "context": "training", "lead_months": int(H),
                "features": pool["features"], "x_path": x, "x_sha256": files[x], "keys_path": k,
                "keys_sha256": files[k], "first_target_month": "2014-01", "definition": naming.TRAINING_POOL_TEXT}
        return add_dataset(desc, naming.training_dataset_name(pool["features"], H))

    for child in children:
        t = child.data.tags
        family, old_arm = t[f"{PROV}original_family"], t[f"{PROV}original_arm"]
        H, seed, kind = str(int(t["lead_months"])), t["seed"], t[f"{PROV}arm_kind"]
        fam, parent = fams[family], parents[family]
        view = json.loads(st.path(child.info.artifact_uri, "view/evaluation_view.json").read_text())
        if view["source_key"] != t[T_KEY] or view["metrics"] != child.data.metrics:
            raise SourceConflict(f"{t[T_KEY]}: archived view disagrees with the live record -- stop")
        na = {x["metric"]: x for x in view["na"]}
        a = t["arm"]
        values, sources, inputs, row_na = {}, {}, {}, {}
        for ns, panel in sorted(view["panels"].items()):
            pm = PANEL.match(ns)
            if not pm:
                continue
            old_ns = panel["original_namespace"]
            old_period, old_cohort = old_ns.split(".")
            if fam["format"] == "window":
                rels = sorted(p.name for p in st.path(parent.info.artifact_uri, "source").glob(
                    f"predictions_h{int(H):02d}_*.csv.gz"))
                pred = pd.concat([predictions(family, rel) for rel in rels], ignore_index=True)
                dates = [rel.rsplit("_", 1)[1].split(".")[0] for rel in rels]
            else:
                rel = fam["predictions"].format(H=int(H), seed=seed if seed != "none" else fam.get("seeds", ["42"])[0])
                pred, dates = predictions(family, rel), None
            frame = cohort(pred, fam["format"], old_period, old_cohort)
            desc = eval_descriptor(family, H, old_period, old_cohort, frame, naming.TRUTH, dates)
            check_span(family, H, old_period, old_cohort, desc)
            if desc["keys_sha256"] != t.get(f"{PROV}cohort_keys.{ns}") or desc["num_rows"] != int(view["metrics"].get(f"{ns}.n_rows", -1)):
                raise SourceConflict(f"{t[T_KEY]} {ns}: prediction keys/n disagree with the detailed record -- stop")
            ds = add_dataset(desc, naming.eval_dataset_name(family, H, old_period, old_cohort, dates))
            for leaf in LEAVES:
                name = f"{ns}.{leaf}"
                if name in view["metrics"]:
                    values[name], sources[name], inputs[name] = view["metrics"][name], name, ds["name"]
                elif name in na:
                    row_na[name] = {"reason": na[name]["reason"], "source_path": na[name]["source_path"]}
                else:
                    raise SourceConflict(f"{t[T_KEY]}: {name} neither finite nor NA -- stop")
            role, coh = pm.groups()
            pairs = []
            if a != "persistence" and coh in ("persistence_available", "regional_model_fitted_and_persistence_available"):
                pairs.append("persistence")
            if a not in ("persistence", "pooled") and coh in ("all_scored", "regional_model_fitted"):
                pairs.append("pooled")
            for b in pairs:
                for suffix, src in contrast_values(view, role, coh, a, b).items():
                    name = f"{ns}.binary.f1.minus_{b}{suffix}"
                    values[name], sources[name], inputs[name] = view["metrics"][src], src, ds["name"]
        model_key = None
        if kind != "persistence_baseline":
            model_key = t[T_KEY]
            models.append({
                "projection_key": model_key, "family": family, "old_arm": old_arm, "H": H, "seed": seed,
                "registered_model": naming.registered_model_name(family, old_arm, H),
                "logged_model_name": naming.logged_model_name(family, old_arm, H, seed),
                "original_run_id": child.info.run_id, "family_run_id": parent.info.run_id,
                "model_bundle_uri": f"runs:/{parent.info.run_id}/models.tar",
                "model_bundle_sha256": parent.data.tags.get(f"{PROV}bundle.models.tar.sha256"),
                "member_manifest_uri": f"runs:/{parent.info.run_id}/manifests/models.tar.members.json",
                "source_code_uri": f"runs:/{parent.info.run_id}/source_code",
                "code_commit": parent.data.tags.get(f"{PROV}source.code_commit")})
        train = training_dataset(family, H)
        rows.append({"projection_key": naming.record_key(family, old_arm, H, seed), "family": family,
                     "old_arm": old_arm, "H": H, "seed": seed, "aggregation": "single_seed" if naming.multi_seed(family)
                     else "single_run", "tags": {k: v for k, v in t.items() if not k.startswith(("_prov.", "mlflow."))},
                     "original_run_id": child.info.run_id, "original_source_key": t[T_KEY],
                     "family_run_id": parent.info.run_id, "model_keys": [model_key] if model_key else [],
                     "values": values, "value_sources": sources, "value_inputs": inputs, "na": row_na,
                     "training_dataset": train["name"]})
    rows += seed_means(rows)
    counts = {"rows": len(rows), "finite": sum(len(r["values"]) for r in rows),
              "na": sum(len(r["na"]) for r in rows), "model_versions": len(models),
              "registered_models": len({m["registered_model"] for m in models}),
              "datasets": len(datasets), "training_datasets": sum(1 for d in datasets.values()
                                                                   if d["descriptor"]["context"] == "training"),
              "rows_by_family": dict(Counter(naming.FAMILIES[r["family"]]["slug"] for r in rows)),
              "metric_names": len({k for r in rows for k in r["values"]})}
    keys = [r["projection_key"] for r in rows]
    if len(set(keys)) != len(keys) or len({m["projection_key"] for m in models}) != len(models):
        raise SourceConflict("duplicate projection keys -- stop")
    plan = {"version": VERSION, "source_experiment_id": exp.experiment_id, "counts": counts, "rows": rows,
            "models": models, "datasets": sorted(datasets.values(), key=lambda d: d["name"])}
    plan["fingerprint"] = sha(canon({k: plan[k] for k in ("version", "rows", "models", "datasets")}))
    return plan


def seed_means(rows: list) -> list:
    """One extra row per multi-seed family x arm x lead: mean of the saved per-seed values.
    A metric is averaged only when every seed has it on the same evaluation dataset."""
    groups = defaultdict(list)
    for r in rows:
        if naming.multi_seed(r["family"]) and r["seed"] != "none":
            groups[(r["family"], r["old_arm"], r["H"])].append(r)
    out = []
    for (family, old_arm, H), rs in sorted(groups.items()):
        seeds = sorted(r["seed"] for r in rs)
        if seeds != sorted(naming.FAMILIES[family]["seeds"]):
            raise SourceConflict(f"{family} {old_arm} H{H}: seeds {seeds} incomplete -- stop")
        values, sources, inputs, na = {}, {}, {}, {}
        for name in sorted({k for r in rs for k in r["values"]} | {k for r in rs for k in r["na"]}):
            if name.endswith((".ci_low", ".ci_high")):
                continue                                   # no interval for a seed mean
            have = [r for r in rs if name in r["values"]]
            if len(have) != len(rs):
                na[name] = {"reason": f"defined for {len(have)} of {len(rs)} seeds", "source_path": None}
            elif len({r["value_inputs"][name] for r in rs}) != 1:
                na[name] = {"reason": "evaluation rows differ between seeds", "source_path": None}
            else:
                values[name] = sum(r["values"][name] for r in rs) / len(rs)
                sources[name] = {r["seed"]: r["original_run_id"] for r in rs}
                inputs[name] = rs[0]["value_inputs"][name]
        base = rs[0]
        out.append({"projection_key": naming.record_key(family, old_arm, H, "mean"), "family": family,
                    "old_arm": old_arm, "H": H, "seed": "mean", "aggregation": "mean_of_3_seeds",
                    "tags": {**base["tags"], "seed": "mean"}, "original_run_id": "",
                    "original_source_key": ";".join(r["original_source_key"] for r in rs),
                    "family_run_id": base["family_run_id"], "model_keys": [m for r in rs for m in r["model_keys"]],
                    "values": values, "value_sources": sources, "value_inputs": inputs, "na": na,
                    "training_dataset": base["training_dataset"]})
    return out


def check_expected(plan: dict) -> None:
    c = plan["counts"]
    drift = {k: (c[k], v) for k, v in EXPECTED.items() if c[k] != v}
    if drift:
        raise SourceConflict(f"plan drifted from the approved scope {drift} -- stop")


# ------------------------------------------------------------------ snapshots

def inventory(client, store: Path, experiment: str) -> dict:
    """Detailed experiment state: run data plus artifact files (path, size, mtime)."""
    st = Store(store)
    exp = client.get_experiment_by_name(experiment)
    out = {}
    for r in client.search_runs([exp.experiment_id], max_results=5000, run_view_type=3):
        base = st.path(r.info.artifact_uri, "")
        files = {str(p.relative_to(base)): [p.stat().st_size, p.stat().st_mtime_ns]
                 for p in sorted(base.rglob("*")) if p.is_file()} if base.exists() else {}
        out[r.info.run_id] = {"tags": dict(r.data.tags), "metrics": dict(r.data.metrics),
                              "params": dict(r.data.params), "status": r.info.status,
                              "end_time": r.info.end_time, "lifecycle": r.info.lifecycle_stage,
                              "inputs": len(r.inputs.dataset_inputs) + len(r.inputs.model_inputs or []),
                              "outputs": len((r.outputs.model_outputs if r.outputs else None) or []),
                              "files": files}
    return {"experiment_id": exp.experiment_id, "runs": out, "sha256": sha(canon(out))}


# ------------------------------------------------------------------ names, tags, text

def row_name(row: dict) -> str:
    return naming.run_name(row["family"], row["old_arm"], row["H"], row["seed"])


def row_description(row: dict) -> str:
    f = naming.FAMILIES[row["family"]]
    new, window, role = naming.arm(row["family"], row["old_arm"])
    agg = ("Mean of the three seed rows (42/43/44); intervals are not averaged. " if row["seed"] == "mean" else "")
    return "\n\n".join([
        f"**{row_name(row)}**",
        (f"**What:** {naming.arm_meaning(row['family'], row['old_arm'])} {naming.lead_label(row['H'])} lead. "
         f"Family: {f['long']}. {f['what']} {agg}"),
        (f"**Compare with:** {naming.compare_with(row['family'], row['old_arm'])} Metric keys read "
         "<period_role>.<cohort>.<metric>; each metric is bound to the evaluation dataset of its period and cohort "
         "(Inputs). Compare values only on the same dataset. *.binary.f1.minus_persistence / minus_pooled are crisis-F1 "
         "differences copied from the family's report (ci_low/ci_high = 95% country-cluster bootstrap where saved)."),
        f"**Status:** {f['status_text']}",
        (f"**Caveats + original:** values copied from the detailed run(s) "
         f"(tag _prov.original_run_id), no rescoring. Undefined values are absent and listed in "
         "tag _prov.na_metrics and dashboard/row.json. Family conclusion: see the family run in "
         "'IPCCH - detailed runs'."),
    ])


def row_tags(row: dict, plan_fp: str, model_ids: list) -> dict:
    t = {**row["tags"], "record_kind": "dashboard_row", "aggregation": row["aggregation"],
         NOTE: row_description(row), f"{PROV}projection_key": row["projection_key"],
         f"{PROV}original_run_id": row["original_run_id"], f"{PROV}original_source_key": row["original_source_key"],
         f"{PROV}family_run_id": row["family_run_id"], FP_TAG: plan_fp, f"{PROV}dashboard_version": VERSION,
         f"{PROV}na_metrics": ",".join(sorted(row["na"])) or "none",
         f"{PROV}execution": "projection of a historical import; no fit or rescoring",
         f"{PROV}mlflow_timestamps": "MLflow times are projection times, not source fit times"}
    if model_ids:
        t[f"{PROV}model_ids"] = ",".join(model_ids)
        t["registered_model"] = naming.registered_model_name(row["family"], row["old_arm"], row["H"])
    return {k: str(v) for k, v in t.items()}


def model_tags(m: dict, plan_fp: str) -> dict:
    f = naming.FAMILIES[m["family"]]
    new, window, role = naming.arm(m["family"], m["old_arm"])
    t = {"family": f["slug"], "family_title": f["short"], "model_type": f["model_type"], "arm": new,
         "arm_role": role, "lead_months": naming.lead_tag(m["H"]), "seed": m["seed"], "features": f["features"],
         "maps": f["maps"], "training_window": f["training_window"], "catalog": "external",
         "inference": "not loadable (external descriptor)", **({"window": window} if window else {}),
         f"{PROV}projection_key": m["projection_key"], f"{PROV}original_run_id": m["original_run_id"],
         f"{PROV}family_run_id": m["family_run_id"], f"{PROV}model_bundle_uri": m["model_bundle_uri"],
         f"{PROV}model_bundle_sha256": m["model_bundle_sha256"], f"{PROV}member_manifest_uri": m["member_manifest_uri"],
         f"{PROV}source_code_uri": m["source_code_uri"], f"{PROV}code_commit": m["code_commit"],
         FP_TAG: plan_fp,
         NOTE: (f"{naming.logged_model_name(m['family'], m['old_arm'], m['H'], m['seed'])}. "
                f"{naming.arm_meaning(m['family'], m['old_arm'])} Family: {f['long']}. {EXTERNAL_NOTE}")}
    return {k: str(v) for k, v in t.items()}


def registered_model_description(m: dict) -> str:
    f = naming.FAMILIES[m["family"]]
    return (f"{naming.arm_meaning(m['family'], m['old_arm'])} {naming.lead_label(m['H'])} lead. Family: {f['long']}. "
            f"Versions are seeds of the historical import ({', '.join(f['seeds'])}); version numbers are MLflow "
            f"allocation. {EXTERNAL_NOTE}")


def dashboard_description(plan: dict) -> str:
    c = plan["counts"]
    return "\n\n".join([
        "**IPCCH dashboard** - one row per model family x arm x lead time (x seed), copied from 'IPCCH - detailed "
        f"runs' without rescoring. {c['rows']} rows; MLP rows per seed plus a 'mean of 3 seeds' row.",
        "**Vocabulary.** family = experiment design (tag family / family_title). arm = predictor within a family: "
        + "; ".join(f"{k}: {v}" for k, v in naming.ARM_MEANING.items())
        + ". arm_role: " + "; ".join(f"{k} = {v}" for k, v in naming.ROLE_MEANING.items())
        + ". lead_months = forecast lead (origin = target month minus lead).",
        "**Metric keys** read <period_role>.<cohort>.<metric>. period_role: "
        + "; ".join(f"{k} = {v}" for k, v in naming.PERIOD_ROLE_MEANING.items() if k in ROLES)
        + " (actual spans in tags period.*). Cohorts: "
        + " ".join(f"{k}: {naming.COHORT_MEANING[k]}" for k in COHORTS)
        + " Metrics: binary.* = crisis (phase 3+) vs not; four_class.* = phase 1|2|3|4-5; share_phase3plus_r2 = R2 "
        "of the population share in phase 3+; n_rows = rows scored. *.binary.f1.minus_persistence / minus_pooled = "
        "crisis-F1 difference to persistence / pooled on the same rows (ci_low/ci_high where a bootstrap was saved).",
        "**Comparability.** Compare values only on the same evaluation dataset (Inputs; the dataset name states lead, "
        "period span and cohort). GeoXGB maps to 2024 has a different primary period (2025 only), so its primary "
        "values are not comparable with the other families' primary values. The window probe covers selected months "
        "only. No global best-model ranking is implied.",
        "**Start here.** Filter `tags.lead_months = '03' AND tags.seed != 'mean'` (or `tags.family = "
        "'geoxgb_reference'`), then chart metric primary.persistence_available.binary.f1 grouped by tag arm, or "
        "primary.persistence_available.binary.f1.minus_persistence with ci_low/ci_high. For the MLP use "
        "`tags.seed = 'mean'`.",
        "**Models tab:** external catalog descriptors only (not loadable). Details, gate-decision subsets, coverage "
        "and per-period diagnostics stay in 'IPCCH - detailed runs' (tag _prov.original_run_id).",
    ])


# ------------------------------------------------------------------ apply

def _row_doc(row: dict, plan: dict, model_ids: list) -> bytes:
    ds = {d["name"]: d for d in plan["datasets"]}
    used = sorted(set(row["value_inputs"].values()) | {row["training_dataset"]})
    doc = {"projection_key": row["projection_key"], "values": row["values"], "value_sources": row["value_sources"],
           "value_datasets": row["value_inputs"], "na": row["na"],
           "datasets": {n: {"digest": ds[n]["digest"], "full_sha256": ds[n]["full_sha256"],
                            "descriptor": ds[n]["descriptor"]} for n in used},
           "model_ids": model_ids, "original_run_id": row["original_run_id"],
           "original_source_key": row["original_source_key"], "plan_fingerprint": plan["fingerprint"]}
    return json.dumps(doc, indent=1, sort_keys=True).encode()


def _dataset_entity(ds: dict):
    from mlflow.entities import Dataset
    d = ds["descriptor"]
    if d["context"] == "training":
        schema = {"mlflow_colspec": [{"name": "features", "type": "double"}, {"name": "q2..q5 targets", "type": "double"}]}
        source = {"authority": "prepared training pool of the imported IPCCH runs", "x_path": d["x_path"],
                  "x_sha256": d["x_sha256"], "keys_path": d["keys_path"], "keys_sha256": d["keys_sha256"]}
        profile = {"lead_months": d["lead_months"], "features": d["features"]}
    else:
        schema = {"mlflow_colspec": [{"name": "admin_code", "type": "long"}, {"name": "target_ord", "type": "long"},
                                     {"name": "target_month", "type": "string"}, {"name": "phase_truth", "type": "long"},
                                     {"name": "q3_truth", "type": "double"}]}
        source = {"authority": "saved predictions of the imported IPCCH runs (evaluation keys and truth only)",
                  "descriptor_sha256": ds["full_sha256"], "lead_months": d["lead_months"], "period": d["period"],
                  "cohort": d["cohort"]}
        profile = {"num_rows": d["num_rows"]}
    return Dataset(name=ds["name"], digest=ds["digest"], source_type=f"ipcch_{d['context']}",
                   source=json.dumps(source, sort_keys=True), schema=json.dumps(schema), profile=json.dumps(profile))


def _input_tags(ds: dict):
    from mlflow.entities import InputTag
    d = ds["descriptor"]
    tags = [InputTag("mlflow.data.context", d["context"]), InputTag(f"{PROV}descriptor_sha256", ds["full_sha256"])]
    if d["context"] == "evaluation":
        tags += [InputTag("cohort", d["cohort"]), InputTag("period", d["period"]),
                 InputTag(f"{PROV}keys_sha256", d["keys_sha256"]), InputTag(f"{PROV}truth_sha256", d["truth_sha256"])]
    return tags


class Journal:
    def __init__(self, path: Path, plan_fp: str):
        self.path = path
        self.data = json.loads(path.read_text()) if path.is_file() else {"plan_fingerprint": plan_fp, "objects": {}}
        if self.data["plan_fingerprint"] != plan_fp:
            raise SourceConflict(f"journal {path} belongs to another plan -- stop")

    def record(self, kind: str, key: str, value: str) -> None:
        self.data["objects"].setdefault(kind, {})[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        os.replace(tmp, self.path)


def _dashboard_experiment(client, plan: dict, artifact_location: str | None = None):
    exp = client.get_experiment_by_name(DASHBOARD_EXPERIMENT)
    if exp is None:
        eid = client.create_experiment(DASHBOARD_EXPERIMENT, artifact_location=artifact_location,
                                       tags={FP_TAG: plan["fingerprint"], "catalog_status": "incomplete",
                                             NOTE: dashboard_description(plan)})
        return client.get_experiment(eid)
    fp = exp.tags.get(FP_TAG)
    if fp != plan["fingerprint"]:
        raise SourceConflict(f"{DASHBOARD_EXPERIMENT} belongs to plan {fp}; this plan is {plan['fingerprint']} -- stop")
    return exp


def _existing_models(client, exp_id: str) -> dict:
    out, token = {}, None
    while True:
        page = client.search_logged_models([exp_id], max_results=500, page_token=token)
        for m in page:
            k = m.tags.get(f"{PROV}projection_key")
            if k:
                if k in out:
                    raise SourceConflict(f"two logged models share projection key {k} -- stop")
                out[k] = m
        token = page.token
        if not token:
            return out


def _finalize_pending(client, model) -> None:
    from mlflow.entities import LoggedModelStatus
    from mlflow.models.model import MLMODEL_FILE_NAME
    from mlflow.models.utils import get_external_mlflow_model_spec
    if not any(a.path == MLMODEL_FILE_NAME for a in client.list_logged_model_artifacts(model.model_id)):
        with tempfile.TemporaryDirectory() as d:
            get_external_mlflow_model_spec(model).save(os.path.join(d, MLMODEL_FILE_NAME))
            client.log_model_artifacts(model_id=model.model_id, local_dir=d)
    client.finalize_logged_model(model.model_id, LoggedModelStatus.READY)


def apply_plan(client, plan: dict, store: Path, journal: Journal, fail_after: str | None = None,
               artifact_location: str | None = None) -> dict:
    import mlflow
    from mlflow.entities import DatasetInput, LoggedModelInput, Metric, RunTag
    stats = Counter()
    exp = _dashboard_experiment(client, plan, artifact_location)
    eid = exp.experiment_id
    client.set_experiment_tag(eid, "catalog_status", "incomplete")
    existing = _existing_models(client, eid)
    model_ids = {}
    by_name = defaultdict(list)
    for m in plan["models"]:
        by_name[m["registered_model"]].append(m)
    for name, ms in sorted(by_name.items()):
        f = naming.FAMILIES[ms[0]["family"]]
        new, window, role = naming.arm(ms[0]["family"], ms[0]["old_arm"])
        rm_tags = {"family": f["slug"], "family_title": f["short"], "arm": new, "arm_role": role,
                   "lead_months": naming.lead_tag(ms[0]["H"]), "catalog": "external", FP_TAG: plan["fingerprint"],
                   **({"window": window} if window else {})}
        try:
            rm = client.get_registered_model(name)
            if {k: rm.tags.get(k) for k in rm_tags} != {k: str(v) for k, v in rm_tags.items()}:
                raise SourceConflict(f"registered model {name} exists with different tags -- stop")
        except SourceConflict:
            raise
        except Exception:
            client.create_registered_model(name, tags=rm_tags, description=registered_model_description(ms[0]))
            stats["registered_models_created"] += 1
        journal.record("registered_model", name, name)
        versions = {v.tags.get(f"{PROV}projection_key"): v for v in client.search_model_versions(f"name='{name}'")}
        for m in sorted(ms, key=lambda x: x["seed"]):
            tags = model_tags(m, plan["fingerprint"])
            lm = existing.get(m["projection_key"])
            if lm is None:
                lm = mlflow.create_external_model(name=m["logged_model_name"], source_run_id=m["original_run_id"],
                                                  tags=tags, params={"seed": m["seed"], "lead_months": m["H"],
                                                                     "arm": naming.arm(m["family"], m["old_arm"])[0]},
                                                  model_type=f"historical {naming.FAMILIES[m['family']]['model_type']}",
                                                  experiment_id=eid)
                stats["logged_models_created"] += 1
            else:
                if {k: lm.tags.get(k) for k in tags} != tags:
                    raise SourceConflict(f"logged model {m['projection_key']} exists with different tags -- stop")
                if str(lm.status) not in ("READY", "LoggedModelStatus.READY"):
                    _finalize_pending(client, lm)
                    stats["logged_models_finalized"] += 1
            model_ids[m["projection_key"]] = lm.model_id
            journal.record("logged_model", m["projection_key"], lm.model_id)
            if m["projection_key"] not in versions:
                mv = client.create_model_version(
                    name, source=f"models:/{lm.model_id}", model_id=lm.model_id,
                    tags={f"{PROV}projection_key": m["projection_key"], "seed": m["seed"],
                          f"{PROV}original_run_id": m["original_run_id"], FP_TAG: plan["fingerprint"]},
                    description=f"Seed {m['seed']}. {EXTERNAL_NOTE}")
                stats["model_versions_created"] += 1
                journal.record("model_version", m["projection_key"], f"{name}/{mv.version}")
            else:
                journal.record("model_version", m["projection_key"], f"{name}/{versions[m['projection_key']].version}")
    if fail_after == "models":
        raise import_runs.Interrupt("injected interruption after models")
    have = {}
    for r in client.search_runs([eid], max_results=5000, run_view_type=3):
        k = r.data.tags.get(f"{PROV}projection_key")
        if k in have:
            raise SourceConflict(f"two dashboard rows share projection key {k} -- stop")
        have[k] = r
    ds_by_name = {d["name"]: d for d in plan["datasets"]}
    for i, row in enumerate(plan["rows"]):
        if fail_after == "row-5" and i == 5:
            raise import_runs.Interrupt("injected interruption during rows")
        mids = [model_ids[k] for k in row["model_keys"]]
        tags = row_tags(row, plan["fingerprint"], mids)
        run = have.get(row["projection_key"])
        if run is None:
            run = client.create_run(eid, run_name=row_name(row),
                                    tags={f"{PROV}projection_key": row["projection_key"], ROW_STATUS: "in_progress",
                                          FP_TAG: plan["fingerprint"]})
            stats["rows_created"] += 1
        elif run.data.tags.get(FP_TAG) != plan["fingerprint"]:
            raise SourceConflict(f"{row['projection_key']}: dashboard row of another plan -- stop")
        elif run.data.tags.get(ROW_STATUS) == "complete":
            stats["rows_noop"] += 1
            journal.record("dashboard_run", row["projection_key"], run.info.run_id)
            continue
        rid = run.info.run_id
        journal.record("dashboard_run", row["projection_key"], rid)
        run = client.get_run(rid)
        for k, v in tags.items():
            if k in run.data.tags and run.data.tags[k] != v:
                raise SourceConflict(f"{row['projection_key']}: tag {k} differs -- stop")
        new_tags = [RunTag(k, v) for k, v in sorted(tags.items()) if run.data.tags.get(k) != v]
        for j in range(0, len(new_tags), 100):
            client.log_batch(rid, tags=new_tags[j:j + 100])
        params = {"family": naming.FAMILIES[row["family"]]["slug"], "arm": tags["arm"], "lead_months": row["H"],
                  "seed": row["seed"], **({"window": tags["window"]} if "window" in tags else {})}
        from mlflow.entities import Param
        ps = [Param(k, v) for k, v in sorted(params.items()) if k not in run.data.params]
        if ps:
            client.log_batch(rid, params=ps)
        ts = int(time.time() * 1000)
        mid = mids[0] if len(mids) == 1 else None
        mets = []
        for k, v in sorted(row["values"].items()):
            if k in run.data.metrics:
                if run.data.metrics[k] != v:
                    raise SourceConflict(f"{row['projection_key']}: metric {k} differs -- stop")
                continue
            ds = ds_by_name[row["value_inputs"][k]]
            mets.append(Metric(k, float(v), ts, 0, model_id=mid, dataset_name=ds["name"], dataset_digest=ds["digest"]))
        for j in range(0, len(mets), 1000):
            client.log_batch(rid, metrics=mets[j:j + 1000])
        stats["metrics_logged"] += len(mets)
        want = sorted(set(row["value_inputs"].values()) | {row["training_dataset"]})
        di = {(d.dataset.name, d.dataset.digest) for d in run.inputs.dataset_inputs}
        mi = {x.model_id for x in (run.inputs.model_inputs or [])}
        new_ds = [DatasetInput(_dataset_entity(ds_by_name[n]), tags=_input_tags(ds_by_name[n])) for n in want
                  if (n, ds_by_name[n]["digest"]) not in di]
        new_m = [LoggedModelInput(m) for m in mids if m not in mi]
        if new_ds or new_m:
            client.log_inputs(rid, datasets=new_ds or None, models=new_m or None)
            stats["inputs_logged"] += len(new_ds) + len(new_m)
        doc = _row_doc(row, plan, mids)
        arts = import_runs.existing_artifacts(client, rid)
        import_runs.upload(client, rid, arts, "dashboard/row.json", data=doc)
        verify_row(client, client.get_run(rid), row, plan, mids, deep=True)
        client.set_tag(rid, ROW_STATUS, "complete")
        client.set_terminated(rid, "FINISHED")
    return {"experiment_id": eid, **stats}


# ------------------------------------------------------------------ verify

def verify_row(client, run, row: dict, plan: dict, mids: list, deep: bool) -> None:
    key = row["projection_key"]
    if run.data.metrics != {k: float(v) for k, v in row["values"].items()}:
        raise SourceConflict(f"{key}: metric readback differs")
    for k, v in row_tags(row, plan["fingerprint"], mids).items():
        if run.data.tags.get(k) != v:
            raise SourceConflict(f"{key}: tag {k} readback differs")
    ds_by_name = {d["name"]: d for d in plan["datasets"]}
    want = sorted((n, ds_by_name[n]["digest"]) for n in set(row["value_inputs"].values()) | {row["training_dataset"]})
    got = sorted((d.dataset.name, d.dataset.digest) for d in run.inputs.dataset_inputs)
    if got != want:
        raise SourceConflict(f"{key}: dataset inputs {got[:3]} != plan")
    if sorted(x.model_id for x in (run.inputs.model_inputs or [])) != sorted(mids):
        raise SourceConflict(f"{key}: model inputs differ")
    if deep:
        for k in row["values"]:
            if len(client.get_metric_history(run.info.run_id, k)) != 1:
                raise SourceConflict(f"{key}: metric {k} has more than one history entry")
        if import_runs._download_bytes(client, run.info.run_id, "dashboard/row.json") != _row_doc(row, plan, mids):
            raise SourceConflict(f"{key}: dashboard/row.json readback differs")


def verify_all(client, plan: dict, store: Path, before: dict | None, cfg: dict) -> dict:
    exp = client.get_experiment_by_name(DASHBOARD_EXPERIMENT)
    if exp is None or exp.tags.get(FP_TAG) != plan["fingerprint"]:
        raise SourceConflict("dashboard experiment missing or of another plan")
    eid = exp.experiment_id
    models = _existing_models(client, eid)
    out = Counter()
    if set(models) != {m["projection_key"] for m in plan["models"]}:
        raise SourceConflict("logged model set differs from plan")
    names = defaultdict(set)
    for m in plan["models"]:
        lm = models[m["projection_key"]]
        if {k: lm.tags.get(k) for k in model_tags(m, plan["fingerprint"])} != model_tags(m, plan["fingerprint"]):
            raise SourceConflict(f"{m['projection_key']}: model tags differ")
        if lm.source_run_id != m["original_run_id"] or lm.tags.get("mlflow.model.isExternal") != "true":
            raise SourceConflict(f"{m['projection_key']}: model provenance differs")
        if lm.name != m["logged_model_name"]:
            raise SourceConflict(f"{m['projection_key']}: model name {lm.name!r} != {m['logged_model_name']!r}")
        names[m["registered_model"]].add(m["projection_key"])
        out["logged_models"] += 1
    for name, keys in names.items():
        vs = client.search_model_versions(f"name='{name}'")
        got = Counter(v.tags.get(f"{PROV}projection_key") for v in vs)
        if set(got) != keys or any(n != 1 for n in got.values()):
            raise SourceConflict(f"{name}: versions {dict(got)} != plan")
        for v in vs:
            if v.source != f"models:/{models[v.tags[f'{PROV}projection_key']].model_id}":
                raise SourceConflict(f"{name} v{v.version}: source differs")
        out["registered_models"] += 1
        out["model_versions"] += len(vs)
    runs = client.search_runs([eid], max_results=5000, run_view_type=3)
    by_key = defaultdict(list)
    for r in runs:
        by_key[r.data.tags.get(f"{PROV}projection_key")].append(r)
    if set(by_key) != {r["projection_key"] for r in plan["rows"]} or any(len(v) != 1 for v in by_key.values()):
        raise SourceConflict("dashboard run set differs from plan or has duplicates")
    used = set()
    for row in plan["rows"]:
        run = client.get_run(by_key[row["projection_key"]][0].info.run_id)   # search omits model inputs
        if run.data.tags.get(ROW_STATUS) != "complete":
            raise SourceConflict(f"{row['projection_key']}: incomplete")
        mids = [models[k].model_id for k in row["model_keys"]]
        verify_row(client, run, row, plan, mids, deep=True)
        used |= set(row["value_inputs"].values()) | {row["training_dataset"]}
        out["rows"] += 1
        out["metrics"] += len(row["values"])
    out["datasets"] = len(used)
    out["distinct_metric_names"] = len({k for r in runs for k in r.data.metrics})
    if before is not None:
        after = inventory(client, store, cfg["experiment"])
        if after["sha256"] != before["sha256"]:
            changed = [k for k in set(before["runs"]) | set(after["runs"]) if before["runs"].get(k) != after["runs"].get(k)]
            raise SourceConflict(f"detailed experiment changed: {len(changed)} runs, e.g. {changed[:3]}")
        out["detailed_runs_unchanged"] = len(after["runs"])
    out["catalog_status"] = exp.tags.get("catalog_status")
    return dict(out)


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "inventory", "apply", "verify"])
    ap.add_argument("--sources", default=str(Path(__file__).resolve().parent / "sources.json"))
    ap.add_argument("--store", default=str(import_runs.DEFAULT_ROOT))
    ap.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    ap.add_argument("--plan-fingerprint", help="apply/verify: the frozen plan fingerprint to act on")
    ap.add_argument("--backup", help="apply: backup dir made just before (backup-manifest.json must match the DB)")
    ap.add_argument("--inventory", help="apply/verify: pre-apply inventory JSON of the detailed experiment")
    ap.add_argument("--skip-expected", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--artifact-location", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fail-after", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(a.tracking_uri)
    client = MlflowClient(a.tracking_uri)
    cfg = json.loads(Path(a.sources).read_text())
    store = Path(a.store)
    sdir = store / "dashboard"
    sdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with import_runs.Lock(store / "import.lock"):
        if a.command == "inventory":
            inv = inventory(client, store, cfg["experiment"])
            res = {"command": "inventory", "runs": len(inv["runs"]), "sha256": inv["sha256"]}
            Path(a.out or sdir / f"inventory-{int(t0)}.json").write_text(json.dumps(inv, sort_keys=True))
        else:
            plan = build_plan(client, cfg, store)
            if not a.skip_expected:
                check_expected(plan)
            (sdir / f"plan-{plan['fingerprint'][:16]}.json").write_text(json.dumps(plan, indent=1, sort_keys=True))
            res = {"command": a.command, "plan_fingerprint": plan["fingerprint"], "counts": plan["counts"]}
            if a.command in ("apply", "verify"):
                if a.plan_fingerprint != plan["fingerprint"]:
                    raise SourceConflict(f"recomputed plan {plan['fingerprint']} != frozen {a.plan_fingerprint} -- stop")
                before = json.loads(Path(a.inventory).read_text()) if a.inventory else None
                if a.command == "apply":
                    if not a.backup or not a.inventory:
                        raise SourceConflict("apply needs --backup and --inventory made just before")
                    man = json.loads((Path(a.backup) / "backup-manifest.json").read_text())
                    import backup_restore
                    if man["db"]["counts"] != backup_restore.db_counts(store / "mlflow.db"):
                        raise SourceConflict("store changed since the backup -- make a fresh backup")
                    if inventory(client, store, cfg["experiment"])["sha256"] != before["sha256"]:
                        raise SourceConflict("detailed experiment changed since the inventory -- stop")
                    journal = Journal(sdir / f"journal-{plan['fingerprint'][:16]}.json", plan["fingerprint"])
                    res["apply"] = apply_plan(client, plan, store, journal, a.fail_after, a.artifact_location)
                    res["verify"] = verify_all(client, plan, store, before, cfg)
                    client.set_experiment_tag(res["apply"]["experiment_id"], "catalog_status", "complete")
                    res["verify"]["catalog_status"] = "complete"
                else:
                    res["verify"] = verify_all(client, plan, store, before, cfg)
    res["seconds"] = round(time.time() - t0, 1)
    if a.out and a.command != "inventory":
        Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
