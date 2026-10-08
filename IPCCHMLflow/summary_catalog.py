"""Compact `IPCCH Summary` records, external model catalog and row-free evaluation Inputs.

  plan    read-only: project the accepted IPCCH evaluation views into Summary rows, dataset
          descriptors and external-model entries; freeze counts and a plan fingerprint
  apply   write ONLY new Summary/catalog objects for one frozen plan (needs a fresh backup)
  verify  read back every Summary row, dataset input, model and version against the plan,
          and check the original IPCCH experiment against the pre-apply inventory

Scores are copied from the accepted views (their logged metrics); prediction rows are read
only to bind evaluation keys and truth, never to recompute a score. The original IPCCH
records are not modified. External models are catalog descriptors (not loadable).
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
from extract import SourceConflict, key_digest  # noqa: E402

import pandas as pd  # noqa: E402

VERSION = "ipcch-summary-catalog-v1"
SUMMARY_EXPERIMENT = "IPCCH Summary"
KEYS = ("binary.accuracy", "binary.precision", "binary.recall", "binary.f1", "binary.f2",
        "four_class.accuracy", "four_class.macro_f1", "q3_r2_projected", "n")
WHITELIST = re.compile(r"^(?:(main|supplementary)\.(E_all|E_persist|local_eligible|local_persist_matched)"
                       r"|(selected_dates)\.(all|mapped|common_local_support|new_local_support))$")
TRUTH_COLUMNS = ("phase_truth", "q3_truth")
EXPECTED = {"rows": 492, "finite": 4424, "na": 4, "registered_models": 68, "model_versions": 100}
EXTERNAL_NOTE = ("External catalog descriptor of a historical multi-fold recipe imported from saved "
                 "artifacts. Not loadable, not retrained, no inference wrapper. Weights stay in the "
                 "source parent's models.tar; member manifests describe selection, and archive members "
                 "are not exclusive to this arm.")


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
    """Local artifact access for an accepted import (mlflow-artifacts proxy or file URIs)."""

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
    """Same cohort rules as extract.py (the accepted import)."""
    if fmt == "window":
        region = pred["region"].fillna("")
        bl = pred["base_local_ok"].astype(str).str.lower().isin(["true", "1"])
        el = pred["exp_local_ok"].astype(str).str.lower().isin(["true", "1"])
        return {"all": pred, "mapped": pred[region != ""], "common_local_support": pred[bl & el],
                "new_local_support": pred[~bl & el]}[name]
    sub = pred[pred["period"] == period]
    pa = sub["persistence_available"].astype(int) == 1
    if name == "E_all":
        return sub
    if name == "E_persist":
        return sub[pa]
    le = sub["local_eligible"].astype(int) == 1
    return sub[le] if name == "local_eligible" else sub[le & pa]


def descriptor(H: str, period: str, name: str, frame: pd.DataFrame, truth_definition: str, dates) -> dict:
    lines = sorted(f"{int(a)}|{int(t)}|{p}|{q}" for a, t, p, q in
                   zip(frame["admin_code"], frame["target_ord"], frame["phase_truth"], frame["q3_truth"]))
    keys, n = key_digest(frame)
    months = sorted(set(frame["target_month"].astype(str)))
    d = {"schema": "ipcch-evaluation-dataset-v1", "context": "evaluation", "horizon_months": int(H),
         "period": period, "cohort": name, "num_rows": n, "keys_sha256": keys,
         "truth_sha256": sha("\n".join(lines).encode()), "truth_columns": list(TRUTH_COLUMNS),
         "truth_definition": truth_definition,
         "key_definition": "one row per admin_code|target_ord; truth text as saved in the predictions",
         "target_month_min": months[0] if months else None, "target_month_max": months[-1] if months else None}
    if dates:
        d["selected_dates"] = list(dates)
    return d


def build_plan(client, cfg: dict, store: Path) -> dict:
    st = Store(store)
    exp = client.get_experiment_by_name(cfg["experiment"])
    if exp is None:
        raise SourceConflict(f"experiment {cfg['experiment']} not found")
    runs = client.search_runs([exp.experiment_id], max_results=5000)
    parents = {r.data.tags["family"]: r for r in runs if r.data.tags.get("record_kind") == "source_run"}
    children = sorted((r for r in runs if r.data.tags.get("record_kind") == "evaluation_view"),
                      key=lambda r: r.data.tags["source_key"])
    fams = {f["family"]: f for f in cfg["families"]}
    for r in runs:
        if r.data.tags.get("import_status") != "complete":
            raise SourceConflict(f"{r.data.tags.get('source_key')}: original record not complete -- stop")
    pred_cache, include_cache, rows, models, datasets = {}, {}, [], [], {}

    def predictions(parent, rel: str) -> pd.DataFrame:
        if (parent.info.run_id, rel) not in pred_cache:
            if parent.info.run_id not in include_cache:
                inc = json.loads(st.path(parent.info.artifact_uri, "manifests/include.json").read_text())
                include_cache[parent.info.run_id] = {x["path"]: x["sha256"] for x in inc}
            p = st.path(parent.info.artifact_uri, f"source/{rel}")
            if include_cache[parent.info.run_id].get(rel) != sha_file(p):
                raise SourceConflict(f"{rel}: archived predictions differ from the parent manifest -- stop")
            pred_cache[(parent.info.run_id, rel)] = pd.read_csv(p, dtype=str, keep_default_na=False)
        return pred_cache[(parent.info.run_id, rel)]

    window_dates = {}
    for child in children:
        t = child.data.tags
        family, H, arm, seed, kind = t["family"], t["horizon"], t["arm"], t["seed"], t["arm_kind"]
        fam, parent = fams[family], parents[family]
        view_p = st.path(child.info.artifact_uri, "view/evaluation_view.json")
        view = json.loads(view_p.read_text())
        if view["key"] != t["source_key"] or view["metrics"] != child.data.metrics:
            raise SourceConflict(f"{t['source_key']}: archived view disagrees with the live record -- stop")
        nas = {x["metric"]: x for x in view["na"]}
        model_key = None
        if kind != "persistence_baseline":
            model_key = t["source_key"]
            models.append({
                "projection_key": model_key, "registered_model": f"ipcch.{family}.h{H}.{arm}",
                "logged_model_name": f"{family}-h{H}-{arm}-seed{seed}", "original_run_id": child.info.run_id,
                "source_parent_run_id": parent.info.run_id, "family": family, "horizon": H, "arm": arm,
                "seed": seed, "arm_kind": kind,
                "model_bundle_uri": f"runs:/{parent.info.run_id}/models.tar",
                "model_bundle_sha256": parent.data.tags.get("bundle.models.tar.sha256"),
                "member_manifest_uri": f"runs:/{parent.info.run_id}/manifests/models.tar.members.json",
                "source_code_uri": f"runs:/{parent.info.run_id}/source_code",
                "code_commit": parent.data.tags.get("source.code_commit"),
                "features": parent.data.tags.get("source.features"), "maps": parent.data.tags.get("source.maps"),
                "fit_protocol": parent.data.tags.get("source.fit_protocol")})
        for ns in sorted(view["panels"]):
            m = WHITELIST.match(ns)
            if not m:
                continue
            period = m.group(1) or m.group(3)
            cname = m.group(2) or m.group(4)
            values, sources, na = {}, {}, {}
            for k in KEYS:
                name = f"{ns}.{k}"
                if name in view["metrics"]:
                    values[k] = view["metrics"][name]
                    sources[k] = view["provenance"].get(name)
                elif name in nas:
                    na[k] = {"reason": nas[name]["reason"], "source_path": nas[name]["source_path"]}
                else:
                    raise SourceConflict(f"{t['source_key']}: {name} neither finite nor NA -- stop")
            if fam["format"] == "window":
                rels = window_dates.setdefault(H, sorted(
                    p.name for p in st.path(parent.info.artifact_uri, "source").glob(f"predictions_h{int(H):02d}_*.csv.gz")))
                pred = pd.concat([predictions(parent, rel) for rel in rels], ignore_index=True)
                dates = [rel.rsplit("_", 1)[1].split(".")[0] for rel in rels]
            else:
                rel = fam["predictions"].format(H=int(H), seed=seed if seed != "none" else fam.get("seeds", ["42"])[0])
                pred, dates = predictions(parent, rel), None
            frame = cohort(pred, fam["format"], period, cname)
            desc = descriptor(H, period, cname, frame, parent.data.tags.get("source.truth", ""), dates)
            if desc["keys_sha256"] != t.get(f"cohort_keys.{ns}") or desc["num_rows"] != int(values.get("n", -1)):
                raise SourceConflict(f"{t['source_key']} {ns}: prediction keys/n disagree with the accepted record -- stop")
            full = sha(canon(desc))
            dname, digest = f"ipcch-eval.h{H}.{period}.{cname}", full[:32]
            prev = datasets.setdefault((dname, digest), {"name": dname, "digest": digest, "full_sha256": full,
                                                         "descriptor": desc, "users": 0})
            if prev["full_sha256"] != full:
                raise SourceConflict(f"dataset digest collision for {dname}/{digest} -- stop")
            prev["users"] += 1
            pred_rel = (fam["predictions"].format(H=int(H), seed=seed if seed != "none" else fam.get("seeds", ["42"])[0])
                        if fam["format"] != "window" else f"predictions_h{int(H):02d}_*.csv.gz")
            rows.append({
                "projection_key": f"{t['source_key']}#{ns}", "family": family, "horizon": H, "arm": arm, "seed": seed,
                "arm_kind": kind, "period": period, "cohort": cname, "namespace": ns,
                "original_run_id": child.info.run_id, "original_parent_run_id": parent.info.run_id,
                "original_source_key": t["source_key"], "original_fingerprint": t["import_fingerprint"],
                "model_key": model_key, "dataset_name": dname, "dataset_digest": digest, "dataset_full_sha256": full,
                "values": values, "value_sources": sources, "na": na,
                "prediction_artifact": f"runs:/{parent.info.run_id}/source/{pred_rel}",
                "cohort_keys_tag": f"cohort_keys.{ns}",
                "truth_definition": parent.data.tags.get("source.truth", ""),
                "original_view_artifact": f"runs:/{child.info.run_id}/view/evaluation_view.json"})
    counts = {"rows": len(rows), "finite": sum(len(r["values"]) for r in rows),
              "na": sum(len(r["na"]) for r in rows), "model_versions": len(models),
              "registered_models": len({m["registered_model"] for m in models}),
              "dataset_descriptors": len(datasets), "dataset_names": len({k[0] for k in datasets}),
              "rows_by_family": dict(Counter(r["family"] for r in rows)),
              "models_by_kind": dict(Counter(m["arm_kind"] for m in models)),
              "metric_names": sorted({k for r in rows for k in r["values"]})}
    keys = [r["projection_key"] for r in rows]
    if len(set(keys)) != len(keys) or len({m["projection_key"] for m in models}) != len(models):
        raise SourceConflict("duplicate projection keys -- stop")
    plan = {"version": VERSION, "source_experiment_id": exp.experiment_id, "counts": counts, "rows": rows,
            "models": models, "datasets": sorted(datasets.values(), key=lambda d: (d["name"], d["digest"]))}
    plan["fingerprint"] = sha(canon({k: plan[k] for k in ("version", "rows", "models", "datasets")}))
    return plan


def check_expected(plan: dict) -> None:
    c = plan["counts"]
    drift = {k: (c[k], v) for k, v in EXPECTED.items() if c[k] != v}
    if drift or len(c["metric_names"]) > 9:
        raise SourceConflict(f"plan drifted from the approved scope {drift} names={len(c['metric_names'])} -- stop")


# ------------------------------------------------------------------ snapshots

def inventory(client, store: Path, experiment: str) -> dict:
    """Original experiment state: run data plus artifact files (path, size, mtime)."""
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


# ------------------------------------------------------------------ apply

def _summary_tags(row: dict, plan_fp: str, model_id: str | None) -> dict:
    t = {k: str(row[k]) for k in ("projection_key", "family", "horizon", "arm", "seed", "arm_kind", "period",
                                  "cohort", "original_run_id", "original_parent_run_id", "original_source_key",
                                  "original_fingerprint", "dataset_name", "dataset_digest", "truth_definition")}
    t.update({"record_kind": "summary_row", "summary_plan_fingerprint": plan_fp, "summary_version": VERSION,
              "original_run_link": f"#/experiments/{{src}}/runs/{row['original_run_id']}",
              "na_metrics": ",".join(sorted(row["na"])) or "none",
              "execution": "projection of a historical import; no fit or rescoring",
              "mlflow_timestamps": "MLflow times are projection times, not source fit times"})
    if model_id:
        t["model_id"] = model_id
        t["registered_model"] = f"ipcch.{row['family']}.h{row['horizon']}.{row['arm']}"
    return t


def _row_doc(row: dict, plan: dict, model_id: str | None) -> bytes:
    ds = next(d for d in plan["datasets"] if d["name"] == row["dataset_name"] and d["digest"] == row["dataset_digest"])
    doc = {"projection_key": row["projection_key"], "values": row["values"], "value_sources": row["value_sources"],
           "na": row["na"], "dataset": {"name": ds["name"], "digest": ds["digest"], "full_sha256": ds["full_sha256"],
                                        "descriptor": ds["descriptor"]},
           "model_id": model_id, "links": {k: row[k] for k in ("original_run_id", "original_parent_run_id",
                                                                 "prediction_artifact", "original_view_artifact")},
           "plan_fingerprint": plan["fingerprint"]}
    return json.dumps(doc, indent=1, sort_keys=True).encode()


def _dataset_entity(ds: dict):
    from mlflow.entities import Dataset
    d = ds["descriptor"]
    schema = {"mlflow_colspec": [{"name": "admin_code", "type": "long"}, {"name": "target_ord", "type": "long"},
                                 {"name": "target_month", "type": "string"}, {"name": "phase_truth", "type": "long"},
                                 {"name": "q3_truth", "type": "double"}]}
    source = {"authority": "saved predictions of imported IPCCH runs (evaluation keys and truth only)",
              "descriptor_sha256": ds["full_sha256"], "horizon_months": d["horizon_months"], "period": d["period"],
              "cohort": d["cohort"]}
    return Dataset(name=ds["name"], digest=ds["digest"], source_type="ipcch_evaluation_keys",
                   source=json.dumps(source, sort_keys=True), schema=json.dumps(schema), profile=json.dumps({"num_rows": d["num_rows"]}))


def _input_tags(row: dict, ds: dict):
    from mlflow.entities import InputTag
    return [InputTag("mlflow.data.context", "evaluation"), InputTag("descriptor_sha256", ds["full_sha256"]),
            InputTag("keys_sha256", ds["descriptor"]["keys_sha256"]), InputTag("truth_sha256", ds["descriptor"]["truth_sha256"]),
            InputTag("prediction_artifact", row["prediction_artifact"]), InputTag("original_run_id", row["original_run_id"]),
            InputTag("historical_import", "true")]


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


def _summary_experiment(client, plan: dict, description: str, artifact_location: str | None = None):
    exp = client.get_experiment_by_name(SUMMARY_EXPERIMENT)
    if exp is None:
        eid = client.create_experiment(SUMMARY_EXPERIMENT, artifact_location=artifact_location, tags={"summary_plan_fingerprint": plan["fingerprint"],
                                                                "catalog_status": "incomplete",
                                                                "mlflow.note.content": description})
        return client.get_experiment(eid)
    fp = exp.tags.get("summary_plan_fingerprint")
    if fp != plan["fingerprint"]:
        raise SourceConflict(f"{SUMMARY_EXPERIMENT} belongs to plan {fp}; this plan is {plan['fingerprint']} -- stop")
    return exp


def _existing_models(client, exp_id: str) -> dict:
    out, token = {}, None
    while True:
        page = client.search_logged_models([exp_id], max_results=500, page_token=token)
        for m in page:
            k = m.tags.get("projection_key")
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


def model_tags(m: dict, plan_fp: str) -> dict:
    t = {k: str(m[k]) for k in ("projection_key", "family", "horizon", "arm", "seed", "arm_kind", "original_run_id",
                                "source_parent_run_id", "model_bundle_uri", "model_bundle_sha256",
                                "member_manifest_uri", "source_code_uri", "code_commit", "features", "maps",
                                "fit_protocol")}
    t.update({"catalog": "external", "inference": "not loadable (external descriptor)",
              "bundle_note": EXTERNAL_NOTE, "summary_plan_fingerprint": plan_fp,
              "mlflow.note.content": f"{m['family']} H{m['horizon']} {m['arm']} seed {m['seed']} ({m['arm_kind']}). "
                                     f"{EXTERNAL_NOTE} Source run = original IPCCH evaluation view."})
    return t


def apply_plan(client, plan: dict, store: Path, journal: Journal, fail_after: str | None = None,
               artifact_location: str | None = None) -> dict:
    import mlflow
    from mlflow.entities import DatasetInput, LoggedModelInput, Metric, RunTag
    stats = Counter()
    exp = _summary_experiment(client, plan, summary_description(plan), artifact_location)
    eid = exp.experiment_id
    client.set_experiment_tag(eid, "catalog_status", "incomplete")
    # registered models + external models + versions
    existing = _existing_models(client, eid)
    model_ids = {}
    by_name = defaultdict(list)
    for m in plan["models"]:
        by_name[m["registered_model"]].append(m)
    for name, ms in sorted(by_name.items()):
        rm_tags = {"family": ms[0]["family"], "horizon": ms[0]["horizon"], "arm": ms[0]["arm"],
                   "arm_kind": ms[0]["arm_kind"], "catalog": "external", "summary_plan_fingerprint": plan["fingerprint"]}
        try:
            rm = client.get_registered_model(name)
            if {k: rm.tags.get(k) for k in rm_tags} != {k: str(v) for k, v in rm_tags.items()}:
                raise SourceConflict(f"registered model {name} exists with different tags -- stop")
        except SourceConflict:
            raise
        except Exception:
            client.create_registered_model(name, tags=rm_tags, description=(
                f"{ms[0]['family']} H{ms[0]['horizon']} {ms[0]['arm']} ({ms[0]['arm_kind']}). Versions are seeds/"
                f"source runs of the historical import; version numbers are MLflow allocation. {EXTERNAL_NOTE}"))
            stats["registered_models_created"] += 1
        journal.record("registered_model", name, name)
        versions = {v.tags.get("projection_key"): v for v in client.search_model_versions(f"name='{name}'")}
        for m in sorted(ms, key=lambda x: x["seed"]):
            tags = model_tags(m, plan["fingerprint"])
            lm = existing.get(m["projection_key"])
            if lm is None:
                lm = mlflow.create_external_model(name=m["logged_model_name"], source_run_id=m["original_run_id"],
                                                  tags=tags, params={"seed": m["seed"], "horizon_months": m["horizon"],
                                                                     "arm": m["arm"]},
                                                  model_type=f"historical-{m['arm_kind']}", experiment_id=eid)
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
                mv = client.create_model_version(name, source=f"models:/{lm.model_id}", model_id=lm.model_id,
                                                 tags={"projection_key": m["projection_key"], "seed": m["seed"],
                                                       "original_run_id": m["original_run_id"],
                                                       "summary_plan_fingerprint": plan["fingerprint"]},
                                                 description=f"seed {m['seed']}; {EXTERNAL_NOTE}")
                stats["model_versions_created"] += 1
                journal.record("model_version", m["projection_key"], f"{name}/{mv.version}")
            else:
                journal.record("model_version", m["projection_key"], f"{name}/{versions[m['projection_key']].version}")
    if fail_after == "models":
        raise import_runs.Interrupt("injected interruption after models")
    # summary rows
    have = {}
    for r in client.search_runs([eid], max_results=5000, run_view_type=3):
        k = r.data.tags.get("projection_key")
        if k in have:
            raise SourceConflict(f"two summary rows share projection key {k} -- stop")
        have[k] = r
    ds_by_key = {(d["name"], d["digest"]): d for d in plan["datasets"]}
    for i, row in enumerate(plan["rows"]):
        if fail_after == "row-5" and i == 5:
            raise import_runs.Interrupt("injected interruption during rows")
        mid = model_ids.get(row["model_key"]) if row["model_key"] else None
        tags = _summary_tags(row, plan["fingerprint"], mid)
        tags["original_run_link"] = tags["original_run_link"].replace("{src}", plan["source_experiment_id"])
        run = have.get(row["projection_key"])
        if run is None:
            run = client.create_run(eid, run_name=f"{row['family']} H{row['horizon']} {row['arm']} s{row['seed']} "
                                                   f"{row['period']}.{row['cohort']}",
                                    tags={"projection_key": row["projection_key"], "summary_status": "in_progress",
                                          "summary_plan_fingerprint": plan["fingerprint"]})
            stats["rows_created"] += 1
        elif run.data.tags.get("summary_plan_fingerprint") != plan["fingerprint"]:
            raise SourceConflict(f"{row['projection_key']}: summary row of another plan -- stop")
        elif run.data.tags.get("summary_status") == "complete":
            stats["rows_noop"] += 1
            journal.record("summary_run", row["projection_key"], run.info.run_id)
            continue
        rid = run.info.run_id
        journal.record("summary_run", row["projection_key"], rid)
        run = client.get_run(rid)
        for k, v in tags.items():
            if k in run.data.tags and run.data.tags[k] != v and k != "summary_status":
                raise SourceConflict(f"{row['projection_key']}: tag {k} differs -- stop")
        new_tags = [RunTag(k, v) for k, v in sorted(tags.items()) if run.data.tags.get(k) != v]
        if new_tags:
            client.log_batch(rid, tags=new_tags)
        ds = ds_by_key[(row["dataset_name"], row["dataset_digest"])]
        ts = int(time.time() * 1000)
        mets = []
        for k, v in row["values"].items():
            if k in run.data.metrics:
                if run.data.metrics[k] != v:
                    raise SourceConflict(f"{row['projection_key']}: metric {k} differs -- stop")
                continue
            mets.append(Metric(k, float(v), ts, 0, model_id=mid, dataset_name=ds["name"], dataset_digest=ds["digest"]))
        if mets:
            client.log_batch(rid, metrics=mets)
            stats["metrics_logged"] += len(mets)
        di = {(d.dataset.name, d.dataset.digest) for d in run.inputs.dataset_inputs}
        mi = {x.model_id for x in (run.inputs.model_inputs or [])}
        want_ds = (ds["name"], ds["digest"]) not in di
        want_m = bool(mid) and mid not in mi
        if want_ds or want_m:
            client.log_inputs(rid, datasets=[DatasetInput(_dataset_entity(ds), tags=_input_tags(row, ds))] if want_ds else None,
                              models=[LoggedModelInput(mid)] if want_m else None)
            stats["inputs_logged"] += 1
        doc = _row_doc(row, plan, mid)
        arts = import_runs.existing_artifacts(client, rid)
        import_runs.upload(client, rid, arts, "summary/row.json", data=doc)
        verify_row(client, client.get_run(rid), row, plan, mid, deep=True)
        client.set_tag(rid, "summary_status", "complete")
        client.set_terminated(rid, "FINISHED")
    return {"experiment_id": eid, **stats}


def summary_description(plan: dict) -> str:
    c = plan["counts"]
    return (
        f"Compact view of the accepted IPCCH imports: {c['rows']} rows, one per original evaluation view x saved own "
        "panel (period/cohort are tags). Metrics: up to nine per row (binary accuracy/precision/recall/f1/f2, "
        "four_class accuracy/macro_f1, q3_r2_projected, n), copied from the original records; undefined values are "
        "absent and listed in tag na_metrics and summary/row.json. Gaps: split2024 H12 main has no saved panel; the "
        "window probe has new_local_support only at H1/H6 and reports selected_dates (11 dates), not full periods. "
        "Comparison rule: compare rows only with equal dataset_name AND dataset_digest (same keys and truth); "
        "persistence exists only on E_persist-type cohorts. No global best-model ranking. Start with a filter such as "
        "tags.period = 'main' AND tags.cohort = 'E_all' AND tags.horizon = '1' AND tags.seed IN ('42','none'). "
        "Models tab: external catalog descriptors only (not loadable). Detailed reports, deltas, intervals and "
        "diagnostics stay in the original IPCCH experiment (tag original_run_id). Times are projection times.")


# ------------------------------------------------------------------ verify

def verify_row(client, run, row: dict, plan: dict, mid: str | None, deep: bool) -> None:
    key = row["projection_key"]
    if run.data.metrics != {k: float(v) for k, v in row["values"].items()}:
        raise SourceConflict(f"{key}: metric readback differs")
    tags = _summary_tags(row, plan["fingerprint"], mid)
    tags["original_run_link"] = tags["original_run_link"].replace("{src}", plan["source_experiment_id"])
    for k, v in tags.items():
        if run.data.tags.get(k) != v:
            raise SourceConflict(f"{key}: tag {k} readback differs")
    ds = [(d.dataset.name, d.dataset.digest) for d in run.inputs.dataset_inputs]
    if ds != [(row["dataset_name"], row["dataset_digest"])]:
        raise SourceConflict(f"{key}: dataset inputs {ds}")
    ms = [x.model_id for x in (run.inputs.model_inputs or [])]
    if ms != ([mid] if mid else []):
        raise SourceConflict(f"{key}: model inputs {ms}")
    for k in row["values"]:
        hist = client.get_metric_history(run.info.run_id, k)
        if len(hist) != 1:
            raise SourceConflict(f"{key}: metric {k} has {len(hist)} history entries")
    if deep:
        got = import_runs._download_bytes(client, run.info.run_id, "summary/row.json")
        if got != _row_doc(row, plan, mid):
            raise SourceConflict(f"{key}: summary/row.json readback differs")


def verify_all(client, plan: dict, store: Path, before: dict | None, cfg: dict) -> dict:
    exp = client.get_experiment_by_name(SUMMARY_EXPERIMENT)
    if exp is None or exp.tags.get("summary_plan_fingerprint") != plan["fingerprint"]:
        raise SourceConflict("summary experiment missing or of another plan")
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
        names[m["registered_model"]].add(m["projection_key"])
        out["logged_models"] += 1
    for name, keys in names.items():
        vs = client.search_model_versions(f"name='{name}'")
        got = Counter(v.tags.get("projection_key") for v in vs)
        if set(got) != keys or any(n != 1 for n in got.values()):
            raise SourceConflict(f"{name}: versions {dict(got)} != plan")
        for v in vs:
            if v.source != f"models:/{models[v.tags['projection_key']].model_id}":
                raise SourceConflict(f"{name} v{v.version}: source differs")
        out["registered_models"] += 1
        out["model_versions"] += len(vs)
    runs = client.search_runs([eid], max_results=5000, run_view_type=3)
    by_key = defaultdict(list)
    for r in runs:
        by_key[r.data.tags.get("projection_key")].append(r)
    if set(by_key) != {r["projection_key"] for r in plan["rows"]} or any(len(v) != 1 for v in by_key.values()):
        raise SourceConflict("summary run set differs from plan or has duplicates")
    dsets = set()
    for row in plan["rows"]:
        run = client.get_run(by_key[row["projection_key"]][0].info.run_id)   # search omits model inputs
        if run.data.tags.get("summary_status") != "complete":
            raise SourceConflict(f"{row['projection_key']}: incomplete")
        mid = models[row["model_key"]].model_id if row["model_key"] else None
        verify_row(client, run, row, plan, mid, deep=True)
        dsets.add((row["dataset_name"], row["dataset_digest"]))
        out["rows"] += 1
        out["metrics"] += len(row["values"])
    out["dataset_descriptors"] = len(dsets)
    out["distinct_metric_names"] = len({k for r in runs for k in r.data.metrics})
    if before is not None:
        after = inventory(client, store, cfg["experiment"])
        if after["sha256"] != before["sha256"]:
            changed = [k for k in set(before["runs"]) | set(after["runs"]) if before["runs"].get(k) != after["runs"].get(k)]
            raise SourceConflict(f"original experiment changed: {len(changed)} runs, e.g. {changed[:3]}")
        out["original_runs_unchanged"] = len(after["runs"])
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
    ap.add_argument("--inventory", help="apply/verify: pre-apply inventory JSON of the original experiment")
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
    sdir = store / "summary"
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
                        raise SourceConflict("original experiment changed since the inventory -- stop")
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
