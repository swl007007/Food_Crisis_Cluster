"""Read-only IPCCH model catalog lookup (logged and registered models) for people and the UI assistant.

The mlflow CLI has no model commands; this fills the gap without any write path. It uses the normal tracking
client, so it sees exactly what the logged-in user may read (the assistant runs it as the READ-only user).

  ipcch-model families                      family / arm / lead overview of both dashboards
  ipcch-model search [--experiment NAME|ID] [--family F] [--arm A] [--lead L] [--feature-set FS] [--max-results N]
  ipcch-model get NAME_OR_WORDS_OR_ID       full model card (description), tags, params, registered version;
                                            words from the name are enough ("q3 optimization share_history_residual 3-month")
  ipcch-model registered NAME               registered model description and its versions
  ipcch-model features NAME_OR_MODEL_ID     full ordered input-feature list (from the card's archived source)
"""
import argparse
import csv
import gzip
import io
import json
import os
import sys
from pathlib import Path

DASHBOARDS = ("IPCCH dashboard", "IPCCH forecasting dashboard")
# local feature-list files may only come from these roots (the tag is written by the catalog admin)
LOCAL_ROOTS = (Path("/home/swl007007/ipcch-alphaearth-runs"), Path("/home/swl007007/.local/share/ipcch-mlflow/artifacts"),
               Path("/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/2.source_code/Step5_Geo_RF_trial"))
FEATURE_TAG = "card.feature_list"
SHOW_TAGS = ("project", "family", "family_title", "arm", "arm_role", "role", "lead_months", "seed", "snapshot", "stage",
             "model_scope", "model_type", "feature_set", "n_features", "target", "status", "features", "maps",
             "training_window")


def client():
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from mlflow.tracking import MlflowClient
    return MlflowClient()


def experiment_ids(c, which):
    names = [which] if which else DASHBOARDS
    out = []
    for n in names:
        e = c.get_experiment(n) if str(n).isdigit() else c.get_experiment_by_name(n)
        if e is None:
            sys.exit(f"unknown experiment {n!r}")
        out.append(e.experiment_id)
    return out


def _norm(text: str) -> list:
    return [t for t in text.lower().replace("|", " ").replace("-month", " month").split() if t]


def find_model(c, key):
    if key.startswith("m-"):
        return c.get_logged_model(key)
    models = c.search_logged_models(experiment_ids=experiment_ids(c, None), max_results=1000)
    exact = [m for m in models if m.name == key]
    if exact:
        return exact[0]
    words = _norm(key)                      # every word must appear in the name, family or arm tag
    hits = [m for m in models
            if all(w in _norm(m.name) + _norm(m.tags.get("family", "").replace("_", " ")) + [m.tags.get("family", ""), m.tags.get("arm", "")]
                   or w.lstrip("0") in [t.lstrip("0") for t in _norm(m.name)] for w in words)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        sys.exit(f"no model matches {key!r}; run `ipcch-model families` for family/arm names")
    names = sorted(m.name for m in hits)
    sys.exit(f"{len(hits)} models match {key!r}; repeat with the exact name, e.g.:\n  " + "\n  ".join(names[:25]))


def cmd_families(c, a):
    from collections import defaultdict
    seen = defaultdict(lambda: defaultdict(set))
    titles = {}
    for m in c.search_logged_models(experiment_ids=experiment_ids(c, None), max_results=1000):
        f = m.tags.get("family", "?")
        titles[f] = (m.tags.get("project", ""), m.name.split(" | ")[0])
        seen[f][m.tags.get("arm", "?")].add(m.tags.get("lead_months", ""))
    for f in sorted(seen):
        print(f"{titles[f][0]} | family={f} ({titles[f][1]})")
        for arm, leads in sorted(seen[f].items()):
            print(f"    arm={arm} leads={','.join(sorted(leads))}")


def cmd_search(c, a):
    lead = f"{int(a.lead):02d}" if a.lead else None          # tags store leads as two digits
    filters = [f"tags.{k} = '{v}'" for k, v in (("family", a.family), ("arm", a.arm), ("lead_months", lead),
                                                ("feature_set", a.feature_set)) if v]
    models = c.search_logged_models(experiment_ids=experiment_ids(c, a.experiment),
                                    filter_string=" AND ".join(filters) or None, max_results=a.max_results)
    cols = ("project", "family", "arm", "lead_months", "feature_set", "n_features", "role")
    print("name | " + " | ".join(cols))
    for m in sorted(models, key=lambda m: m.name):
        print(m.name + " | " + " | ".join(m.tags.get(k, "") for k in cols))
    print(f"({len(models)} models)")


def cmd_get(c, a):
    m = find_model(c, a.key)
    out = {"name": m.name, "model_id": m.model_id, "experiment_id": m.experiment_id, "source_run_id": m.source_run_id,
           "tags": {k: m.tags[k] for k in SHOW_TAGS if k in m.tags},
           "params": dict(m.params or {}), "registered": json.loads(m.tags.get("mlflow.modelVersions", "[]"))}
    print(json.dumps(out, indent=1))
    print("\n" + m.tags.get("mlflow.note.content", "(no description)"))


def cmd_registered(c, a):
    rm = c.get_registered_model(a.name)
    print(json.dumps({"name": rm.name, "tags": dict(rm.tags)}, indent=1))
    print("\n" + (rm.description or "(no description)") + "\n")
    for v in sorted(c.search_model_versions(f"name='{a.name}'"), key=lambda v: int(v.version)):
        print(f"version {v.version}: {v.source} | {v.description or ''}")


def read_columns(data: bytes, gz: bool, column: str | None) -> list:
    text = gzip.decompress(data).decode("utf-8-sig") if gz else data.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    where = []
    if column and "?" in column:            # name?col=value&col~N keeps matching rows (= equals, ~ JSON list contains N)
        column, _, cond = column.partition("?")
        for c in cond.split("&"):
            op = "~" if "~" in c else "="
            where.append((*c.split(op, 1), op))
    if column is None:
        return [r[0] for r in rows]
    if column == "@header":                 # a wide table: the feature names are its columns after the key
        return rows[0][1:]
    i = rows[0].index(column)
    def keep(r):
        for col, val, op in where:
            cell = r[rows[0].index(col)].strip()
            if op == "=" and cell.lower() != val.lower():
                return False
            if op == "~" and int(val) not in json.loads(cell or "[]"):
                return False
        return True
    return [r[i] for r in rows[1:] if keep(r)]


def cmd_features(c, a):
    m = find_model(c, a.key)
    spec = m.tags.get(FEATURE_TAG)
    if not spec:
        sys.exit(f"{m.name}: no archived ordered feature list (see the feature groups in the model card)")
    for part in spec.split(" ; "):          # one entry per input block, e.g. compact296 ; alphaearth128
        label, _, loc = part.partition("=")
        loc, _, column = loc.partition("#")
        if loc.startswith("runs:/"):
            run_id, path = loc[len("runs:/"):].split("/", 1)
            data = Path(c.download_artifacts(run_id, path, dst_path=os.environ.get("TMPDIR", "/tmp"))).read_bytes()
        else:
            path, _, member = loc.partition("!")     # file.tar!member reads one member of an archived tar
            p = Path(path).resolve()
            if not any(p.is_relative_to(r) for r in LOCAL_ROOTS):
                sys.exit(f"refusing feature list outside the catalog roots: {p}")
            if member:
                import tarfile
                with tarfile.open(p) as tar:
                    data = tar.extractfile(member).read()
            else:
                data = p.read_bytes()
        names = read_columns(data, loc.split("!")[-1].endswith(".gz"), column or None)
        print(f"# {label}: {len(names)} features (source {loc}{'#' + column if column else ''})")
        print("\n".join(names))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ipcch-model", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("families")
    s = sub.add_parser("search")
    s.add_argument("--experiment")
    s.add_argument("--family")
    s.add_argument("--arm")
    s.add_argument("--lead")
    s.add_argument("--feature-set")
    s.add_argument("--max-results", type=int, default=500)
    sub.add_parser("get").add_argument("key")
    sub.add_parser("registered").add_argument("name")
    sub.add_parser("features").add_argument("key")
    a = ap.parse_args(argv)
    c = client()
    {"families": cmd_families, "search": cmd_search, "get": cmd_get, "registered": cmd_registered, "features": cmd_features}[a.cmd](c, a)


if __name__ == "__main__":
    main()
