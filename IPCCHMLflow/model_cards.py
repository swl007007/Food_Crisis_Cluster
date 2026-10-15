"""AI-readable model cards and human tags for every external model in the two dashboards.

Facts live in ``model_cards.json`` (one entry per family, optional per-arm overrides; every fact cites its source,
anything not recorded says so). From them this module builds, per logged model:

- ``card(...)``: the logged-model description (``mlflow.note.content``, <= 8000 chars), a sectioned key: value
  text meant for the UI assistant (environment, data, features, feature engineering, architecture, settings,
  evaluation, status, provenance);
- ``registered_card(...)``: the registered-model description (<= 5000 chars, family x arm x lead core);
- ``human_tags(...)``: a uniform tag set for people to tell models apart (project, model_type, feature_set,
  n_features, target, role, status, scope) plus ``card.feature_list`` for ``ipcch-model features``.

``summary_catalog`` uses the same functions for the IPCCH dashboard, so creation and verification agree.

  model_cards.py plan  [--out PLAN.json]                diff of descriptions/tags against the live store (read-only)
  model_cards.py apply --plan-fingerprint FP [--only NAME ...]   write exactly that plan (descriptions + tags)
  model_cards.py verify --plan-fingerprint FP           read back
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FACTS = HERE / "model_cards.json"
LOGGED_LIMIT, REGISTERED_LIMIT = 8000, 5000
NOTE = "mlflow.note.content"
DASHBOARDS = {"IPCCH dashboard": "IPCCH", "IPCCH forecasting dashboard": "IPCCH forecasting"}
HUMAN_KEYS = ("project", "model_type", "feature_set", "n_features", "target", "role", "status", "scope")
FEATURE_TAG = "card.feature_list"
CARD_VERSION = "model-card-v1"


def load_facts(path: Path = FACTS) -> dict:
    return json.loads(path.read_text())["families"]


def _merged(facts: dict, family: str, arm: str) -> dict:
    f = facts.get(family)
    if f is None:
        raise KeyError(f"model_cards.json has no family {family!r}")
    a = f.get("arms", {}).get(arm)
    if a is None:
        raise KeyError(f"model_cards.json: family {family!r} has no arm {arm!r}")
    out = {k: v for k, v in f.items() if k != "arms"}
    out["features"] = {**f.get("features", {}), **a.get("features", {})}
    out.update({k: v for k, v in a.items() if k != "features"})
    out["arm"], out["family"] = arm, family
    return out


def _per_lead(value, lead: str | None):
    """Facts may vary by lead: a dict keyed by lead in months ("3"), or a string with {H}."""
    h = str(int(lead)) if lead not in (None, "") else None
    if isinstance(value, dict):
        if h is None or h not in value:
            return " / ".join(f"{v} ({k}-month)" for k, v in value.items()) if h is None else value.get("default", "")
        return value[h]
    if isinstance(value, str) and "{H}" in value and h is not None:
        return value.replace("{H}", h)
    return value


def human_tags(facts: dict, family: str, arm: str, lead: str | None = None) -> dict:
    m = _merged(facts, family, arm)
    feats = m["features"]
    tags = {"project": m["project"], "model_type": m["model_type"],
            "feature_set": str(_per_lead(feats["feature_set"], lead)),
            "n_features": str(_per_lead(feats["n_features"], lead)), "target": m["target"], "role": m["role"],
            "status": m["status_tag"], "scope": m["scope"]}
    if feats.get("list") and lead not in (None, ""):
        tags[FEATURE_TAG] = _per_lead(feats["list"], lead)
    return tags


def _sections(m: dict, ident: str, extra: dict, catalog: str = "", lead: str | None = None) -> list:
    feats = m["features"]
    sec = [
        ("card", f"{CARD_VERSION}; audience: AI assistant; generated from IPCCHMLflow/model_cards.json; "
                 "'not recorded' means the source did not record it."),
        ("identity", ident),
        ("summary", f"{m['one_line']} Arm {m['arm']} ({m['role']}): {m['meaning']}"),
        ("status", m["status"]),
        *([("catalog", catalog)] if catalog else []),
        ("data", m["data"]),
        ("features", f"feature_set={_per_lead(feats['feature_set'], lead)}; n_features={_per_lead(feats['n_features'], lead)}; "
                     f"groups: {feats['groups']}"
                     + (f"; full ordered list: `ipcch-model features \"<this model name>\"`" if feats.get("list") else
                        "; full ordered list: not archived")),
        ("feature_engineering", m["feature_engineering"]),
        ("architecture", m["architecture"] + (f" Arm difference: {m['differences']}" if m.get("differences") else "")),
        ("settings", m["settings"] + (f" At this lead: {_per_lead(m['lead_settings'], lead)}"
                                      if m.get("lead_settings") and lead not in (None, "") else "")),
        ("environment", m["environment"]),
        ("evaluation", m["evaluation"]),
    ]
    if extra:
        sec.append(("observed", "; ".join(f"{k}={v}" for k, v in extra.items())))
    sec.append(("sources", m["sources"]))
    return sec


def _fit(sections: list, limit: int) -> str:
    """Join sections; if too long, shorten the longest sections (never drop identity/status) and say so."""
    def render(secs):
        return "\n".join(f"## {k}\n{v}" for k, v in secs)
    secs = [list(s) for s in sections]
    text = render(secs)
    while len(text) > limit:
        longest = max((s for s in secs if s[0] not in ("card", "identity", "status")), key=lambda s: len(s[1]))
        over = len(text) - limit + 40
        longest[1] = longest[1][: max(80, len(longest[1]) - over)].rstrip() + " [truncated]"
        text = render(secs)
    return text


def card(facts: dict, family: str, arm: str, name: str, ident_tags: dict, extra: dict | None = None,
         catalog: str = "") -> str:
    m = _merged(facts, family, arm)
    ident = f"model={name}; project={m['project']}; family={family} ({m['title']}); arm={arm}; " + \
            "; ".join(f"{k}={v}" for k, v in ident_tags.items() if v)
    return _fit(_sections(m, ident, extra or {}, catalog, ident_tags.get("lead_months")), LOGGED_LIMIT)


def registered_card(facts: dict, family: str, arm: str, name: str, lead: str, catalog_note: str = "",
                    catalog: str = "") -> str:
    m = _merged(facts, family, arm)
    ident = f"registered_model={name}; project={m['project']}; family={family} ({m['title']}); arm={arm}; lead_months={lead}"
    secs = _sections(m, ident, {}, catalog, lead)
    feats = m["features"]
    short = {"features": f"feature_set={_per_lead(feats['feature_set'], lead)}; n_features={_per_lead(feats['n_features'], lead)}; "
                         "groups, ordered list and feature engineering: see the version's logged-model card."}
    secs = [(k, short.get(k, v)) for k, v in secs if k not in ("environment", "sources", "feature_engineering", "evaluation")]
    secs.append(("versions", "each version is one seed/snapshot; its logged model holds the full card "
                             "(`ipcch-model get \"<logged model name>\"`)."))
    if catalog_note:
        secs.append(("catalog_note", catalog_note))
    return _fit(secs, REGISTERED_LIMIT)


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ live store: plan / apply / verify

FORECAST_IDENT = ("lead_months", "snapshot", "stage", "model_scope")
FORECAST_OBSERVED = ("n_inputs", "boosters_total", "boosters_new", "boosters_reused", "fit_units_new")
FORECAST_CATALOG = ("External catalog descriptor registered by ipcch-forecasting-import-v1: not loadable, no inference "
                    "wrapper, nothing retrained. Booster members are listed in the family run's manifests.")


def catalog_text(project: str, tags: dict) -> str:
    if project != "IPCCH":
        return FORECAST_CATALOG
    import summary_catalog                              # same wording as the IPCCH dashboard writer
    return summary_catalog.external_note({"model_bundle_sha256": tags.get("zz_prov.model_bundle_sha256", "")})


def _ident(project: str, tags: dict) -> dict:
    keys = ("lead_months", "seed", "window") if project == "IPCCH" else FORECAST_IDENT
    return {k: tags.get(k, "") for k in keys}


def logged_target(facts: dict, project: str, lm) -> dict:
    family, arm = lm.tags["family"], lm.tags["arm"]
    extra = {k: lm.params[k] for k in FORECAST_OBSERVED if project != "IPCCH" and k in (lm.params or {})}
    return {"description": card(facts, family, arm, lm.name, _ident(project, lm.tags), extra,
                                catalog_text(project, lm.tags)),
            "tags": human_tags(facts, family, arm, lm.tags.get("lead_months"))}


def build_plan(client, facts: dict) -> dict:
    logged, registered = [], {}
    for exp_name, project in DASHBOARDS.items():
        exp = client.get_experiment_by_name(exp_name)
        for lm in client.search_logged_models(experiment_ids=[exp.experiment_id], max_results=1000):
            t = logged_target(facts, project, lm)
            logged.append({"model_id": lm.model_id, "name": lm.name, "experiment": exp_name, **t})
            for v in json.loads(lm.tags.get("mlflow.modelVersions", "[]")):
                rm = registered.setdefault(v["name"], {"name": v["name"], "family": lm.tags["family"],
                                                       "arm": lm.tags["arm"], "lead": lm.tags.get("lead_months", ""),
                                                       "project": project,
                                                       "catalog": catalog_text(project, lm.tags)})
                if (rm["family"], rm["arm"]) != (lm.tags["family"], lm.tags["arm"]):
                    raise SystemExit(f"registered model {v['name']} spans two family/arm pairs -- stop")
    for rm in registered.values():
        current = client.get_registered_model(rm["name"])
        note = "" if rm["project"] == "IPCCH" else _catalog_note(current.description or "")
        rm["description"] = registered_card(facts, rm["family"], rm["arm"], rm["name"], rm["lead"], note,
                                            rm.pop("catalog"))
        rm["tags"] = {k: v for k, v in human_tags(facts, rm["family"], rm["arm"], rm["lead"]).items() if k != FEATURE_TAG}
    body = {"version": CARD_VERSION, "facts_sha256": fingerprint(facts),
            "logged": sorted(logged, key=lambda x: x["model_id"]),
            "registered": sorted(registered.values(), key=lambda x: x["name"])}
    body["fingerprint"] = fingerprint(body)
    return body


def _catalog_note(description: str) -> str:
    """Forecasting registered models keep their importer text once (not re-nested on refresh)."""
    if "## catalog_note\n" in description:
        return description.split("## catalog_note\n", 1)[1].strip()
    return description.strip()


def diff(client, plan: dict) -> dict:
    out = {"logged_description_changes": 0, "logged_tag_changes": 0, "registered_description_changes": 0,
           "registered_tag_changes": 0}
    for x in plan["logged"]:
        lm = client.get_logged_model(x["model_id"])
        out["logged_description_changes"] += lm.tags.get(NOTE) != x["description"]
        out["logged_tag_changes"] += any(lm.tags.get(k) != v for k, v in x["tags"].items())
    for x in plan["registered"]:
        rm = client.get_registered_model(x["name"])
        out["registered_description_changes"] += (rm.description or "") != x["description"]
        out["registered_tag_changes"] += any(rm.tags.get(k) != v for k, v in x["tags"].items())
    return out


def apply(client, plan: dict, only: set | None = None) -> dict:
    n = {"logged": 0, "registered": 0}
    for x in plan["logged"]:
        if only and x["name"] not in only:
            continue
        lm = client.get_logged_model(x["model_id"])
        want = {**x["tags"], NOTE: x["description"]}
        if any(lm.tags.get(k) != v for k, v in want.items()):
            client.set_logged_model_tags(x["model_id"], want)
            n["logged"] += 1
    names = {r for x in plan["logged"] if not only or x["name"] in only
             for r in [v["name"] for v in json.loads(client.get_logged_model(x["model_id"]).tags.get("mlflow.modelVersions", "[]"))]}
    for x in plan["registered"]:
        if x["name"] not in names:
            continue
        rm = client.get_registered_model(x["name"])
        if (rm.description or "") != x["description"]:
            client.update_registered_model(x["name"], description=x["description"])
            n["registered"] += 1
        for k, v in x["tags"].items():
            if rm.tags.get(k) != v:
                client.set_registered_model_tag(x["name"], k, v)
    return n


def main(argv=None) -> int:
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "apply", "verify"])
    ap.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    ap.add_argument("--plan-fingerprint")
    ap.add_argument("--only", action="append", help="apply: limit to these logged model names (samples)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from mlflow.tracking import MlflowClient
    client = MlflowClient(a.tracking_uri)
    plan = build_plan(client, load_facts())
    res = {"command": a.command, "plan_fingerprint": plan["fingerprint"], "logged": len(plan["logged"]),
           "registered": len(plan["registered"]),
           "max_logged_chars": max(len(x["description"]) for x in plan["logged"]),
           "max_registered_chars": max(len(x["description"]) for x in plan["registered"])}
    if a.command != "plan" and a.plan_fingerprint != plan["fingerprint"]:
        raise SystemExit(f"recomputed plan {plan['fingerprint']} != {a.plan_fingerprint} -- stop")
    if a.command == "apply":
        res["written"] = apply(client, plan, set(a.only) if a.only else None)
    res["remaining_diff"] = diff(client, plan)
    if a.out:
        Path(a.out).write_text(json.dumps({"result": res, "plan": plan}, indent=1))
    print(json.dumps(res, indent=1))
    if a.command == "verify" and any(res["remaining_diff"].values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
