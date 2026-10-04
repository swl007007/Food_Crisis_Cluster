"""Pre-fit reconciliation of a prepared interruption run (launch gate; reads metadata, keys and
masks only; no fit). Usage: python reconcile_prepared.py RUN_DIR LEDGER_SHA ALIGNMENT_SHA

Checks KEYS, MASKS and WEIGHTS only (derived hidden-feature values rely on the existing synthetic
tests): accepted preparation (code/runtime/outputs), committed ledger/alignment bytes, pinned sources,
108 scenario inputs and the pinned feature order (exactly 129), the exact 648 schedule, and in every
input: eval keys inside [O-59, O), no fitting/eval label in the outer hidden cycles, labelled E3
targets only, per original key 1 (A) or 3 (B) variant rows whose weights sum to 1, fit/eval key
agreement. The Stage 1 N is descriptive only; the certified total fit bound stays the conservative
N <= 5714 (mapped universe 5718 gives the same floor) until every scheduled population narrows it.
"""
EXPECTED_FEATURES = 129
CONSERVATIVE_N = 5714
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[5] / "FEWSNETGeoXGBExperiment"
sys.path.insert(0, str(PKG))
import pandas as pd  # noqa: E402

from scripts import run_stage1 as s1  # noqa: E402
from src.experiment import availability as av, plan  # noqa: E402
from src.feature import fourclass_features as ff  # noqa: E402
from src.utils import acceptance as acc  # noqa: E402
from src.utils.run_identity import file_sha256  # noqa: E402

run, ledger_sha, align_sha = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
prep = run / "prepared"
out, problems = {}, []
identity = acc.accept_prepared(run)
out["prepared_outputs_sha256"] = identity["outputs_sha256"]
out["git_head"] = identity.get("git_head")
out["code_equals_git_head"] = identity.get("code_equals_git_head")
if file_sha256(prep / "manifests" / "release_ledger.csv") != ledger_sha:
    problems.append("prepared ledger differs from the committed ledger")
if file_sha256(prep / "manifests" / "alignment.json") != align_sha:
    problems.append("prepared alignment differs from the committed alignment")
out["sources"] = {k: v["sha256"] for k, v in json.loads((prep / "manifests" / "sources.json").read_text())
                  ["sources"].items()}
schema = ff.load_schema(PKG / "feature-schema.json")
alignment = json.loads((prep / "manifests" / "alignment.json").read_text())
features = json.loads((prep / "scenario" / "features.json").read_text())["ordered_features"]
if features != ff.aligned_feature_names(schema, alignment):
    problems.append("pinned scenario features differ from the alignment")
if len(features) != EXPECTED_FEATURES:
    problems.append(f"{len(features)} features, expected {EXPECTED_FEATURES}")
out["features"] = len(features)
schedule = json.loads((prep / "manifests" / "schedule.json").read_text())
entries = s1.scenario_entries(schedule)
out["schedule_candidates"] = len(entries)
inputs = sorted({e["input"] for e in entries})
present = sorted(p.name for p in (prep / "scenario").glob("*.parquet"))
out["inputs_expected"], out["inputs_present"] = len(inputs), len(present)
if inputs != present:
    problems.append(f"scenario inputs differ: missing {sorted(set(inputs) - set(present))[:3]}, "
                    f"extra {sorted(set(present) - set(inputs))[:3]}")
ledger = av.ReleaseLedger(pd.read_csv(prep / "manifests" / "release_ledger.csv", dtype=str), real=True)
n_max, per_input = 0, {}
for name in inputs:
    f = pd.read_parquet(prep / "scenario" / name)
    strategy, h, t, k = f["strategy"].iloc[0], int(f["horizon"].iloc[0]), name.split("_")[2], int(f["scenario_k"].iloc[0])
    o = ff.month_index([f"{t}-01"])[0] - h
    fit, ev, tgt = (f[f["role"] == r] for r in ("fit_variant", "eval", "target"))
    hidden = ledger.hidden(o, k)
    bad = []
    if [c for c in f.columns if c in features] != features:
        bad.append("feature order")
    if len(ev) and (ev["target_month"].min() < o - plan.WINDOW or ev["target_month"].max() >= o):
        bad.append("eval key outside [O-59, O)")
    if f.loc[f["role"] != "target", "target_month"].isin(list(hidden)).any():
        bad.append("label in an outer hidden cycle")
    if tgt["class_code"].isna().any() or (tgt["target_month"] != o + h).any():
        bad.append("E3 target rows not labelled at T")
    per_key = fit.groupby("orig_key").agg(n=("weight", "size"), w=("weight", "sum"))
    if (per_key["n"] != (1 if strategy == "A" else 3)).any() or not ((per_key["w"] - 1).abs() < 1e-9).all() \
            or set(fit["orig_key"]) != set(ev["orig_key"]) or ev["orig_key"].duplicated().any():
        bad.append("per-key variant count / weight sum / key agreement")
    if bad:
        problems.append(f"{name}: {bad}")
    n_areas = int(ev["area"].nunique())
    n_max = max(n_max, n_areas)
    per_input[name] = {"eval_keys": int(len(ev)), "fit_rows": int(len(fit)), "targets": int(len(tgt)),
                       "areas": n_areas, "hidden": [ff.month_label([m])[0] for m in sorted(hidden)]}
out["lawful_N_max_areas_stage1_inputs_descriptive"] = n_max
out["certified_fit_ceiling"] = 40824 + 931 * (1 + CONSERVATIVE_N // 50)   # 147,889 (conservative N)
out["per_input"] = per_input
out["problems"] = problems
print(json.dumps({k: v for k, v in out.items() if k != "per_input"}, indent=1))
(run.parent / f"{run.name}.reconcile.json").write_text(json.dumps(out, indent=1))
sys.exit(1 if problems else 0)
