"""The bounded GeoXGBoost experiment driver (experiment-plan sections 2, 4-7).

python scripts/run_experiment.py --run-dir RUN gscreen   [--workers N]
python scripts/run_stage1.py      --run-dir RUN           (between gscreen and develop)
python scripts/run_experiment.py --run-dir RUN maps      [--workers N]
python scripts/run_experiment.py --run-dir RUN develop   [--workers N]
python scripts/run_experiment.py --run-dir RUN select
python scripts/run_experiment.py --run-dir RUN oldmap    [--workers N]
python scripts/run_experiment.py --run-dir RUN freeze
python scripts/run_experiment.py --run-dir RUN final     [--workers N]

Order and information boundaries:

* gscreen  G1-G4 pooled on the 18 development folds; per H the largest main-cohort
           macro-F1 over its six folds (exact counts; ties: fewer rounds, shallower, number).
* maps     one general consensus per (L vector, strategy, origin) from candidates whose
           SCORING target T' < O, all horizons pooled; identical pools share one map.
           Plus the v7 RF candidates truncated the same way (old-map diagnostic).
* develop  the 24 schemes on the 18 development folds (shared arm, six-date gate).
* select   lexicographic rule on development predictions only; writes selection.json.
* oldmap   the selected G/L on the truncated v7 maps: independent and shared arms.
* freeze   the selected scheme's final map from all 2018-2020 candidates (cutoff 2020-12)
           and the frozen rules; written before any final fold is fitted.
* final    four XGB arms on the original 2021-2024 schedule, run once.

No phase reads a later phase's output; final folds never feed any choice.

Interruption task (10-02 design D2/D4/G4), separate phases on the scenario schedule:

python scripts/run_experiment.py --run-dir RUN scen-develop [--workers N]
python scripts/run_experiment.py --run-dir RUN scen-select
python scripts/run_experiment.py --run-dir RUN scen-freeze
python scripts/run_experiment.py --run-dir RUN scen-historical
python scripts/run_experiment.py --run-dir RUN scen-actual --actual-availability CSV --actual-scaffold JSON
python scripts/run_experiment.py --run-dir RUN scen-report [--expert-table CSV]
python scripts/run_experiment.py --run-dir RUN scen-evaluate --truth-release DIR [--expert-table CSV]

* scen-develop  72 complete development folds = A/B x H4/H8 x k 0/1/2 x six 2019-2020
                targets: per strategy and origin the common origin-legal crisis-weighted map
                (run_stage2.scenario_map_pool), then the full Stage 3 fold (ScenarioPanel,
                same-intensity gate, crisis gate, fixed G/L).
* scen-select   per H: normal-scenario parity (crisis F1 - matched persistence F1 >= -0.02),
                then the mean of the one/two-cycle F1; exact ties A; no qualifier = no winner.
* scen-freeze   per H with a winner: that strategy's final map from candidates through 2020-12;
                horizons without a winner record that no final model is released.
* scen-historical  frozen recipe/maps on the common post-freeze 2021-2024 calendar, k = 0/1/2.
* scen-actual   2025 prediction-only folds (truth never loaded): outer forecast on the real
                ledger (k=0), gate replay at the verified missed-cycle count; refuses without a
                verified country/product availability table.
* scen-report   Study1/Study2/country tables and paired country-block crisis-F1 intervals from the
                saved historical predictions (persistence on matched keys).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from src.experiment import plan  # noqa: E402
from src.experiment import stage3 as s3  # noqa: E402
from src.feature.fourclass_features import load_schema  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.utils import acceptance as acc  # noqa: E402
from src.utils.run_identity import (SCHEMA_PATH, code_identity, file_sha256, output_hashes,  # noqa: E402
                                    runtime_identity, write_json_atomic)

KEY = ["area", "target_month", "horizon"]


# ------------------------------------------------------------------------------ helpers

def features():
    return load_schema(SCHEMA_PATH)["ordered_features"]


def panel(run: Path, horizon: int) -> s3.Panel:
    return s3.Panel(run / "prepared" / f"snapshot_h{horizon}.parquet", features(), horizon)


def store(run: Path) -> s3.GlobalStore:
    return s3.GlobalStore(run / "globals")


def write_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, float_format="%.17g")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def finish(base: Path, name: str, record: dict) -> dict:
    """Completion record written last with every output hash."""
    record.update(code=code_identity(), runtime=runtime_identity(),
                  outputs={rel: sha for rel, sha in output_hashes(base).items() if rel != name})
    write_json_atomic(base / name, record)
    return record


def main_cohort(base: pd.DataFrame, horizon: int) -> pd.Series:
    has = base["persistence_code"].notna()
    if horizon in (4, 8):
        has &= base["expert_code"].notna()
    return has


def dev_baselines(run: Path) -> pd.DataFrame:
    base = read_csv(run / "prepared" / "ledgers" / "dev_baselines.csv")
    base["target_month"] = base["target_label"]
    return base


def keyed(base: pd.DataFrame, preds: pd.DataFrame, horizon: int, where: str) -> pd.DataFrame:
    """Predictions joined 1:1 to every truth key of the horizon/targets; never a subset."""
    rows = base[(base["horizon"] == horizon) & base["target_month"].isin(preds["target_month"].unique())]
    merged = rows.merge(preds, on=KEY, how="outer", indicator=True, validate="one_to_one")
    if (merged["_merge"] != "both").any():
        raise RuntimeError(f"{where}: prediction keys differ from the truth keys")
    if not (merged["truth_code"] == merged["y_true_code"]).all():
        raise RuntimeError(f"{where}: truth differs from the baseline ledger")
    return merged.drop(columns="_merge")


def exact(frame: pd.DataFrame, column: str) -> Fraction:
    """Primary exact score of the active endpoint (D26: crisis-positive F1)."""
    return fourclass.endpoint_exact(frame["truth_code"].to_numpy(int), frame[column].to_numpy(float).astype(int))


def exact_fourclass(frame: pd.DataFrame, column: str) -> Fraction:
    return fourclass.macro_f1_exact(frame["truth_code"].to_numpy(int), frame[column].to_numpy(float).astype(int))


REUSED_PREPARED = ("snapshot_h4.parquet", "snapshot_h8.parquet", "snapshot_h12.parquet", "ledgers/dev_baselines.csv")


def dev_schedule_digest(schedule: dict) -> str:
    """SHA-256 of the canonical JSON of the development section (all the G predictions use)."""
    import hashlib
    return hashlib.sha256(json.dumps(schedule["development"], sort_keys=True).encode()).hexdigest()


def source_effective_params(source: Path) -> dict:
    """The effective xgb.train parameters (XGB_BASE merged with each G) and rounds actually
    used by the source run's G fits, read from its stored global-booster records
    (globals/h{H}/{G}/O*.json). Every record of one G must agree; every H x G must exist."""
    found = {}
    for h in plan.HORIZONS:
        for g in plan.G_CONFIGS:
            records = sorted((source / "globals" / f"h{h}" / g).glob("O*.json"))
            if not records:
                raise RuntimeError(f"source run has no stored h{h} {g} global records")
            for path in records:
                rec = json.loads(path.read_text(encoding="utf-8"))
                value = (json.dumps(rec["params"], sort_keys=True), int(rec["rounds_total"]))
                if found.setdefault(g, value) != value:
                    raise RuntimeError(f"source {g} global records disagree on parameters ({path.name})")
    return found


def reused_gscreen_predictions(run: Path, source: Path):
    """The saved 72 development pooled predictions of an earlier run, reused without refit
    (D26) only if every input they depend on is identical: snapshots and development
    truth/baselines (prepared output hashes), the parsed development schedule (D27: other
    schedule sections such as the tb3 lists may differ), the producing G configs, a
    complete 6-target x H x G prediction set, and the source G screen's recorded
    predictions hash."""
    mine = json.loads((run / "prepared" / "manifests" / "outputs.json").read_text(encoding="utf-8"))
    theirs = json.loads((source / "prepared" / "manifests" / "outputs.json").read_text(encoding="utf-8"))
    differ = [f for f in REUSED_PREPARED if mine.get(f) is None or mine.get(f) != theirs.get(f)]
    if differ:
        raise RuntimeError(f"cannot reuse {source} G predictions: prepared inputs differ {differ}")
    sched_path = ("prepared", "manifests", "schedule.json")
    my_sched = json.loads(run.joinpath(*sched_path).read_text(encoding="utf-8"))
    their_sched = json.loads(source.joinpath(*sched_path).read_text(encoding="utf-8"))
    if my_sched["development"] != their_sched["development"]:
        raise RuntimeError(f"cannot reuse {source} G predictions: the development schedule differs")
    if theirs.get("manifests/schedule.json") != file_sha256(source.joinpath(*sched_path)):
        raise RuntimeError("the source schedule.json differs from its prepared record")
    record = json.loads((source / "gscreen" / "selection.json").read_text(encoding="utf-8"))
    source_identity = json.loads((source / "prepared" / "manifests" / "identity.json").read_text(encoding="utf-8"))
    source_outputs = file_sha256(source / "prepared" / "manifests" / "outputs.json")
    if source_identity.get("stage") != "prepare" or source_identity.get("outputs_sha256") != source_outputs \
            or record.get("prepared") != source_outputs:
        raise RuntimeError("source G screen is not bound to the source run's completed preparation")
    if record.get("reused_predictions"):
        raise RuntimeError("reuse only from the run that fitted the G predictions (it holds their global records)")
    effective = source_effective_params(source)
    want = {g: (json.dumps(plan.booster_params(c)[0], sort_keys=True), plan.booster_params(c)[1])
            for g, c in plan.G_CONFIGS.items()}
    if effective != want:
        raise RuntimeError("source G predictions were fitted with other effective XGBoost parameters")
    sha = file_sha256(source / "gscreen" / "predictions.csv.gz")
    if record["outputs"].get("predictions.csv.gz") != sha:
        raise RuntimeError("source G predictions differ from their completion record")
    preds = read_csv(source / "gscreen" / "predictions.csv.gz")
    for h in plan.HORIZONS:
        want = {f["target_month"] for f in my_sched["development"] if f["horizon"] == h}
        if len(want) != len(plan.DEV_TARGETS):
            raise RuntimeError(f"h{h}: the development schedule does not list {len(plan.DEV_TARGETS)} targets")
        for g in plan.G_CONFIGS:
            have = set(preds.loc[(preds["horizon"] == h) & (preds["g_config"] == g), "target_month"].astype(str))
            if have != want:
                raise RuntimeError(f"source G predictions h{h} {g}: targets {sorted(have)} != {sorted(want)}")
    if set(preds["g_config"]) != set(plan.G_CONFIGS) or set(preds["horizon"]) != set(plan.HORIZONS):
        raise RuntimeError("source G predictions name configurations or horizons outside the plan")
    return preds, {"source_run": str(source), "source_selection_sha256": file_sha256(source / "gscreen" / "selection.json"),
                   "source_predictions_sha256": sha, "source_code": record.get("code"),
                   "source_prepared_outputs_sha256": source_outputs,
                   "effective_params_checked": "XGB_BASE + G_CONFIGS == params/rounds of every stored source "
                                               "global record (h x G)",
                   "matched_prepared_outputs": {f: mine[f] for f in REUSED_PREPARED},
                   "matched_development_schedule_sha256": dev_schedule_digest(my_sched),
                   "rule": ("no refit: same frozen G configs, byte-identical snapshots/dev truth, identical parsed "
                            "development schedule, all 6 development targets per H x G")}


def pool(executor_workers: int, fn, jobs):
    if executor_workers <= 1:
        return [fn(*job) for job in jobs]
    with ProcessPoolExecutor(max_workers=executor_workers) as ex:
        futures = [ex.submit(fn, *job) for job in jobs]
        return [f.result() for f in futures]


def dev_folds(run: Path) -> list:
    return acc.schedule(run)["development"]


# ------------------------------------------------------------------------------ gscreen

def _gscreen_fold(run: str, fold: dict) -> pd.DataFrame:
    run = Path(run)
    p, st = panel(run, fold["horizon"]), store(run)
    frames = []
    for g in plan.G_CONFIGS:
        res = s3.run_fold(p, st, fold, g, [{"arm": "pooled", "label": g}])
        frames.append(res[g]["predictions"].assign(g_config=g))
    return pd.concat(frames, ignore_index=True)


def gscreen(run: Path, workers: int, reuse: Path | None = None) -> None:
    prepared = acc.accept_prepared(run)
    out = run / "gscreen"
    if out.exists():
        raise FileExistsError(f"{out} exists")
    started = time.time()
    folds = dev_folds(run)
    source = None
    if reuse is not None:
        preds, source = reused_gscreen_predictions(run, reuse)
    else:
        preds = pd.concat(pool(workers, _gscreen_fold, [(str(run), f) for f in folds]), ignore_index=True)
    base = dev_baselines(run)
    rows, selected = [], {}
    for h in plan.HORIZONS:
        best = None
        for g in plan.G_CONFIGS:
            k = keyed(base, preds[(preds["horizon"] == h) & (preds["g_config"] == g)].drop(columns="g_config"),
                      h, f"gscreen h{h} {g}")
            cohort = k[main_cohort(k, h)]
            f1 = exact(cohort, "y_pred_code")
            f4 = exact_fourclass(cohort, "y_pred_code")
            rows.append({"horizon": h, "g_config": g, "endpoint": plan.ENDPOINT, "score": float(f1),
                         "score_exact": str(f1), "macro_f1_fourclass": float(f4), "macro_f1_fourclass_exact": str(f4),
                         "n_main_cohort": int(len(cohort)), "folds": int(k["target_month"].nunique())})
            rank = (f1, tuple(-x for x in plan.g_tiebreak_key(g)))
            if best is None or rank > best[0]:
                best = (rank, g)
        selected[str(h)] = best[1]
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "scores.csv", index=False, float_format="%.17g")
    write_csv_gz(out / "predictions.csv.gz", preds)
    finish(out, "selection.json", {
        "selected": selected, "prepared": prepared["outputs_sha256"], "endpoint": plan.ENDPOINT,
        "g_configs": plan.G_CONFIGS, "xgb_base": plan.XGB_BASE,
        "reused_predictions": source,
        "rule": (f"per horizon: largest {plan.ENDPOINT} (D26: crisis-positive F1, four-class argmax collapsed to "
                 "IPC>=3) over the six development folds on the main cohort (H4/H8 persistence+expert, H12 "
                 "persistence), counts pooled; exact ties -> fewer rounds, shallower trees, lower config number"),
        "bias_disclosure": ("G is chosen on the whole development period; later E3 scores, maps and "
                            "development scores are conditional on this choice (D24)"),
        "seconds": round(time.time() - started, 1)})
    print(json.dumps(selected), flush=True)


# ------------------------------------------------------------------------------ maps

def scheme_pool(frame: pd.DataFrame, scheme: dict, origin: int) -> pd.DataFrame:
    """Candidates of a scheme whose scoring target precedes the origin (all H pooled)."""
    families = plan.STRATEGY_FAMILIES[scheme["strategy"]]
    names = frame["name"].str.split("_")
    local, family = names.str[3], names.str[-1]
    want = np.array([local.iloc[i] == scheme["l_vector"][int(frame["horizon"].iloc[i])] for i in range(len(frame))],
                    dtype=bool) if len(frame) else np.zeros(0, dtype=bool)
    keep = want & family.isin(families).to_numpy() & (frame["target_month"].map(s3.mi) < origin).to_numpy()
    return frame[keep].reset_index(drop=True)


def v7_pool(frame: pd.DataFrame, origin: int) -> pd.DataFrame:
    return frame[frame["target_month"].map(s3.mi) < origin].reset_index(drop=True)


def map_plan(run: Path) -> list:
    """Every (pool label, candidate frame) the development needs, deduplicated by pool identity."""
    from scripts.run_stage2 import pool_identity
    xgb, xgb_paths = acc.candidate_frame(acc.accept_stage1(run))
    v7, v7_paths = acc.candidate_frame(acc.v7_candidates())
    origins = sorted({s3.mi(f["origin_month"]) for f in dev_folds(run)})
    jobs, seen = [], set()
    for scheme in plan.schemes():
        for o in origins:
            sub = scheme_pool(xgb, scheme, o)
            ident = pool_identity(sub)
            if ident not in seen:
                seen.add(ident)
                jobs.append((ident, sub, {n: xgb_paths[n] for n in sub["name"]}, f"{scheme['scheme']}@O{s3.ml(o)}",
                             False, None))
    for o in origins:
        sub = v7_pool(v7, o)
        ident = pool_identity(sub)
        if ident not in seen:
            seen.add(ident)
            # v7 candidates name areas of the full universe: rebuild with v7's own committed
            # geometry (byte-identical to the pinned full coordinates, checked)
            jobs.append((ident, sub, {n: v7_paths[n] for n in sub["name"]}, f"v7@O{s3.ml(o)}",
                         False, str(acc.v7_geometry())))
    return jobs


def _build_map(run: str, ident: str, sub: pd.DataFrame, paths: dict, label: str, keep: bool = False,
               geometry: str | None = None) -> dict:
    from scripts.run_stage2 import build_consensus
    run = Path(run)
    geometry = Path(geometry) if geometry else run / "prepared" / "geometry" / "FEWSNET_admin_code_lat_lon.csv"
    record = build_consensus(run / "maps" / ident, sub, paths, geometry, label, keep)
    return {"map": ident, "route": record["route"], "seconds": record.get("seconds"), "label": label,
            "candidates": record["candidates"], "positive": record.get("positive_weight_candidates")}


def maps(run: Path, workers: int) -> None:
    jobs = map_plan(run)
    print(f"{len(jobs)} distinct candidate pools", flush=True)
    for outcome in pool(workers, _build_map, [(str(run), *j) for j in jobs]):
        print(json.dumps(outcome), flush=True)


def map_for(run: Path, sub: pd.DataFrame, paths: dict | None = None):
    """(map id, route, area->cluster or None) of an already-built map for exactly this pool."""
    from scripts.run_stage2 import accept_consensus, cluster_map, pool_identity
    ident = pool_identity(sub)
    record = accept_consensus(run / "maps" / ident, sub)
    return ident, record, cluster_map(run / "maps" / ident, record)


# ------------------------------------------------------------------------------ fold persistence

def save_fold(base: Path, result: dict, identity: dict, keep_boosters: bool, extra: dict | None = None) -> dict:
    """``extra`` (interruption task): further keyed frames written before the completion record,
    e.g. the same-input pooled diagnostic ``pooled_predictions.csv.gz``."""
    if (base / "fold.json").exists():
        raise FileExistsError(f"{base} is complete; folds are never refitted")
    base.mkdir(parents=True, exist_ok=False)
    write_csv_gz(base / "predictions.csv.gz", result["predictions"])
    for name, frame in (extra or {}).items():
        write_csv_gz(base / name, frame)
    if result["gate_pairs"] is not None:
        write_csv_gz(base / "gate_pairs.csv.gz", result["gate_pairs"])
    (base / "gate.json").write_text(json.dumps({"regions": result["gate"], "locals": result["locals"],
                                                "global": result["global"]}, indent=1, default=str),
                                    encoding="utf-8")
    if keep_boosters:
        (base / "models").mkdir()
        for name, payload in result["boosters"].items():
            (base / "models" / f"{name}.ubj").write_bytes(payload)
    routes = result["predictions"]["route"].value_counts().to_dict()
    return finish(base, "fold.json", {**identity, "status": "fitted", "routes": routes,
                                      "rows": int(len(result["predictions"]))})


def save_incomplete(base: Path, identity: dict) -> dict:
    """An arm-fold the inherited all-labelled coverage gate blocks: recorded, never fitted
    and never replaced by a smaller key set (plan section 6)."""
    base.mkdir(parents=True, exist_ok=False)
    return finish(base, "fold.json", {**identity, "status": "incomplete_coverage_gate"})


def _fold_job(run: str, fold: dict, g: str, specs: list, out_dirs: dict, identities: dict, keep: bool) -> list:
    run = Path(run)
    todo = [s for s in specs if not (out_dirs[s["label"]] / "fold.json").exists()]
    blocked = [s for s in todo if (identities[s["label"]].get("coverage") or {}).get("passed") is False]
    for spec in blocked:
        save_incomplete(out_dirs[spec["label"]], identities[spec["label"]])
    todo = [s for s in todo if s not in blocked]
    if not todo:
        return [(fold["horizon"], fold["target_month"], s["label"], "incomplete_coverage_gate") for s in blocked]
    p, st = panel(run, fold["horizon"]), store(run)
    started = time.time()
    results = s3.run_fold(p, st, fold, g, todo, keep_boosters=keep)
    for spec in todo:
        save_fold(out_dirs[spec["label"]], results[spec["label"]],
                  {**identities[spec["label"]], "seconds_fold_job": round(time.time() - started, 1)}, keep)
    return [(fold["horizon"], fold["target_month"], s["label"]) for s in todo]


# ------------------------------------------------------------------------------ develop

def dev_specs(run: Path, fold: dict, xgb: pd.DataFrame, paths: dict, g_of: dict):
    """Unique arms of one development fold: pooled + one shared arm per distinct map."""
    origin = s3.mi(fold["origin_month"])
    specs, identities, dirs, scheme_maps = [], {}, {}, {}
    base = run / "development" / "folds" / f"h{fold['horizon']}" / fold["target_month"]
    label = "pooled"
    specs.append({"arm": "pooled", "label": label, "local": None, "route": None, "map_id": None})
    identities[label] = {"phase": "development", "arm": "pooled", "horizon": fold["horizon"],
                         "target_month": fold["target_month"], "g_config": g_of[str(fold["horizon"])]}
    dirs[label] = base / label
    for scheme in plan.schemes():
        local = scheme["l_vector"][fold["horizon"]]
        ident, record, cluster_of = map_for(run, scheme_pool(xgb, scheme, origin))
        label = f"shared_{local}_{ident}"
        scheme_maps[scheme["scheme"]] = label if record["route"] == "learned_map" else "pooled"
        if record["route"] != "learned_map" or label in identities:
            continue
        specs.append({"arm": "shared", "label": label, "local": local, "route": "learned_map",
                      "cluster_of": cluster_of, "map_id": ident})
        identities[label] = {"phase": "development", "arm": "shared", "horizon": fold["horizon"],
                             "target_month": fold["target_month"], "g_config": g_of[str(fold["horizon"])],
                             "local_config": local, "map_id": ident, "map_sha256": record["cluster_map_sha256"]}
        dirs[label] = base / label
    return specs, identities, dirs, scheme_maps


def develop(run: Path, workers: int) -> None:
    g_of, _ = acc.accept_g_selection(run)
    xgb, paths = acc.candidate_frame(acc.accept_stage1(run))
    observations = read_csv(run / "prepared" / "ledgers" / "observations.csv")
    jobs, index = [], {}
    for fold in dev_folds(run):
        specs, identities, dirs, scheme_maps = dev_specs(run, fold, xgb, paths, g_of)
        for spec in specs:
            if spec["route"] == "learned_map":
                gate = s3.unmapped_gate(spec["cluster_of"], observations)
                identities[spec["label"]]["coverage"] = gate
        index[f"h{fold['horizon']}_{fold['target_month']}"] = scheme_maps
        jobs.append((str(run), fold, g_of[str(fold["horizon"])], specs, dirs, identities, False))
    out = run / "development"
    out.mkdir(exist_ok=True)
    write_json_atomic(out / "scheme_fold_index.json", index)
    for done in pool(workers, _fold_job, jobs):
        print(json.dumps(done), flush=True)


def dev_predictions(run: Path, horizon: int, target: str, label: str):
    """Accepted predictions of one development arm-fold, or None if its coverage gate
    made it incomplete (the scheme is then incomplete, never scored on fewer keys)."""
    base = run / "development" / "folds" / f"h{horizon}" / target / label
    record = acc.accept_record(base, "fold.json")
    if record.get("status") == "incomplete_coverage_gate":
        return None
    acc.accept_record(base, "fold.json", ["predictions.csv.gz", "gate.json"])
    return read_csv(base / "predictions.csv.gz")


# ------------------------------------------------------------------------------ interruption scenarios

SCEN_PARITY = Fraction(-2, 100)
FREEZE_MONTH = plan.PARTITION_INFO_CUTOFF
ACTUAL_COLUMNS = ("country", "product", "origin_month", "missed_cycles", "evidence", "source")
EXTENSION_FIELDS = ("path", "sha256", "first_month", "last_month", "overlap_months", "source")
#: actual cases: (target, horizon) - October primary, June supplementary forecast-only
ACTUAL_CASES = (("2025-10", 4), ("2025-10", 8), ("2025-06", 4), ("2025-06", 8))
REAL_EVIDENCE = ("verified_vintage", "reconstructed")


def scenario_dev_plan() -> list:
    """The frozen development forecasting folds (design G4: 72 with the six DEV_TARGETS)."""
    folds = [{"strategy": s, "horizon": h, "scenario_k": k, "target_month": t,
              "origin_month": s3.ml(s3.mi(t) - h)}
             for s in plan.SCENARIO_STRATEGIES for h in plan.SCENARIO_HORIZONS for k in plan.SCENARIO_KS
             for t in plan.DEV_TARGETS]
    expected = (len(plan.SCENARIO_STRATEGIES) * len(plan.SCENARIO_HORIZONS) * len(plan.SCENARIO_KS)
                * len(plan.DEV_TARGETS))
    if len(folds) != expected or len({tuple(f.values()) for f in folds}) != expected:
        raise RuntimeError("the scenario development plan is not the frozen fold set")
    return folds


def scen_dev_fold(ctx, strategy: str, horizon: int, k: int, target: str, store, map_record: dict,
                  cluster_of: dict | None, gate_k=None, prediction_areas=None) -> dict:
    """One complete forecasting fold (development, historical or actual): pooled global and the
    system arm (strategy map + gated L1 locals; non-learned map routes are the pooled global
    with their reason). ``gate_k`` = internal replay intensity, int or {country: k} (actual)."""
    panel = s3.ScenarioPanel(ctx, k, strategy, prediction_areas=prediction_areas, gate_k=gate_k)
    fold = {"target_month": target, "origin_month": s3.ml(s3.mi(target) - horizon)}
    route = map_record["route"]
    arms = [{"arm": "pooled", "label": "pooled", "local": None, "route": None},
            {"arm": "shared", "label": "system", "local": plan.SCENARIO_LOCAL, "route": route,
             "cluster_of": cluster_of if route == "learned_map" else None}]
    return s3.run_fold(panel, store, fold, plan.SCENARIO_G[horizon], arms)


def scen_fold_scores(preds: pd.DataFrame) -> dict:
    """Pooled crisis confusion counts of one fold: model on genuine-truth keys, and model vs the
    lawful persistence comparator on identical matched keys (D4/D5)."""
    truth = preds["y_true_code"].to_numpy(dtype=float)
    labelled = np.isfinite(truth)
    pers = preds["persistence_class_code"].to_numpy(dtype=float)
    matched = labelled & np.isfinite(pers)
    y = np.where(labelled, truth, 0).astype(np.int64)
    pred = preds["y_pred_code"].to_numpy(dtype=np.int64)
    return {"model": fourclass.crisis_counts(y[labelled], pred[labelled]),
            "model_matched": fourclass.crisis_counts(y[matched], pred[matched]),
            "persistence": fourclass.crisis_counts(y[matched], pers[matched].astype(np.int64)),
            "keys": int(len(preds)), "labelled": int(labelled.sum()), "matched": int(matched.sum())}


def _f1(counts: dict):
    d = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return Fraction(2 * counts["tp"], d) if d else None


def _add(a: dict, b: dict) -> dict:
    return {key: a.get(key, 0) + b[key] for key in ("tp", "fp", "fn", "tn")}


def _txt(value):
    return None if value is None else str(value)


def ab_select(fold_scores: list) -> dict:
    """D4 selection from complete development predictions (exact rationals).

    ``fold_scores``: dicts with strategy, horizon, scenario_k, target_month and the
    ``scen_fold_scores`` counts; their identities must equal ``scenario_dev_plan()`` exactly
    (no missing, duplicate or extra fold). Per H, a strategy qualifies when its normal-scenario
    crisis F1 minus matched persistence F1 is defined and >= -0.02 AND its one/two-cycle F1 are
    defined; qualifiers are ranked by the equal-weight mean of those two F1; exact ties favour A;
    none qualifying -> no winner, with the unmet criteria."""
    want = sorted((f["strategy"], f["horizon"], f["scenario_k"], f["target_month"]) for f in scenario_dev_plan())
    got = sorted((f["strategy"], int(f["horizon"]), int(f["scenario_k"]), f["target_month"]) for f in fold_scores)
    if got != want:
        raise RuntimeError("development scores do not match the frozen fold identities exactly")
    pooled = {}
    for f in fold_scores:
        cur = pooled.setdefault((f["strategy"], int(f["horizon"]), int(f["scenario_k"])),
                                {"model": {}, "model_matched": {}, "persistence": {}})
        for side in ("model", "model_matched", "persistence"):
            cur[side] = _add(cur[side], f[side])
    out = {}
    for h in plan.SCENARIO_HORIZONS:
        rows = {}
        for s in plan.SCENARIO_STRATEGIES:
            normal, one, two = (pooled[(s, h, k)] for k in plan.SCENARIO_KS)
            fm, fp = _f1(normal["model_matched"]), _f1(normal["persistence"])
            parity = None if fm is None or fp is None else fm - fp
            f1, f2 = _f1(one["model"]), _f1(two["model"])
            mean = None if f1 is None or f2 is None else (f1 + f2) / 2
            unmet = ([] if parity is not None and parity >= SCEN_PARITY else
                     ["normal_parity_undefined" if parity is None else "normal_parity_below_-0.02"])
            if mean is None:
                unmet.append("interruption_f1_undefined")
            rows[s] = {"normal_model_matched_f1": _txt(fm), "normal_persistence_f1": _txt(fp),
                       "normal_parity": _txt(parity), "one_cycle_f1": _txt(f1), "two_cycle_f1": _txt(f2),
                       "interruption_mean_f1": _txt(mean),
                       "one_cycle_persistence_f1": _txt(_f1(one["persistence"])),
                       "two_cycle_persistence_f1": _txt(_f1(two["persistence"])),
                       "qualifies": not unmet, "unmet": unmet,
                       "counts": {f"k{k}": pooled[(s, h, k)] for k in plan.SCENARIO_KS}}
        qualified = [s for s in plan.SCENARIO_STRATEGIES if rows[s]["qualifies"]]
        if qualified:
            best = max(Fraction(rows[s]["interruption_mean_f1"]) for s in qualified)
            winners = [s for s in qualified if Fraction(rows[s]["interruption_mean_f1"]) == best]
            decision = {"winner": "A" if "A" in winners else winners[0], "tie": len(winners) > 1}
        else:
            decision = {"winner": None, "reason": "no strategy meets the normal parity and defined-interruption "
                                                  "criteria; final-model release stops for this horizon"}
        out[str(h)] = {**decision, "strategies": rows}
    return out


def historical_targets(ctx, horizon: int, first: str = "2021-01", last: str = "2024-12") -> dict:
    """D2 common historical calendar for one H over the frozen Feb/Jun/Oct schedule in
    [first, last]: a target is eligible when its origin O = T - H and both simulated missed
    cycles at O (k=2) fall strictly after the 2020-12 freeze; the same targets serve k=0/1/2.
    Targets without genuine truth stay prediction targets (coverage ``truth_available``)."""
    from src.experiment.availability import UnsupportedScenario
    freeze = s3.mi(FREEZE_MONTH)
    labelled = set(int(m) for m in ctx.obs["month"].unique())
    schedule = [m for m in range(s3.mi(first), s3.mi(last) + 1) if m % 12 + 1 in (2, 6, 10)]
    eligible, excluded, coverage = [], [], {}
    for t in schedule:
        o = t - horizon
        try:
            hidden = ctx.ledger.hidden(o, max(plan.SCENARIO_KS))
        except UnsupportedScenario as exc:
            excluded.append({"target_month": s3.ml(t), "reason": f"unsupported_scenario: {exc}"})
            continue
        if o <= freeze or min(hidden) <= freeze:
            excluded.append({"target_month": s3.ml(t), "reason": "origin_or_missed_cycle_not_after_2020-12_freeze"})
            continue
        eligible.append(s3.ml(t))
        coverage[s3.ml(t)] = {"truth_available": t in labelled}
    return {"horizon": horizon, "targets": eligible, "excluded": excluded, "coverage": coverage}


def actual_gate_intensity(table: pd.DataFrame | None, origin: str, countries) -> dict:
    """Actual-case contract (G2): {country: verified missed ordinarily-due cycles} at this origin.

    Refuses (SystemExit) when the table is absent, lacks a column, carries non-real evidence or
    no source, lacks a cohort country, or gives one country two counts. Countries may differ:
    the per-country intensity masks that country's rows inside the shared fit population."""
    if table is None:
        raise SystemExit(f"actual {origin}: no verified country/product availability table; run refused")
    missing = [c for c in ACTUAL_COLUMNS if c not in table.columns]
    if missing:
        raise SystemExit(f"actual availability table lacks {missing}")
    rows = table[(table["origin_month"].astype(str) == origin) & (table["product"] == "CS")]
    if (~rows["evidence"].isin(REAL_EVIDENCE)).any() or rows["source"].isna().any():
        raise SystemExit(f"actual {origin}: availability evidence must be verified/reconstructed with a source")
    counts = pd.to_numeric(rows["missed_cycles"], errors="coerce")
    if counts.isna().any() or (counts < 0).any() or (counts != np.round(counts)).any():
        raise SystemExit(f"actual {origin}: missed cycles must be non-negative integers")
    per = rows.assign(missed_cycles=counts.astype(int)).groupby(rows["country"].astype(str))["missed_cycles"]
    if (per.nunique() > 1).any():
        raise SystemExit(f"actual {origin}: a country has conflicting missed-cycle counts")
    k = per.first().to_dict()
    lacking = sorted(set(map(str, countries)) - set(k))
    if lacking:
        raise SystemExit(f"actual {origin}: no availability entry for countries {lacking[:5]}; run refused")
    return {c: int(k[c]) for c in map(str, countries)}


def _covariate_columns(schema: dict) -> list:
    return list(schema["static_sources"] + schema["dynamic_sources_at_origin"])


def _certified_panel(path: Path, sha: str, schema: dict):
    """The pinned panel's key and covariate columns only, after re-certifying its bytes."""
    from scripts.prepare_fourclass import load_panel
    from src.utils.run_identity import file_sha256 as _sha
    if _sha(path) != sha:
        raise SystemExit(f"{path}: source bytes differ from the prepared source identity")
    panel = load_panel(path)
    return panel[["FEWSNET_admin_code", "date"] + _covariate_columns(schema)]


def load_extension(manifest_path: Path, schema: dict, pinned) -> pd.DataFrame:
    """Actual-case covariate extension (hashed manifest contract, D1/D7).

    The manifest names the file, its SHA-256, its month range, the overlap months used for the
    identity check and its source. Only key and schema covariate columns are read (never
    outcome/expert columns). On every overlap month the retained covariates must equal the pinned
    panel; months after the pinned panel then extend the scaffold. Any gap refuses."""
    from src.utils.run_identity import file_sha256 as _sha
    if manifest_path is None:
        raise SystemExit("actual cases need --actual-scaffold (hashed covariate extension manifest); refused")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    missing = [f for f in EXTENSION_FIELDS if f not in manifest]
    if missing or not manifest["overlap_months"]:
        raise SystemExit(f"extension manifest lacks {missing or ['overlap_months']}")
    path = Path(manifest["path"])
    if _sha(path) != manifest["sha256"]:
        raise SystemExit("extension file bytes differ from its manifest")
    cols = ["FEWSNET_admin_code", "date"] + _covariate_columns(schema)
    ext = pd.read_csv(path, usecols=cols)
    ext["date"] = pd.to_datetime(ext["date"].astype(str).str[:7], format="%Y-%m")
    if ext.duplicated(["FEWSNET_admin_code", "date"]).any():
        raise SystemExit("extension has duplicate area-month keys")
    overlap = pd.to_datetime(pd.Series(manifest["overlap_months"]), format="%Y-%m")
    a = pinned[pinned["date"].isin(overlap)].sort_values(["FEWSNET_admin_code", "date"]).reset_index(drop=True)
    b = ext[ext["date"].isin(overlap)].sort_values(["FEWSNET_admin_code", "date"]).reset_index(drop=True)
    if len(a) == 0 or len(a) != len(b) or not (a[["FEWSNET_admin_code", "date"]].to_numpy()
                                             == b[["FEWSNET_admin_code", "date"]].to_numpy()).all():
        raise SystemExit("extension overlap keys differ from the pinned panel")
    for c in _covariate_columns(schema):
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        if not np.allclose(x, y, equal_nan=True, rtol=0, atol=1e-9):
            raise SystemExit(f"extension disagrees with the pinned panel on {c} over the overlap months")
    later = ext[ext["date"] > pinned["date"].max()]
    if later.empty:
        raise SystemExit("extension adds no month after the pinned panel")
    return pd.concat([pinned, later[pinned.columns]], ignore_index=True)


def scenario_context(run: Path, horizon: int, development_truth: bool = True, extension: Path | None = None,
                     real: bool = True, schema: dict | None = None):
    """Availability of an accepted prepared interruption run.

    Accepts the preparation, re-certifies the pinned panel bytes on every load, reads only its
    key/covariate columns, and uses the prepared ledger/alignment (``real`` refuses synthetic
    evidence; tests may pass real=False). ``development_truth`` supplies input labels as
    evaluator truth (development/historical); actual production passes False. ``extension``
    (actual only) is the hashed covariate-extension manifest; without it the scaffold ends at
    the pinned panel."""
    from src.experiment import availability as av
    from src.feature import fourclass_features as ff
    acc.accept_prepared(run)
    prepared = run / "prepared"
    manifests = prepared / "manifests"
    for name in ("release_ledger.csv", "alignment.json", "sources.json"):
        if not (manifests / name).is_file():
            raise SystemExit(f"{name} missing: this run was not prepared with a release ledger and alignment")
    schema = schema or load_schema(SCHEMA_PATH)
    alignment = json.loads((manifests / "alignment.json").read_text(encoding="utf-8"))
    ledger = av.ReleaseLedger(pd.read_csv(manifests / "release_ledger.csv", dtype=str), real=real)
    obs = read_csv(prepared / "ledgers" / "observations.csv")[["area", "month", "country", "class_code"]]
    source = json.loads((manifests / "sources.json").read_text(encoding="utf-8"))["sources"]["panel"]
    panel = _certified_panel(Path(source["path"]), source["sha256"], schema)
    if extension is not None:
        panel = load_extension(extension, schema, panel)
    scaffold = ff.Scaffold(panel, _covariate_columns(schema))
    return av.Availability(obs, ledger, scaffold, schema, horizon, alignment,
                           truth=obs if development_truth else None)


def scen_candidate_ledger(run: Path) -> pd.DataFrame:
    """The complete 648-row crisis ledger from ACCEPTED Stage 1 scenario evidence (identity,
    code, preparation and every output hash first; ineligible statuses stay as rows)."""
    from scripts.run_stage2 import scenario_candidate_row
    accepted = acc.accept_scenario_stage1(run)
    return pd.DataFrame([scenario_candidate_row(run / "stage1_scenario", v["entry"]) for v in accepted.values()])


def _inputs_identity(run: Path, ctx, **extra) -> dict:
    """Lawful-input identity bound into every scenario fold record: the accepted preparation, the
    Availability input digest (labels, releases, ledger, alignment, features) and, for actual
    cases, the extension manifest / availability table hashes."""
    return {"prepared_outputs_sha256": file_sha256(run / "prepared" / "manifests" / "outputs.json"),
            "availability_inputs_sha256": ctx.inputs_sha256, **extra}


def _fold_dir(root: Path, fold: dict) -> Path:
    return root / fold["strategy"] / f"h{fold['horizon']}" / f"k{fold['scenario_k']}" / fold["target_month"]


def _run_or_accept(base: Path, identity: dict, compute) -> dict:
    """A fold is computed once: an existing completion record is accepted only if its identity
    equals the expected one (never silently reused when incompatible)."""
    if (base / "fold.json").exists():
        return acc.accept_fold(base, identity)
    result = compute()
    return save_fold(base, result["system"], identity, False,
                     extra={"pooled_predictions.csv.gz": result["pooled"]["predictions"]})


def _scenario_map(run: Path, ledger: pd.DataFrame, strategy: str, cutoff: int, release, strict: bool,
                  k_max: int, label: str):
    from scripts.run_stage2 import build_consensus, cluster_map, pool_identity, scenario_map_pool
    sub = scenario_map_pool(ledger, strategy, cutoff, release, strict=strict, k_max=k_max)
    paths = {n: run / "stage1_scenario" / "candidates" / n / "correspondence_table.csv"
             for n in sub.loc[sub["status"] == "scored", "name"]}
    ident = pool_identity(sub, "crisis")
    geometry = run / "prepared" / "geometry" / "FEWSNET_admin_code_lat_lon.csv"
    record = build_consensus(run / "scenario_maps" / ident, sub, paths, geometry, label, metric="crisis")
    return ident, record, cluster_map(run / "scenario_maps" / ident, record)


def scen_develop(run: Path, workers: int) -> None:
    ledger = scen_candidate_ledger(run)
    contexts = {h: scenario_context(run, h) for h in plan.SCENARIO_HORIZONS}
    release = contexts[plan.SCENARIO_HORIZONS[0]].ledger
    st = s3.GlobalStore(run / "scenario_globals")
    maps = {}
    for fold in scenario_dev_plan():
        key = (fold["strategy"], fold["origin_month"])
        if key not in maps:
            maps[key] = _scenario_map(run, ledger, key[0], s3.mi(key[1]), release, True, max(plan.SCENARIO_KS),
                                      f"{key[0]}@O{key[1]}")
        ident, record, cluster_of = maps[key]
        identity = {**fold, "phase": "scenario_development", "map_id": ident, "map_route": record["route"],
                    **_inputs_identity(run, contexts[fold["horizon"]])}
        _run_or_accept(_fold_dir(run / "scenario_development", fold), identity,
                       lambda: scen_dev_fold(contexts[fold["horizon"]], fold["strategy"], fold["horizon"],
                                             fold["scenario_k"], fold["target_month"], st, record, cluster_of))
        print(json.dumps({**fold, "map_route": record["route"]}), flush=True)


def scen_select(run: Path) -> None:
    out = run / "scenario_development"
    if (out / "selection.json").exists():
        raise FileExistsError("selection is written once")
    scores = []
    for fold in scenario_dev_plan():
        base = _fold_dir(out, fold)
        record = acc.accept_fold(base, {k: fold[k] for k in ("strategy", "horizon", "scenario_k", "target_month")})
        scores.append({**fold, **scen_fold_scores(read_csv(base / "predictions.csv.gz")),
                       "fold_sha256": file_sha256(base / "fold.json"), "map_id": record["map_id"]})
    finish(out, "selection.json", {"phase": "scenario_select", "rule": "D4 normal parity >= -0.02, then mean one/"
                                   "two-cycle crisis F1, ties A, no qualifier no winner",
                                   "decisions": ab_select(scores), "folds": len(scores),
                                   "fold_records": {f"{s['strategy']}_h{s['horizon']}_k{s['scenario_k']}_"
                                                    f"{s['target_month']}": s["fold_sha256"] for s in scores}})


def _accept_selection(run: Path) -> dict:
    """The accepted selection whose recorded fold records still equal the current fold files."""
    out = run / "scenario_development"
    selection = acc.accept_record(out, "selection.json")
    for fold in scenario_dev_plan():
        key = f"{fold['strategy']}_h{fold['horizon']}_k{fold['scenario_k']}_{fold['target_month']}"
        if selection["fold_records"].get(key) != file_sha256(_fold_dir(out, fold) / "fold.json"):
            raise RuntimeError(f"selection fold record {key} differs from the current fold")
    return selection


def scen_freeze(run: Path) -> None:
    out = run / "scenario_final"
    if (out / "frozen.json").exists():
        raise FileExistsError("the recipe is frozen once")
    selection = _accept_selection(run)
    ledger = scen_candidate_ledger(run)
    release = scenario_context(run, plan.SCENARIO_HORIZONS[0]).ledger
    frozen = {}
    for h in plan.SCENARIO_HORIZONS:
        decision = selection["decisions"][str(h)]
        if decision["winner"] is None:
            frozen[str(h)] = {"released": False, "reason": decision["reason"]}
            continue
        ident, record, _ = _scenario_map(run, ledger, decision["winner"], s3.mi(FREEZE_MONTH), release, False, 0,
                                         f"{decision['winner']}@final")
        frozen[str(h)] = {"released": True, "strategy": decision["winner"], "map_id": ident,
                          "map_route": record["route"], "g_config": plan.SCENARIO_G[h],
                          "local_config": plan.SCENARIO_LOCAL}
    out.mkdir(parents=True, exist_ok=True)
    finish(out, "frozen.json", {"phase": "scenario_freeze", "cutoff": FREEZE_MONTH, "recipe": frozen,
                                "selection_sha256": file_sha256(run / "scenario_development" / "selection.json")})


def _accept_frozen(run: Path) -> dict:
    frozen = acc.accept_record(run / "scenario_final", "frozen.json")
    _accept_selection(run)
    if frozen["selection_sha256"] != file_sha256(run / "scenario_development" / "selection.json"):
        raise RuntimeError("frozen recipe is not bound to the current accepted selection")
    return frozen


def _frozen_map(run: Path, entry: dict):
    from scripts.run_stage2 import accept_consensus, cluster_map
    record = accept_consensus(run / "scenario_maps" / entry["map_id"], None, "crisis")
    return record, cluster_map(run / "scenario_maps" / entry["map_id"], record)


def scen_historical(run: Path) -> None:
    frozen = _accept_frozen(run)
    out = run / "scenario_historical"
    if (out / "historical.json").exists():
        raise FileExistsError("historical evaluation is run once")
    out.mkdir(parents=True, exist_ok=True)
    st = s3.GlobalStore(run / "scenario_globals")
    calendar = {}
    for h in plan.SCENARIO_HORIZONS:
        entry = frozen["recipe"][str(h)]
        if not entry["released"]:
            calendar[str(h)] = {"released": False, "reason": entry["reason"]}
            continue
        ctx = scenario_context(run, h)
        record, cluster_of = _frozen_map(run, entry)
        cal = {**historical_targets(ctx, h), "released": True, "strategy": entry["strategy"]}
        calendar[str(h)] = cal
        for k in plan.SCENARIO_KS:
            for target in cal["targets"]:
                fold = {"strategy": entry["strategy"], "horizon": h, "scenario_k": k, "target_month": target}
                identity = {**fold, "phase": "scenario_historical", "map_id": entry["map_id"],
                            **_inputs_identity(run, ctx)}
                _run_or_accept(_fold_dir(out, fold), identity,
                               lambda: scen_dev_fold(ctx, entry["strategy"], h, k, target, st, record, cluster_of))
    finish(out, "historical.json", {"phase": "scenario_historical", "calendar": calendar,
                                    "frozen_sha256": file_sha256(run / "scenario_final" / "frozen.json")})


def scen_actual(run: Path, availability_csv: Path | None, extension: Path | None) -> None:
    """2025 prediction-only folds. Refuses without the verified availability table or the hashed
    covariate extension; truth is never loaded."""
    frozen = _accept_frozen(run)
    table = None if availability_csv is None else pd.read_csv(availability_csv, dtype={"origin_month": str})
    out = run / "scenario_actual"
    st = s3.GlobalStore(run / "scenario_globals")
    done = {}
    for target, h in ACTUAL_CASES:
        entry = frozen["recipe"][str(h)]
        if not entry["released"]:
            done[f"h{h}_{target}"] = {"released": False, "reason": entry["reason"]}
            continue
        if extension is None:
            raise SystemExit("actual cases need --actual-scaffold (hashed covariate extension); refused")
        ctx = scenario_context(run, h, development_truth=False, extension=extension)
        origin = s3.ml(s3.mi(target) - h)
        gate_k = actual_gate_intensity(table, origin, ctx.countries)
        record, cluster_of = _frozen_map(run, entry)
        fold = {"strategy": entry["strategy"], "horizon": h, "scenario_k": 0, "target_month": target}
        identity = {**fold, "phase": "scenario_actual", "gate_k": gate_k, "map_id": entry["map_id"],
                    "truth": "not loaded; evaluated only after a separate truth release",
                    **_inputs_identity(run, ctx, extension_manifest_sha256=file_sha256(extension),
                                       availability_table_sha256=file_sha256(availability_csv))}
        rec = _run_or_accept(out / f"h{h}" / target, identity,
                             lambda: scen_dev_fold(ctx, entry["strategy"], h, 0, target, st, record, cluster_of,
                                                   gate_k=gate_k))
        done[f"h{h}_{target}"] = {"released": True, "fold_sha256": file_sha256(out / f"h{h}" / target / "fold.json"),
                                  "rows": rec.get("rows")}
    out.mkdir(parents=True, exist_ok=True)
    finish(out, "actual.json", {"phase": "scenario_actual", "cases": done,
                                "frozen_sha256": file_sha256(run / "scenario_final" / "frozen.json")})


def _with_pooled(base: Path, preds: pd.DataFrame) -> pd.DataFrame:
    """System predictions joined 1:1 to the saved same-input pooled diagnostic of the same fold."""
    pooled = read_csv(base / "pooled_predictions.csv.gz")[["area", "target_month", "y_pred_code"]]
    out = preds.merge(pooled.rename(columns={"y_pred_code": "y_pred_pooled"}), on=["area", "target_month"],
                      how="left", validate="one_to_one")
    if out["y_pred_pooled"].isna().any():
        raise RuntimeError(f"{base}: pooled diagnostic keys differ from the system keys")
    return out


def _origin_truth(preds: pd.DataFrame, horizon: int, keyed: pd.Series) -> np.ndarray:
    """Evaluator-only exact-origin truth at (area, T - H) from keyed genuine labels; NaN if absent
    (never the latest earlier label)."""
    months = preds["target_month"].map(s3.mi) - int(horizon)
    return keyed.reindex(pd.MultiIndex.from_arrays([preds["area"], months])).to_numpy(dtype=float)


def _report_entries(rep, preds: pd.DataFrame, origin_truth, horizon: int, k, expert=None):
    entries, tables = [], {}
    studies = rep.study_rows(preds, origin_truth)
    for name in ("study1", "study2"):
        rows = studies[name]
        matched = rows[rows["persistence_class_code"].notna()]
        entry = {"horizon": horizon, "scenario_k": k, "study": name, "cohort_keys": int(len(preds)),
                 "keys": int(len(rows)), "matched": int(len(matched)),
                 "model_standalone": fourclass.nullable_crisis_summary(rows["truth_code"].astype(int),
                                                                       rows["y_pred_code"].astype(int))
                 if len(rows) else None,
                 "vs_persistence": rep.crisis_paired_bootstrap(matched, "y_pred_code", "persistence_class_code"),
                 "vs_pooled_same_input": rep.crisis_paired_bootstrap(rows, "y_pred_code", "y_pred_pooled")}
        if expert is not None:
            with_expert = rows.merge(expert, on=["area", "target_month"], how="left")
            ex = with_expert[with_expert["expert_class_code"].notna()]
            entry["vs_expert"] = rep.crisis_paired_bootstrap(ex, "y_pred_code", "expert_class_code")
            entry["expert_coverage"] = {str(r or "matched"): int(n) for r, n in
                                        with_expert["expert_reason"].fillna("").value_counts().items()}
        if name == "study2":
            entry.update(onset_model=rep.onset_recall(rows, "y_pred_code"),
                         onset_persistence=rep.onset_recall(matched, "persistence_class_code"),
                         excluded_missing_origin=studies["study2_excluded_missing_origin"],
                         excluded_origin_crisis=studies["study2_excluded_origin_crisis"])
        entries.append(entry)
    keyed = preds.assign(origin_truth=np.asarray(origin_truth, dtype=float))
    if expert is not None:
        keyed = keyed.merge(expert, on=["area", "target_month"], how="left")
    comparators = {"persistence": "persistence_class_code", "pooled": "y_pred_pooled",
                   **({"expert": "expert_class_code"} if expert is not None else {})}
    tables["country"] = rep.country_table(keyed, "y_pred_code", comparators, keyed["origin_truth"])
    tables["keyed"] = keyed     # reconstructible evaluator/comparator join (D6)
    return entries, tables


def _expert_for(rep, preds: pd.DataFrame, expert_csv: Path | None, horizon: int):
    """Keyed expert comparator; without a documented table every key keeps an explicit
    'no_documented_expert_table' coverage reason."""
    keys = preds[["area", "target_month", "origin_month"]].assign(horizon=horizon)
    table = None if expert_csv is None else pd.read_csv(expert_csv)
    return rep.keyed_expert(keys, table)[["area", "target_month", "expert_class_code", "expert_reason"]]


def scen_report(run: Path, expert_csv: Path | None = None) -> None:
    from scripts import report_fourclass as rep
    out = run / "scenario_report"
    if (out / "report.json").exists():
        raise FileExistsError("the report is written once")
    historical = acc.accept_record(run / "scenario_historical", "historical.json")
    _accept_frozen(run)
    if historical["frozen_sha256"] != file_sha256(run / "scenario_final" / "frozen.json"):
        raise RuntimeError("historical evaluation is not bound to the current frozen recipe")
    out.mkdir(parents=True, exist_ok=True)
    obs = read_csv(run / "prepared" / "ledgers" / "observations.csv").set_index(["area", "month"])["class_code"]
    results = []
    for h, cal in historical["calendar"].items():
        if not cal.get("released"):
            results.append({"horizon": int(h), "released": False, "reason": cal.get("reason")})
            continue
        for k in plan.SCENARIO_KS:
            frames = []
            for target in cal["targets"]:
                base = _fold_dir(run / "scenario_historical", {"strategy": cal["strategy"], "horizon": int(h),
                                                               "scenario_k": k, "target_month": target})
                acc.accept_fold(base, {"strategy": cal["strategy"], "horizon": int(h), "scenario_k": k,
                                       "target_month": target})
                frames.append(_with_pooled(base, read_csv(base / "predictions.csv.gz")))
            if not frames:
                results.append({"horizon": int(h), "scenario_k": k, "released": True, "reason": "no eligible target"})
                continue
            preds = pd.concat(frames, ignore_index=True).rename(columns={"y_true_code": "truth_code"})
            origin_truth = _origin_truth(preds, int(h), obs)
            entries, tables = _report_entries(rep, preds, origin_truth, int(h), k, _expert_for(rep, preds, expert_csv,
                                                                                                int(h)))
            results += entries
            tables["country"].to_csv(out / f"country_h{h}_k{k}.csv", index=False)
            write_csv_gz(out / f"keyed_h{h}_k{k}.csv.gz", tables["keyed"])
    finish(out, "report.json", {"phase": "scenario_report", "comparisons": results,
                                "historical_sha256": file_sha256(run / "scenario_historical" / "historical.json"),
                                "expert_table_sha256": None if expert_csv is None else file_sha256(expert_csv),
                                "note": "historical 2021-2024 only; 2025 truth is evaluated by scen-evaluate after its "
                                        "separate approved release"})


TRUTH_RELEASE_FIELDS = ("approved", "approved_by", "crosswalk", "truth_file", "truth_sha256", "frozen_actual")


def scen_evaluate(run: Path, release_dir: Path | None, expert_csv: Path | None = None) -> None:
    """Separate evaluation of FROZEN actual predictions once approved keyed truth is released.

    ``release_dir/release.json`` must be approved, name the crosswalk, the truth file and its
    SHA-256, and record the SHA-256 of the frozen ``scenario_actual/actual.json`` it evaluates
    (so truth is released only after predictions are frozen). Truth rows (area, target_month,
    class_code 0..3) are joined to the frozen keys; unmatched keys stay coverage, never zero-filled;
    a target without any released truth is reported unevaluable (e.g. June)."""
    from scripts import report_fourclass as rep
    if release_dir is None:
        raise SystemExit("scen-evaluate needs --truth-release (approved keyed truth release)")
    actual = acc.accept_record(run / "scenario_actual", "actual.json")
    release = json.loads((Path(release_dir) / "release.json").read_text(encoding="utf-8"))
    missing = [f for f in TRUTH_RELEASE_FIELDS if f not in release]
    if missing or release["approved"] is not True:
        raise SystemExit(f"truth release not approved or lacks {missing}")
    if release["frozen_actual"] != file_sha256(run / "scenario_actual" / "actual.json"):
        raise SystemExit("truth release does not bind the frozen actual predictions")
    truth_path = Path(release_dir) / release["truth_file"]
    if file_sha256(truth_path) != release["truth_sha256"]:
        raise SystemExit("truth file bytes differ from the release record")
    truth = pd.read_csv(truth_path)
    codes = pd.to_numeric(truth["class_code"], errors="coerce")
    if truth.duplicated(["area", "target_month"]).any() or codes.isna().any() or (~codes.isin([0, 1, 2, 3])).any():
        raise SystemExit("released truth must be unique keys with class codes on the 0..3 axis")
    out = run / "scenario_evaluation"
    if (out / "evaluation.json").exists():
        raise FileExistsError("the evaluation is written once")
    out.mkdir(parents=True, exist_ok=True)
    released_truth = truth.assign(month=truth["target_month"].map(s3.mi)).set_index(["area", "month"])["class_code"]
    obs = read_csv(run / "prepared" / "ledgers" / "observations.csv").set_index(["area", "month"])["class_code"]
    keyed = pd.concat([obs, released_truth[~released_truth.index.isin(obs.index)]])   # genuine labels only
    results = []
    for case, info in actual["cases"].items():
        if not info.get("released"):
            results.append({"case": case, "released": False})
            continue
        h, target = int(case.split("_")[0][1:]), case.split("_", 1)[1]
        base = run / "scenario_actual" / f"h{h}" / target
        acc.accept_record(base, "fold.json", ["predictions.csv.gz", "gate.json"])
        if file_sha256(base / "fold.json") != info["fold_sha256"]:
            raise RuntimeError(f"{case}: actual fold differs from the frozen actual record")
        preds = _with_pooled(base, read_csv(base / "predictions.csv.gz").drop(columns=["y_true_code"]))
        joined = preds.merge(truth.rename(columns={"class_code": "truth_code"})[["area", "target_month", "truth_code"]],
                             on=["area", "target_month"], how="left")
        origin_truth = _origin_truth(joined, h, keyed)
        evaluable = bool(joined["truth_code"].notna().any())
        entries, tables = _report_entries(rep, joined, origin_truth, h, "actual",
                                          _expert_for(rep, joined, expert_csv, h))
        results += [{"case": case, "evaluable": evaluable,
                     **({} if evaluable else {"reason": "no released genuine truth for this target "
                                                        "(forecast/coverage only)"}), **e} for e in entries]
        tables["country"].to_csv(out / f"country_{case}.csv", index=False)   # coverage-only countries kept
        write_csv_gz(out / f"keyed_{case}.csv.gz", tables["keyed"])
    finish(out, "evaluation.json", {"phase": "scenario_evaluate", "results": results,
                                    "truth_release": {k: release[k] for k in TRUTH_RELEASE_FIELDS},
                                    "expert_table_sha256": None if expert_csv is None else file_sha256(expert_csv),
                                    "note": "Study2 uses evaluator-only exact-origin truth from genuine historical "
                                            "labels or the released truth at (area, T - H); keys without it are "
                                            "excluded and counted, never filled with an earlier label"})


# ------------------------------------------------------------------------------ select

def select(run: Path) -> None:
    out = run / "development"
    if (out / "selection.json").exists():
        raise FileExistsError("selection exists")
    g_of, g_record = acc.accept_g_selection(run)
    xgb, paths = acc.candidate_frame(acc.accept_stage1(run))
    base = dev_baselines(run)
    folds = dev_folds(run)
    rows = []
    pooled = {h: pd.concat([dev_predictions(run, h, f["target_month"], "pooled") for f in folds
                            if f["horizon"] == h]) for h in plan.HORIZONS}
    # scheme -> arm label of each fold, accepted once per fold (not once per scheme x fold)
    fold_maps = {(f["horizon"], f["target_month"]): dev_specs(run, f, xgb, paths, g_of)[3] for f in folds}
    for scheme in plan.schemes():
        row = {"scheme": scheme["scheme"], "l_vector_id": scheme["l_vector_id"], "strategy": scheme["strategy"],
               **{f"L_h{h}": scheme["l_vector"][h] for h in plan.HORIZONS}}
        deltas, expert = {}, {}
        complete = True
        for h in plan.HORIZONS:
            frames = []
            for f in (f for f in folds if f["horizon"] == h):
                scheme_maps = fold_maps[(h, f["target_month"])]
                frames.append(dev_predictions(run, h, f["target_month"], scheme_maps[scheme["scheme"]]))
            if any(x is None for x in frames):
                complete = False
                row[f"incomplete_h{h}"] = sum(x is None for x in frames)
                continue
            k = keyed(base, pd.concat(frames), h, f"{scheme['scheme']} h{h}")
            cohort = k[main_cohort(k, h)]
            f_s, f_p = exact(cohort, "y_pred_code"), exact(cohort, "persistence_code")
            kp = keyed(base, pooled[h], h, f"pooled h{h}")
            f_pool = exact(kp[main_cohort(kp, h)], "y_pred_code")
            deltas[h] = f_s - f_p
            row.update({f"macro_f1_h{h}": float(f_s), f"persistence_h{h}": float(f_p),
                        f"pooled_h{h}": float(f_pool), f"delta_persistence_h{h}": float(f_s - f_p),
                        f"delta_pooled_h{h}": float(f_s - f_pool), f"n_main_h{h}": int(len(cohort)),
                        f"delta_persistence_exact_h{h}": str(f_s - f_p)})
            if h in (4, 8):
                f_e = exact(cohort, "expert_code")
                expert[h] = f_s - f_e
                row.update({f"expert_h{h}": float(f_e), f"delta_expert_h{h}": float(f_s - f_e)})
        row["complete"] = complete
        if not complete:  # a coverage-blocked scheme is reported, never selected
            rows.append(((0,), row))
            continue
        key = (1, min(deltas.values()), sum(deltas.values()) / 3, (expert[4] + expert[8]) / 2,
               sum(v == "L1" for v in scheme["l_vector"].values()),
               -plan.STRATEGIES.index(scheme["strategy"]), -scheme["l_vector_id"])
        row.update(rule_min_delta=float(key[1]), rule_mean_delta=float(key[2]), rule_mean_expert=float(key[3]),
                   rule_key=[str(x) for x in key])
        rows.append((key, row))
    rows.sort(key=lambda kr: kr[0], reverse=True)
    table = pd.DataFrame([r for _, r in rows])
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.to_csv(out / "selection_table.csv", index=False, float_format="%.17g")
    winner = rows[0][1]
    if not winner["complete"]:
        raise RuntimeError("every development scheme is incomplete under the coverage gate")
    scheme = next(s for s in plan.schemes() if s["scheme"] == winner["scheme"])
    finish(out, "selection.json", {
        "selected_scheme": scheme["scheme"], "l_vector": {str(h): v for h, v in scheme["l_vector"].items()},
        "strategy": scheme["strategy"], "g_selection": g_of, "g_selection_record": g_record,
        "rule": ["complete (coverage gate passed on every fold)", "max min_H(F1 - F1 persistence)", "max mean_H(F1 - F1 persistence)",
                 "max mean_{H4,H8}(F1 - F1 expert)", "more L1", "strict-only > loose-only > merged",
                 "lower L-vector number"],
        "cohort": "main: H4/H8 persistence+expert, H12 persistence; six development folds per H, counts pooled",
        "bias_disclosure": ("development scores were used for G and for this 24-way selection; they are not "
                            "an independent forward test (D24); map construction used scoring targets < O"),
        "schemes_compared": len(rows)})
    print(table[["rank", "scheme", "rule_min_delta", "rule_mean_delta", "rule_mean_expert"]].head(8).to_string(),
          flush=True)


# ------------------------------------------------------------------------------ oldmap diagnostic

def oldmap(run: Path, workers: int) -> None:
    sel = acc.accept_record(run / "development", "selection.json", ["selection_table.csv"])
    v7, _ = acc.candidate_frame(acc.v7_candidates())
    observations = read_csv(run / "prepared" / "ledgers" / "observations.csv")
    jobs = []
    for fold in dev_folds(run):
        h = fold["horizon"]
        local, g = sel["l_vector"][str(h)], sel["g_selection"][str(h)]
        ident, record, cluster_of = map_for(run, v7_pool(v7, s3.mi(fold["origin_month"])))
        base = run / "development" / "oldmap" / f"h{h}" / fold["target_month"]
        specs, identities, dirs = [], {}, {}
        for arm in ("independent", "shared"):
            label = f"rfmap_{arm}"
            specs.append({"arm": arm, "label": label, "local": local, "route": record["route"],
                          "cluster_of": cluster_of, "map_id": ident})
            identities[label] = {"phase": "development_oldmap", "arm": arm, "horizon": h,
                                 "target_month": fold["target_month"], "g_config": g, "local_config": local,
                                 "map_id": ident, "map_route": record["route"],
                                 "coverage": s3.unmapped_gate(cluster_of, observations) if cluster_of else None}
            dirs[label] = base / label
        jobs.append((str(run), fold, g, specs, dirs, identities, False))
    for done in pool(workers, _fold_job, jobs):
        print(json.dumps(done), flush=True)


# ------------------------------------------------------------------------------ freeze

def freeze(run: Path) -> None:
    out = run / "frozen"
    if out.exists():
        raise FileExistsError(f"{out} exists")
    sel = acc.accept_record(run / "development", "selection.json", ["selection_table.csv"])
    xgb, paths = acc.candidate_frame(acc.accept_stage1(run))
    scheme = next(s for s in plan.schemes() if s["scheme"] == sel["selected_scheme"])
    cutoff = s3.mi(plan.PARTITION_INFO_CUTOFF) + 1
    sub = scheme_pool(xgb, scheme, cutoff)
    if sub["target_month"].map(s3.mi).max() > s3.mi(plan.PARTITION_INFO_CUTOFF):
        raise RuntimeError("final map candidates exceed the 2020-12 information cutoff")
    from scripts.run_stage2 import pool_identity
    outcome = _build_map(str(run), pool_identity(sub), sub, {n: paths[n] for n in sub["name"]}, "final_map", True)
    ident, record, cluster_of = map_for(run, sub)
    out.mkdir()
    finish(out, "frozen.json", {
        "information_cutoff": plan.PARTITION_INFO_CUTOFF, "selected_scheme": sel["selected_scheme"],
        "g_selection": sel["g_selection"], "l_vector": sel["l_vector"], "strategy": sel["strategy"],
        "selection_record_sha256": file_sha256(run / "development" / "selection.json"),
        "final_map": {"map_id": ident, "route": record["route"], "cluster_map_sha256": record.get("cluster_map_sha256"),
                      "candidates": record["candidates"], "positive_weight": record.get("positive_weight_candidates"),
                      "clusters": record.get("actual_clusters")},
        "rf_map": {"path": str(acc.V7_FINAL_MAP), "sha256": acc.V7_FINAL_MAP_SHA256},
        "rules": {"window": plan.WINDOW, "gate_dates": plan.GATE_DATES, "stage3_gain": str(plan.STAGE3_GAIN),
                  "fit_support": plan.FIT_SUPPORT, "gate_support": plan.STAGE3_GATE_SUPPORT,
                  "unmapped_route": "global", "argmax": "fixed four-class axis, ties to the first class"},
        "build": outcome})
    print(json.dumps({"map": ident, "route": record["route"], "clusters": record.get("actual_clusters")}), flush=True)


# ------------------------------------------------------------------------------ final

FINAL_ARMS = ("pooled", "rfmap_independent", "rfmap_shared", "xgbmap_shared")


def final(run: Path, workers: int) -> None:
    frozen = acc.accept_record(run / "frozen", "frozen.json")
    acc.accept_stage1(run)  # the final phase runs only on an accepted Stage 1 and selection
    acc.accept_record(run / "development", "selection.json", ["selection_table.csv"])
    if frozen["selection_record_sha256"] != file_sha256(run / "development" / "selection.json"):
        raise RuntimeError("frozen record does not bind the accepted selection")
    from scripts.run_stage2 import accept_consensus, cluster_map
    map_dir = run / "maps" / frozen["final_map"]["map_id"]
    record = accept_consensus(map_dir)
    xgb_map = cluster_map(map_dir, record)
    rf_map = acc.v7_final_map()
    observations = read_csv(run / "prepared" / "ledgers" / "observations.csv")
    evaluated = read_csv(run / "prepared" / "ledgers" / "baselines.csv")[["area"]]
    coverage = {"xgbmap": s3.unmapped_gate(xgb_map, observations, evaluated) if xgb_map else None,
                "rfmap": s3.unmapped_gate(rf_map, observations, evaluated)}
    jobs = []
    for fold in acc.schedule(run)["stage3"]:
        if fold["status"] != "scheduled":
            continue
        h = fold["horizon"]
        g, local = frozen["g_selection"][str(h)], frozen["l_vector"][str(h)]
        base = run / "final" / f"h{h}" / fold["target_month"]
        specs = [{"arm": "pooled", "label": "pooled", "local": None, "route": None, "map_id": None},
                 {"arm": "independent", "label": "rfmap_independent", "local": local, "route": "learned_map",
                  "cluster_of": rf_map, "map_id": "v7_final"},
                 {"arm": "shared", "label": "rfmap_shared", "local": local, "route": "learned_map",
                  "cluster_of": rf_map, "map_id": "v7_final"},
                 {"arm": "shared", "label": "xgbmap_shared", "local": local, "route": record["route"],
                  "cluster_of": xgb_map, "map_id": frozen["final_map"]["map_id"]}]
        identities = {s["label"]: {"phase": "final", "arm": s["arm"], "horizon": h, "target_month": fold["target_month"],
                                   "g_config": g, "local_config": s["local"], "map_id": s["map_id"],
                                   "frozen_sha256": file_sha256(run / "frozen" / "frozen.json"),
                                   "coverage": coverage["rfmap" if s["label"].startswith("rfmap") else "xgbmap"]
                                   if s["arm"] != "pooled" else None}
                      for s in specs}
        dirs = {s["label"]: base / s["label"] for s in specs}
        jobs.append((str(run), fold, g, specs, dirs, identities, True))
    for done in pool(workers, _fold_job, jobs):
        print(json.dumps(done), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("phase", choices=("gscreen", "maps", "develop", "select", "oldmap", "freeze", "final",
                                          "scen-develop", "scen-select", "scen-freeze", "scen-historical",
                                          "scen-actual", "scen-report", "scen-evaluate"))
    parser.add_argument("--actual-availability", type=Path, default=None,
                        help="scen-actual: verified country/product missed-cycle table (refused when absent)")
    parser.add_argument("--actual-scaffold", type=Path, default=None,
                        help="scen-actual: hashed covariate-extension manifest (refused when absent)")
    parser.add_argument("--expert-table", type=Path, default=None,
                        help="scen-report/scen-evaluate: documented same-horizon expert table (optional)")
    parser.add_argument("--truth-release", type=Path, default=None,
                        help="scen-evaluate: approved keyed truth release directory")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reuse-gscreen-from", type=Path, default=None,
                        help="gscreen only: reuse an earlier run's saved G predictions (identity-checked, no refit)")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.phase.startswith("scen-"):   # interruption task: own crisis-aligned phases
        started = time.time()
        {"scen-develop": lambda: scen_develop(run, args.workers), "scen-select": lambda: scen_select(run),
         "scen-freeze": lambda: scen_freeze(run), "scen-historical": lambda: scen_historical(run),
         "scen-actual": lambda: scen_actual(run, args.actual_availability, args.actual_scaffold),
         "scen-report": lambda: scen_report(run, args.expert_table),
         "scen-evaluate": lambda: scen_evaluate(run, args.truth_release, args.expert_table)}[args.phase]()
        print(f"{args.phase} finished in {time.time() - started:.0f}s", flush=True)
        return
    if args.phase != "gscreen" and not plan.DOWNSTREAM_ALIGNED:
        raise SystemExit(f"{args.phase}: Stage 2/3 metric alignment to {plan.ENDPOINT} awaits review of the Stage 1 "
                         "diagnostics (D26); not run")
    started = time.time()
    {"gscreen": lambda: gscreen(run, args.workers, args.reuse_gscreen_from), "maps": lambda: maps(run, args.workers),
     "develop": lambda: develop(run, args.workers), "select": lambda: select(run),
     "oldmap": lambda: oldmap(run, args.workers), "freeze": lambda: freeze(run),
     "final": lambda: final(run, args.workers)}[args.phase]()
    print(f"{args.phase} finished in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
