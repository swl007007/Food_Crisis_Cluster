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

def save_fold(base: Path, result: dict, identity: dict, keep_boosters: bool) -> dict:
    if (base / "fold.json").exists():
        raise FileExistsError(f"{base} is complete; folds are never refitted")
    base.mkdir(parents=True, exist_ok=False)
    write_csv_gz(base / "predictions.csv.gz", result["predictions"])
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
    parser.add_argument("phase", choices=("gscreen", "maps", "develop", "select", "oldmap", "freeze", "final"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reuse-gscreen-from", type=Path, default=None,
                        help="gscreen only: reuse an earlier run's saved G predictions (identity-checked, no refit)")
    args = parser.parse_args()
    run = args.run_dir.resolve()
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
