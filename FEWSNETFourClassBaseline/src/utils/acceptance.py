"""One acceptance chain for every persisted stage output (audit repair round 4).

Every consumer — Stage 2, Stage 3, the report and the verifier — accepts upstream
artifacts ONLY through these functions, so a check can never live at the producer while
a consumer trusts the record. Each accept_* function re-derives membership from
independent evidence and checks row-level agreement, and calls the accept_* of the stage
below it. Any problem raises AcceptanceError.

Chain: accept_prepared -> accept_stage1 -> accept_stage2 -> accept_stage3_horizon.

Every persisted CSV is read with float_precision="round_trip": pandas' default fast
parser is not exact for 17-significant-digit values, and acceptance compares exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import inventories as inv
from src.utils import run_identity as rid

LOCAL = "local_model"


class AcceptanceError(RuntimeError):
    pass


def _raise(problems, where):
    if problems:
        raise AcceptanceError(f"{where}: {problems[:8]}")


def _schedule(run: Path) -> dict:
    return json.loads((Path(run) / "prepared" / "manifests" / "schedule.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------ prepared

def accept_prepared(run: Path) -> dict:
    try:
        return rid.require_prepared(run)
    except RuntimeError as exc:
        raise AcceptanceError(str(exc)) from exc


# ------------------------------------------------------------------------------ Stage 1

def accept_stage1(run: Path) -> dict:
    """Every scheduled fold (exactly those) with a verified completion record; returns
    {fold name: candidate.json}."""
    from scripts.run_stage1 import verify_fold  # local import: scripts depend on this module
    run = Path(run)
    prepared = accept_prepared(run)
    schedule = _schedule(run)
    present = [p.name for p in (run / "stage1" / "folds").iterdir() if p.is_dir()]
    _raise(inv.stage1_population_problems(schedule, present), "Stage 1 population")
    plan = set(json.loads((run / "stage1" / "retain_plan.json").read_text(encoding="utf-8")))
    candidates, problems = {}, []
    for name, status in inv.stage1_names(schedule).items():
        if status != "scheduled":
            continue
        scope, term = name[2], name.split("_")[1]
        try:
            verify_fold(run, name, prepared, retain_expected=f"fs{scope}:{term}" in plan)
            candidate = json.loads((run / "stage1" / "folds" / name / "candidate.json").read_text(encoding="utf-8"))
            scores = candidate["scores"]
            if not np.isfinite([scores["macro_f1"], scores["macro_f1_base"]]).all():
                raise AcceptanceError("non-finite held-out score")
            candidates[name] = candidate
        except Exception as exc:  # missing, partial, stale or foreign fold
            problems.append(f"{name}: {exc}")
    _raise(problems, "Stage 1 folds")
    return candidates


# ------------------------------------------------------------------------------ Stage 2

def expected_ledger(schedule: dict, candidates: dict) -> pd.DataFrame:
    rows = []
    for name, status in inv.stage1_names(schedule).items():
        if status != "scheduled":
            rows.append({"candidate": name, "status": status})
            continue
        c = candidates[name]
        rows.append({"candidate": name, "status": "completed", "scope": c["scope"],
                     "target_month": c["target_month"], "macro_f1": c["scores"]["macro_f1"],
                     "macro_f1_base": c["scores"]["macro_f1_base"],
                     "n_terminal": c["partition"]["n_terminal"],
                     "heldout_rows": c["rows"]["heldout_target"]})
    return pd.DataFrame(rows)


def ledger_row_problems(expected: pd.DataFrame, persisted: pd.DataFrame) -> list:
    """Row-level: the persisted ledger must equal the ledger re-derived from Stage 1."""
    problems = inv._diff("stage2 ledger candidates", expected["candidate"], persisted["candidate"])
    if problems:
        return problems
    e = expected.set_index("candidate").sort_index()
    p = persisted.set_index("candidate").sort_index()
    for col in ("status", "macro_f1", "macro_f1_base", "n_terminal", "heldout_rows"):
        if col not in p:
            problems.append(f"stage2 ledger: column {col} missing")
            continue
        ev, pv = e[col], p[col]
        if col == "status":
            bad = ev.astype(str) != pv.astype(str)
        else:
            bad = ~((ev.isna() & pv.isna()) | np.isclose(ev.astype(float), pv.astype(float), rtol=0, atol=0))
        if bad.any():
            problems.append(f"stage2 ledger: {col} differs for {list(bad[bad].index)[:5]}")
    return problems


def accept_stage2(run: Path):
    """Returns (consensus record, area->cluster map or None). Recomputes the ledger and
    weights from accepted Stage 1 evidence and requires the persisted ones to equal them."""
    from scripts.step4_similarity_matrix import compute_plan_weights
    run = Path(run)
    candidates = accept_stage1(run)
    stage2 = run / "stage2"
    record = json.loads((stage2 / "consensus.json").read_text(encoding="utf-8"))
    if record.get("code") != rid.code_identity() or record.get("runtime") != rid.runtime_identity():
        raise AcceptanceError("Stage 2 consensus was produced by different package code or runtime")
    route = record.get("route")
    if route not in ("null_consensus", "learned_map"):
        raise AcceptanceError(f"unknown consensus route {route!r}")
    required = ["candidate_ledger.csv", "plan_weights.csv"]
    if route == "learned_map":
        cluster_rel = record["cluster_map"]
        if Path(cluster_rel).is_absolute() or ".." in Path(cluster_rel).parts:
            raise AcceptanceError("cluster_map must be a path relative to stage2/")
        required += [cluster_rel, "experiment/linked_tables/main_index.csv",
                     "experiment/knn_sparsification_results/knn_analysis_report_k40_general.json"]
    _raise(rid.check_inventory(stage2, record.get("outputs") or {}, required), "Stage 2 outputs")

    expected = expected_ledger(_schedule(run), candidates)
    persisted = pd.read_csv(stage2 / "candidate_ledger.csv", float_precision="round_trip")
    _raise(ledger_row_problems(expected, persisted), "Stage 2 ledger")
    weights = compute_plan_weights(expected[expected["status"] == "completed"].copy())
    saved = pd.read_csv(stage2 / "plan_weights.csv", float_precision="round_trip")
    if not np.array_equal(saved["weight"].to_numpy(float), weights["weight"].to_numpy(float)):
        raise AcceptanceError("Stage 2 plan weights differ from weights recomputed from Stage 1")
    positive = int((weights["weight"] > 0).sum())
    if (route == "null_consensus") != (positive == 0) or record.get("positive_weight_candidates") != positive:
        raise AcceptanceError(f"consensus route {route!r} contradicts {positive} positive weights")
    if route == "null_consensus":
        return record, None
    if rid.file_sha256(stage2 / cluster_rel) != record["cluster_map_sha256"]:
        raise AcceptanceError("cluster map differs from the digest Stage 2 recorded")
    mapping = pd.read_csv(stage2 / cluster_rel, float_precision="round_trip")
    if mapping["FEWSNET_admin_code"].duplicated().any():
        raise AcceptanceError("cluster map assigns an area twice")
    return record, dict(zip(mapping["FEWSNET_admin_code"].astype(int), mapping["cluster_id"].astype(int)))


# ------------------------------------------------------------------------------ Stage 3

def route_row_problems(month: str, preds: pd.DataFrame, cluster_of, local_support: pd.DataFrame,
                       route: str) -> list:
    """Row-level: each prediction's cluster and route must equal what the consensus map and
    that cluster's single local_support decision imply."""
    problems = []
    if route == "null_consensus":
        bad = preds["partitioned_route"] != "null_consensus_pooled_reuse"
        if bad.any() or (preds["cluster_id"] != -1).any():
            problems.append(f"{month}: null-consensus rows with partition routes")
        if not np.array_equal(preds.filter(like="p_pooled_").to_numpy(), preds.filter(like="p_partitioned_").to_numpy()):
            problems.append(f"{month}: null-consensus partitioned arm is not the pooled arm")
        return problems
    expected_cluster = preds["area"].map(lambda a: cluster_of.get(int(a), -1)).astype(int)
    wrong = expected_cluster.to_numpy() != preds["cluster_id"].to_numpy()
    if wrong.any():
        problems.append(f"{month}: cluster_id differs from the consensus map for {int(wrong.sum())} rows")
    if local_support["cluster_id"].duplicated().any():
        problems.append(f"{month}: local_support has duplicate clusters")
    decision = dict(zip(local_support["cluster_id"].astype(int), local_support["route"]))
    reason = dict(zip(local_support["cluster_id"].astype(int), local_support["reason"].fillna("")))
    expected_route = []
    for cid in preds["cluster_id"].astype(int):
        if cid < 0:
            expected_route.append("unmapped_area_pooled")
        elif decision.get(cid) == LOCAL:
            expected_route.append(LOCAL)
        elif cid in decision:
            expected_route.append(f"pooled_fallback:{reason[cid]}")
        else:
            expected_route.append("<no local_support row>")
    mismatch = np.asarray(expected_route, dtype=object) != preds["partitioned_route"].to_numpy(dtype=object)
    if mismatch.any():
        rows = preds.loc[mismatch, ["area", "cluster_id", "partitioned_route"]].head(3).to_dict("records")
        problems.append(f"{month}: {int(mismatch.sum())} rows' routes disagree with local_support, e.g. {rows}")
    pooled_rows = np.asarray([r != LOCAL for r in expected_route])
    if not np.array_equal(preds.loc[pooled_rows].filter(like="p_pooled_").to_numpy(),
                          preds.loc[pooled_rows].filter(like="p_partitioned_").to_numpy()):
        problems.append(f"{month}: pooled-routed rows do not carry the pooled probabilities")
    return problems


def accept_stage3_horizon(run: Path, horizon: int, stage2=None, only_month=None,
                          out_dir: Path | None = None) -> None:
    """Folds == schedule, predicted keys == truth keys, per-fold models == routes used,
    every row's route == its cluster's decision; Stage 2 accepted underneath."""
    run = Path(run)
    record2, cluster_of = stage2 if stage2 is not None else accept_stage2(run)
    out = Path(out_dir) if out_dir is not None else run / "stage3" / f"h{horizon}"
    schedule = _schedule(run)
    if only_month:
        schedule = {**schedule, "stage3": [r for r in schedule["stage3"] if r["target_month"] == only_month]}
    records = {p.name: json.loads((p / "fold.json").read_text(encoding="utf-8"))
               for p in sorted((out / "folds").iterdir()) if p.is_dir()}
    preds = pd.read_csv(out / "predictions.csv.gz", float_precision="round_trip")
    base = pd.read_csv(run / "prepared" / "ledgers" / "baselines.csv", low_memory=False, float_precision="round_trip")
    months = [r["target_month"] for r in schedule["stage3"] if r["horizon"] == horizon]
    base = base[(base["horizon"] == horizon) & base["target_label"].isin(months)]
    problems = inv.stage3_horizon_problems(horizon, schedule, records, preds,
                                           set(zip(base["area"], base["target_label"])))
    for month, record in records.items():
        if record.get("target_month") != month or record.get("horizon") != horizon:
            problems.append(f"{month}: fold record describes another fold")
            continue
        required = list(rid.REQUIRED_STAGE3_FOLD) if record.get("status") == "fitted" else []
        problems += [f"{month}: {p}" for p in rid.check_inventory(out / "folds" / month,
                                                                  record.get("outputs") or {}, required)]
        if record.get("status") != "fitted":
            continue
        if record.get("route") != record2["route"]:
            problems.append(f"{month}: fold route {record.get('route')!r} != consensus {record2['route']!r}")
        try:
            local = pd.read_csv(out / "folds" / month / "local_support.csv", float_precision="round_trip")
        except pd.errors.EmptyDataError:
            local = pd.DataFrame(columns=["cluster_id", "route", "reason"])
        month_preds = preds[preds["target_month"] == month]
        problems += inv.stage3_fold_problems(month, record, month_preds, local)
        problems += route_row_problems(month, month_preds, cluster_of or {}, local, record2["route"])
    _raise(problems, f"h{horizon} Stage 3")


def accept_stage3(run: Path, horizons=(4, 8, 12)) -> None:
    """Complete published horizons: manifests bound to current code, predictions hash,
    fold records == fold directories, full reconciliation."""
    run = Path(run)
    stage2 = accept_stage2(run)
    for horizon in horizons:
        out = run / "stage3" / f"h{horizon}"
        manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("code") != rid.code_identity() or manifest.get("runtime") != rid.runtime_identity():
            raise AcceptanceError(f"h{horizon}: Stage 3 made by different code or runtime")
        if manifest.get("horizon") != horizon or manifest.get("only_month") or not manifest.get("fold_records"):
            raise AcceptanceError(f"h{horizon}: Stage 3 manifest is for another horizon, partial or empty")
        if rid.file_sha256(out / "predictions.csv.gz") != manifest["predictions_sha256"]:
            raise AcceptanceError(f"h{horizon}: predictions differ from the Stage 3 record")
        on_disk = {p.name for p in (out / "folds").iterdir() if p.is_dir()}
        if set(manifest["fold_records"]) != on_disk:
            raise AcceptanceError(f"h{horizon}: fold records differ from fold directories")
        for month, sha in manifest["fold_records"].items():
            if rid.file_sha256(out / "folds" / month / "fold.json") != sha:
                raise AcceptanceError(f"h{horizon} {month}: fold record changed")
        accept_stage3_horizon(run, horizon, stage2=stage2)


# ------------------------------------------------------------------------------ identity

def identity_problems(run: Path, current_code=None, current_verifier=None) -> list:
    """Run code and verifier must both equal their committed blobs at the run's git_head
    and at HEAD, and the working tree. Injected current values exist only for tests."""
    identity = json.loads((Path(run) / "prepared" / "manifests" / "identity.json").read_text(encoding="utf-8"))
    code = current_code if current_code is not None else rid.code_identity()
    verifier = current_verifier if current_verifier is not None else rid.verifier_identity()
    problems = []
    if not (identity["code"] == code == rid.code_identity_at("HEAD") == rid.code_identity_at(identity["git_head"])):
        problems.append("producer code differs between run, working tree, run git_head and HEAD")
    if not (verifier == rid.verifier_identity_at(identity["git_head"]) == rid.verifier_identity_at("HEAD")):
        problems.append("verifier differs between working tree, run git_head and HEAD")
    if not identity.get("code_equals_git_head"):
        problems.append("preparation did not record code == git_head")
    return problems
