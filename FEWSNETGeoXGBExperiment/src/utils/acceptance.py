"""One acceptance chain for every persisted stage output of the GeoXGBoost experiment.

Every consumer (Stage 1 collectors, map builders, development/final drivers, report,
verifier) accepts upstream artifacts ONLY through these functions. Each re-derives the
expected population from the prepared schedule (never from the record being checked),
checks the record's code/runtime/upstream identity and every recorded output hash, and
raises AcceptanceError on any problem.

Chain: prepared -> G selection -> Stage 1 candidates -> maps -> development folds ->
frozen selection -> final folds.

Every persisted CSV is read with float_precision="round_trip" (exact comparisons).
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment import plan
from src.utils import run_identity as rid


class AcceptanceError(RuntimeError):
    pass


def _raise(problems, where):
    if problems:
        raise AcceptanceError(f"{where}: {problems[:8]}")


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def schedule(run: Path) -> dict:
    return _json(Path(run) / "prepared" / "manifests" / "schedule.json")


def _identity_problems(record: dict, where: str) -> list:
    problems = []
    if record.get("code") != rid.code_identity():
        problems.append(f"{where}: produced by different package code")
    if record.get("runtime") != rid.runtime_identity():
        problems.append(f"{where}: produced by a different runtime")
    return problems


def accept_record(base: Path, record_name: str, required=()) -> dict:
    """A completion record (written last) whose identity is current and whose every
    recorded output exists with its hash; ``required`` files must be recorded."""
    base = Path(base)
    path = base / record_name
    if not path.is_file():
        raise AcceptanceError(f"{path}: no completion record")
    record = _json(path)
    problems = _identity_problems(record, str(path))
    problems += rid.check_inventory(base, record.get("outputs") or {}, required)
    _raise(problems, str(path))
    return record


# ------------------------------------------------------------------------------ prepared

def accept_prepared(run: Path) -> dict:
    try:
        return rid.require_prepared(run)
    except RuntimeError as exc:
        raise AcceptanceError(str(exc)) from exc


# ------------------------------------------------------------------------------ G selection

def accept_g_selection(run: Path):
    """({'4': G, '8': G, '12': G}, completion-record sha) from RUN/gscreen."""
    run = Path(run)
    prepared = accept_prepared(run)
    record = accept_record(run / "gscreen", "selection.json", ["scores.csv", "predictions.csv.gz"])
    if record.get("prepared") != prepared["outputs_sha256"]:
        raise AcceptanceError("G selection was made on another preparation")
    expected = {str(h) for h in plan.HORIZONS}
    if set(record["selected"]) != expected or not set(record["selected"].values()) <= set(plan.G_CONFIGS):
        raise AcceptanceError("G selection does not name one frozen G per horizon")
    scores = pd.read_csv(run / "gscreen" / "scores.csv", float_precision="round_trip", dtype={"macro_f1_exact": str})
    if len(scores) != len(plan.G_CONFIGS) * len(plan.HORIZONS):
        raise AcceptanceError("G screening does not cover the 4 x 3 configurations")
    for h in plan.HORIZONS:
        rows = scores[scores["horizon"] == h]
        best = max(rows.itertuples(), key=lambda r: (Fraction(r.macro_f1_exact),
                                                     tuple(-x for x in plan.g_tiebreak_key(r.g_config))))
        if best.g_config != record["selected"][str(h)]:
            raise AcceptanceError(f"h{h}: selected G differs from the declared selection rule")
    return record["selected"], rid.file_sha256(run / "gscreen" / "selection.json")


# ------------------------------------------------------------------------------ Stage 1

def accept_stage1(run: Path) -> dict:
    """Every scheduled root with a verified completion record, and EXACTLY the 648
    scheduled candidates; returns {candidate: summary} including status."""
    from scripts.run_stage1 import CANDIDATE_FILES, ROOT_FILES, scheduled_candidates, scheduled_roots
    run = Path(run)
    prepared = accept_prepared(run)
    g_of, g_record = accept_g_selection(run)
    sched = schedule(run)
    roots = scheduled_roots(sched, g_of)
    expected = scheduled_candidates(sched, g_of)
    stage1 = run / "stage1"
    present = {p.name for p in (stage1 / "roots").iterdir() if p.is_dir()} if (stage1 / "roots").is_dir() else set()
    problems = [f"missing root {n}" for n in sorted(set(roots) - present)]
    problems += [f"unexpected root {n}" for n in sorted(present - set(roots))]
    out, seen = {}, set()
    for name in sorted(set(roots) & present):
        try:
            record = _json(stage1 / "roots" / name / "completion.json")
        except FileNotFoundError:
            problems.append(f"{name}: no completion record")
            continue
        problems += _identity_problems(record, name)
        if record.get("root") != name or record.get("prepared") != prepared["outputs_sha256"] \
                or record.get("g_selection") != g_record:
            problems.append(f"{name}: record belongs to another root, preparation or G selection")
        want = sorted(c for c, e in expected.items() if e["root"] == name)
        if sorted(record.get("candidates", [])) != want:
            problems.append(f"{name}: candidate list differs from the schedule")
        required = [f"roots/{name}/{f}" for f in ROOT_FILES]
        if record.get("status") == "completed":
            required += [f"candidates/{c}/{f}" for c in want for f in CANDIDATE_FILES]
        elif record.get("status") != "root_insufficient_support":
            problems.append(f"{name}: unknown status {record.get('status')!r}")
        problems += [f"{name}: {p}" for p in rid.check_inventory(stage1, record.get("outputs") or {}, required)]
        root = _json(stage1 / "roots" / name / "root.json")
        for cand in want:
            seen.add(cand)
            entry = {**expected[cand], "name": cand, "status": record.get("status")}
            if record.get("status") == "completed":
                c = _json(stage1 / "candidates" / cand / "candidate.json")
                scores = [c["scores"]["macro_f1"], c["scores"]["macro_f1_base"]]
                if not np.isfinite(scores).all():
                    problems.append(f"{cand}: non-finite held-out score")
                if c["candidate"] != cand or c["local_config"] != entry["local_config"] \
                        or c["threshold_family"] != entry["threshold_family"]:
                    problems.append(f"{cand}: candidate.json describes another candidate")
                entry.update(macro_f1=scores[0], macro_f1_base=scores[1],
                             n_terminal=c["partition"]["n_terminal"],
                             correspondence=str(stage1 / "candidates" / cand / "correspondence_table.csv"),
                             correspondence_sha256=record["outputs"][f"candidates/{cand}/correspondence_table.csv"],
                             heldout_rows=root["rows"]["heldout_target"])
            out[cand] = entry
    problems += [f"candidate {c} has no root record" for c in sorted(set(expected) - seen)]
    _raise(problems, "Stage 1")
    return out


def candidate_frame(candidates: dict) -> tuple[pd.DataFrame, dict]:
    """Completed candidates as a Stage 2 ledger frame plus name -> correspondence path."""
    rows = [{"name": n, "horizon": c["horizon"], "target_month": c["target_month"],
             "macro_f1": c["macro_f1"], "macro_f1_base": c["macro_f1_base"], "n_terminal": c["n_terminal"],
             "correspondence_sha256": c["correspondence_sha256"], "source": c.get("source", "geoxgb_stage1")}
            for n, c in candidates.items() if c["status"] == "completed"]
    frame = pd.DataFrame(rows, columns=["name", "horizon", "target_month", "macro_f1", "macro_f1_base",
                                        "n_terminal", "correspondence_sha256", "source"])
    return frame, {n: Path(c["correspondence"]) for n, c in candidates.items() if c["status"] == "completed"}


# ------------------------------------------------------------------------------ v7 RF candidates

V7_RUN = Path(rid.PACKAGE).parent / "FEWSNETFourClassBaseline" / "runs" / "fourclass-v7-20260928"
#: committed v7 Stage 2 final map (the fixed old-map control in the final evaluation)
V7_FINAL_MAP = V7_RUN / "stage2" / "experiment" / "knn_sparsification_results" / "cluster_mapping_k40_nc13_general.csv"
V7_FINAL_MAP_SHA256 = "fc2c919f3db6a8a8a77e6c8e2a1a6d2fdc65ee4996eac20300523c985ea91b46"


def v7_candidates() -> dict:
    """The 27 committed v7 RF candidates (historical producer fe40de37, 35-month window),
    read-only from their committed correspondence tables and candidate.json scores."""
    import subprocess
    folds = V7_RUN / "stage1" / "folds"
    tracked = set(subprocess.run(["git", "ls-files", str(folds)], cwd=rid.PACKAGE.parent, capture_output=True,
                                 text=True, check=True).stdout.splitlines())
    out = {}
    for fold in sorted(p for p in folds.iterdir() if p.is_dir()):
        corr = fold / "correspondence_table.csv"
        rel = corr.relative_to(rid.PACKAGE.parent).as_posix()
        if rel not in tracked:
            raise AcceptanceError(f"{rel} is not committed evidence")
        status = subprocess.run(["git", "status", "--porcelain", "--", rel, str(fold / "candidate.json")],
                                cwd=rid.PACKAGE.parent, capture_output=True, text=True, check=True).stdout
        if status.strip():
            raise AcceptanceError(f"{rel} differs from its committed version")
        c = _json(fold / "candidate.json")
        horizon = {1: 4, 2: 8, 3: 12}[int(c["scope"])]
        out[f"v7_{fold.name}"] = {
            "name": f"v7_{fold.name}", "status": "completed", "horizon": horizon,
            "target_month": c["target_month"], "macro_f1": c["scores"]["macro_f1"],
            "macro_f1_base": c["scores"]["macro_f1_base"], "n_terminal": c["partition"]["n_terminal"],
            "correspondence": str(corr), "correspondence_sha256": rid.file_sha256(corr), "source": "v7_rf_stage1"}
    if len(out) != 27:
        raise AcceptanceError(f"expected the 27 committed v7 candidates, found {len(out)}")
    return out


def v7_final_map() -> dict:
    if rid.file_sha256(V7_FINAL_MAP) != V7_FINAL_MAP_SHA256:
        raise AcceptanceError("v7 final map differs from its committed digest")
    mapping = pd.read_csv(V7_FINAL_MAP)
    return dict(zip(mapping["FEWSNET_admin_code"].astype(int), mapping["cluster_id"].astype(int)))


# ------------------------------------------------------------------------------ folds

FOLD_FILES = ("predictions.csv.gz", "gate.json")


def accept_fold(fold_dir: Path, expected: dict) -> dict:
    """One external fold of one arm: completion record (fold.json, written last), its
    identity fields equal ``expected`` and every output present with its hash."""
    record = accept_record(fold_dir, "fold.json")
    if record.get("status") not in ("fitted", "incomplete_coverage_gate"):
        raise AcceptanceError(f"{fold_dir}: unknown fold status {record.get('status')!r}")
    if record["status"] == "fitted":  # a coverage-blocked fold legitimately has no predictions
        accept_record(fold_dir, "fold.json", FOLD_FILES)
    bad = {k: (record.get(k), v) for k, v in expected.items() if record.get(k) != v}
    if bad:
        raise AcceptanceError(f"{fold_dir}: identity differs {bad}")
    return record


# ------------------------------------------------------------------------------ identity

def identity_problems(run: Path, current_code=None, current_verifier=None) -> list:
    """Run code and verifier must both equal their committed blobs at the run's git_head
    and at HEAD, and the working tree. Injected current values exist only for tests."""
    identity = _json(Path(run) / "prepared" / "manifests" / "identity.json")
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
