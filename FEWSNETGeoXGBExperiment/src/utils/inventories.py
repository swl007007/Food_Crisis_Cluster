"""Required inventories derived from independent evidence (audit repair round 3).

A completion record may list only what it wants to; so membership is never taken from
the record being checked. Each function returns a list of problems (empty = accepted),
and compares BOTH directions against the authoritative source:

* Stage 3 horizon: fold records == the prepared schedule (months and statuses);
  prediction keys == baseline truth keys.
* Stage 3 fold: saved models, estimators and local-support rows == the routes the
  saved predictions actually used.
* Stage 1 retained checkpoints: one checkpoint per branch in the retained s_branch.
* Stage 1 population / Stage 2 ledger: exactly the scheduled folds.
"""
from __future__ import annotations

import pandas as pd

LOCAL_ROUTE = "local_model"
SPACE_PARTITIONS = ("s_branch.pkl", "branch_table.npy", "X_branch_id.npy")


def _diff(label, expected, actual) -> list:
    expected, actual = set(expected), set(actual)
    return ([f"{label}: missing {sorted(expected - actual)[:5]}"] if expected - actual else []) + \
           ([f"{label}: unexpected {sorted(actual - expected)[:5]}"] if actual - expected else [])


def stage1_names(schedule: dict) -> dict:
    """Scheduled Stage 1 fold name -> status."""
    return {f"fs{f['scope']}_{f['target_month']}": f["status"] for f in schedule["stage1"]}


def stage1_population_problems(schedule: dict, fold_dirs) -> list:
    scheduled = [n for n, s in stage1_names(schedule).items() if s == "scheduled"]
    return _diff("stage1 folds", scheduled, fold_dirs)


def stage2_ledger_problems(schedule: dict, ledger: pd.DataFrame) -> list:
    expected = stage1_names(schedule)
    problems = _diff("stage2 ledger", expected, ledger["candidate"])
    for name, status in zip(ledger["candidate"], ledger["status"]):
        want = "completed" if expected.get(name) == "scheduled" else expected.get(name)
        if name in expected and status != want:
            problems.append(f"stage2 ledger: {name} status {status!r} != {want!r}")
    return problems


def retained_checkpoint_problems(name: str, outputs: dict, s_branch_columns) -> list:
    """Every branch in the retained s_branch needs its checkpoint (routing loads them)."""
    base = f"retained/{name}"
    required = [f"{base}/space_partitions/{f}" for f in SPACE_PARTITIONS]
    required += [f"{base}/checkpoints/rf_{branch}" for branch in s_branch_columns]
    return [f"{name}: required {rel} not recorded" for rel in required if rel not in outputs]


def stage3_horizon_problems(horizon: int, schedule: dict, fold_records: dict,
                            predictions: pd.DataFrame, baseline_keys) -> list:
    rows = [r for r in schedule["stage3"] if r["horizon"] == horizon]
    expected = {r["target_month"]: r["status"] for r in rows}
    problems = _diff(f"h{horizon} fold records", expected, fold_records)
    for month, record in fold_records.items():
        want = "fitted" if expected.get(month) == "scheduled" else expected.get(month)
        if month in expected and record.get("status") != want:
            problems.append(f"h{horizon} {month}: status {record.get('status')!r} != {want!r}")
    keys = set(zip(predictions["area"], predictions["target_month"]))
    problems += _diff(f"h{horizon} prediction keys", baseline_keys, keys)
    fitted = {m for m, r in fold_records.items() if r.get("status") == "fitted"}
    problems += _diff(f"h{horizon} predicted months", fitted, predictions["target_month"].unique())
    return problems


def stage3_fold_problems(month: str, record: dict, month_predictions: pd.DataFrame,
                         local_support: pd.DataFrame | None) -> list:
    """Models, estimators and support rows must equal the routes the predictions used."""
    problems = []
    if len(month_predictions) != record.get("rows", {}).get("test"):
        problems.append(f"{month}: {len(month_predictions)} prediction rows != record {record.get('rows')}")
    routed = sorted({int(c) for c, r in zip(month_predictions["cluster_id"], month_predictions["partitioned_route"])
                     if r == LOCAL_ROUTE})
    estimators = ["pooled"] + [f"local_{c}" for c in routed]
    problems += _diff(f"{month} estimators", estimators, record.get("estimators", {}))
    problems += _diff(f"{month} saved models", [f"models/{e}.pkl.xz" for e in estimators],
                      [k for k in record.get("outputs", {}) if k.startswith("models/")])
    mapped = sorted({int(c) for c in month_predictions["cluster_id"] if int(c) >= 0})
    if record.get("route") == "learned_map":
        if local_support is None or "cluster_id" not in local_support:
            return problems + [f"{month}: local_support missing for a learned map"]
        problems += _diff(f"{month} local_support clusters", mapped, local_support["cluster_id"])
        supported = local_support.loc[local_support["route"] == LOCAL_ROUTE, "cluster_id"]
        problems += _diff(f"{month} local_support local fits", routed, supported)
    elif routed or mapped:
        problems.append(f"{month}: null-consensus fold has partition routes")
    return problems
