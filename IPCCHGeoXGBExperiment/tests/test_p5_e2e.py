"""P5 end to end: learn-map -> predict -> report on a synthetic run, then independent replay
must pass, and each tampering class must be caught by the replay (implement.md P5)."""

from __future__ import annotations

import copy
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from ipcch_geoxgb import learnmap, predict, report
from ipcch_geoxgb.artifacts import sha256_file, write_json
from ipcch_geoxgb.contract import load_experiment_contract
from ipcch_geoxgb.replay import replay_run

H = 3
N_AREAS = 80
M0, N_MONTHS = 24000, 48
GZ = {"method": "gzip", "mtime": 0}


def _contract():
    c = copy.deepcopy(load_experiment_contract())
    c["calendar"]["horizons_months"] = [H]
    c["model"]["global_recipes"] = {"G1": {"max_depth": 3, "rounds": 60}}
    c["model"]["local_recipes"] = {"L1": {"max_depth": 1, "appended_rounds": 40}}
    c["support"]["local_fit"] = {"keys": 60, "areas": 8, "target_months": 3}
    c["support"]["validation"] = {"keys": 40, "areas": 8, "target_months": 3, "crisis_keys": 5, "noncrisis_keys": 5}
    c["partition"]["scan_iterations"] = 30
    c["partition"]["max_member_depth"] = 1
    return c


def _write_prepared(run):
    rng = np.random.default_rng(11)
    prepared = run / "prepared"
    prepared.mkdir(parents=True)
    rows = [(a, M0 + m) for a in range(N_AREAS) for m in range(N_MONTHS)]
    keys = pd.DataFrame(rows, columns=["admin_code", "target_ord"])
    keys["target_month"] = [f"{o // 12:04d}-{o % 12 + 1:02d}" for o in keys["target_ord"]]
    x0 = rng.normal(size=len(keys))
    west = keys["admin_code"].to_numpy() < N_AREAS // 2
    q3 = np.clip(0.2 + np.where(west, 0.2, -0.2) * x0 + rng.normal(0, 0.02, len(keys)), 0, 0.9)
    keys["q2"], keys["q3"], keys["q4"], keys["q5"] = np.minimum(q3 + 0.15, 1), q3, q3 * 0.4, q3 * 0.05
    phase = 1 + (keys[["q2", "q3", "q4", "q5"]].to_numpy() >= 0.2).sum(axis=1)
    keys["phase_truth"], keys["crisis_truth"] = phase, (phase >= 3).astype(int)
    keys["country_key"] = np.where(keys["admin_code"] % 2 == 0, "A", "B")
    keys["horizon_months"], keys["origin_ord"] = H, keys["target_ord"] - H
    # lawful persistence: the same area's observation at the origin month (age 0); none before M0
    prev = keys[["admin_code", "target_ord", "phase_truth", "q3"]].rename(
        columns={"target_ord": "origin_ord", "phase_truth": "persistence_phase", "q3": "persistence_q3"})
    keys = keys.merge(prev, on=["admin_code", "origin_ord"], how="left")
    keys["persistence_available"] = keys["persistence_phase"].notna().astype(int)
    keys["persistence_phase"] = keys["persistence_phase"].fillna(0).astype(int)
    keys["persistence_source_month"] = np.where(keys["persistence_available"] == 1,
                                                [f"{o // 12:04d}-{o % 12 + 1:02d}" for o in keys["origin_ord"]], "")
    keys["persistence_age_months"] = np.where(keys["persistence_available"] == 1, 0, -1)
    X = np.full((len(keys), 561), np.nan)
    X[:, 0], X[:, 1] = x0, rng.normal(size=len(keys))
    np.save(prepared / f"X_rich561_h{H:02d}.npy", X)
    keys.to_csv(prepared / f"keys_h{H:02d}.csv.gz", index=False, compression=GZ)
    split = keys[["admin_code", "target_ord", "phase_truth", "crisis_truth"]].rename(columns={"target_ord": "month_ord"})
    split = split[split["month_ord"] < M0 + 30]
    split["split_role"] = np.where(split["month_ord"] < M0 + 15, "fit", "validation")
    split.to_csv(prepared / "stage1_split.csv.gz", index=False, compression=GZ)
    folds = [{"period": "main", "horizon_months": H, "target_ord": t, "origin_ord": t - H} for t in range(M0 + 40, M0 + 49)]
    cal = pd.DataFrame(folds)
    cal["target_month"] = [f"{o // 12:04d}-{o % 12 + 1:02d}" for o in cal["target_ord"]]
    cal["fold_id"] = [f"main_h{H:02d}_{m}" for m in cal["target_month"]]
    cal.to_csv(prepared / "fold_calendar.csv", index=False)  # last fold (M0+48) has no keys -> ledger only
    names = [f"X_rich561_h{H:02d}.npy", f"keys_h{H:02d}.csv.gz", "stage1_split.csv.gz", "fold_calendar.csv"]
    write_json(prepared / "prepared-manifest.json",
               {"artifacts_sha256": {n: sha256_file(prepared / n) for n in names},
                "availability_policy": {"id": "observation-month-end-v1"}})


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    for module in (learnmap, predict):
        mp.setattr(module, "load_experiment_contract", _contract)
    root = tmp_path_factory.mktemp("e2e") / "run"
    _write_prepared(root)
    learnmap.run_learn_map(root)
    predict.run_predict(root)
    report.run_report(root)
    mp.undo()
    return root


def test_complete_run_replays_cleanly(run):
    result = replay_run(run, _contract())
    assert result["status"] == "passed", result["failures"][:10]
    ledger = pd.read_csv(run / "stage3" / f"h{H:02d}" / "fold_ledger.csv")
    assert ledger["status"].tolist().count("no_valid_target") == 1
    pred = pd.read_csv(run / "stage3" / f"h{H:02d}" / "predictions.csv.gz")
    assert len(pred) == 8 * N_AREAS
    # adoption is data-dependent (shown in the P4 unit tests); here every region's gate is
    # decided on fitted historical local models, which the tamper tests rely on
    gates = [json.loads(x) for x in (run / "stage3" / f"h{H:02d}" / "gate_decisions.jsonl").read_text().splitlines()]
    assert gates and all(g["local_fit_dates"] >= 3 for g in gates)
    requests = [json.loads(x) for x in (run / "stage3" / "model_requests.jsonl").read_text().splitlines()]
    assert any(r["purpose"] == "local" for r in requests)
    rep = json.loads((run / "report" / "report.json").read_text())
    boot = rep["horizons"][str(H)]["main"]["bootstrap"]
    assert set(boot) == {"geo_vs_pool_E_all", "geo_vs_persistence_E_persist"}
    for record in boot.values():
        assert record["K"] == 2 and record["draws"] == 2000 and record["seed"] == 42
        assert (record["interval"] is None) == bool(record["na_reason"])
    assert rep["horizons"][str(H)]["supplementary"]["coverage"]["E_all_keys"] == 0


def _tamper(run, tmp_path, edit):
    copy_dir = tmp_path / "tampered"
    shutil.copytree(run, copy_dir)
    edit(copy_dir)
    return replay_run(copy_dir, _contract())


def _rewrite_predictions(root, change):
    path = root / "stage3" / f"h{H:02d}" / "predictions.csv.gz"
    pred = change(pd.read_csv(path))
    pred.to_csv(path, index=False, compression=GZ)
    summary_path = root / "stage3" / "stage3-summary.json"
    summary = json.loads(summary_path.read_text())
    summary["horizons"][str(H)]["predictions_sha256"] = sha256_file(path)  # digest refreshed: deeper checks must fire
    summary_path.write_text(json.dumps(summary))


def _failed(result, prefix):
    return any(f.split(":")[0].endswith(prefix) for f in result["failures"])


def test_dropped_row_is_caught(run, tmp_path):
    result = _tamper(run, tmp_path, lambda r: _rewrite_predictions(r, lambda p: p.iloc[1:]))
    assert _failed(result, "fold_keys")


def test_wrong_target_order_is_caught(run, tmp_path):
    def swap(p):
        p["geo_q2_raw"], p["geo_q5_raw"] = p["geo_q5_raw"].copy(), p["geo_q2_raw"].copy()
        return p
    result = _tamper(run, tmp_path, lambda r: _rewrite_predictions(r, swap))
    assert _failed(result, "geo_projection") or _failed(result, "non_local_equals_pooled")


def test_future_information_is_caught(run, tmp_path):
    def future(p):
        i = p.index[p["persistence_available"] == 1][0]
        o = int(p.loc[i, "origin_ord"]) + 1
        p.loc[i, "persistence_source_month"] = f"{o // 12:04d}-{o % 12 + 1:02d}"
        return p
    result = _tamper(run, tmp_path, lambda r: _rewrite_predictions(r, future))
    assert _failed(result, "persistence_not_after_origin")


def test_stale_gate_is_caught(run, tmp_path):
    def flip(root):
        path = root / "stage3" / f"h{H:02d}" / "gate_decisions.jsonl"
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        lines[0]["enabled"] = not lines[0]["enabled"]
        path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    result = _tamper(run, tmp_path, flip)
    assert _failed(result, "gate_decision")


def test_inherited_local_prefix_is_caught(run, tmp_path):
    def reparent(root):
        for path in (root / "models").glob("*/*/record.json"):
            record = json.loads(path.read_text())
            if record["identity"]["scope"] == "stage3-local":
                record["fit_records"]["q3"]["parent_booster_sha256"] = "0" * 64
                path.write_text(json.dumps(record))
                return
        pytest.fail("no Stage3 local model in the synthetic run")
    result = _tamper(run, tmp_path, reparent)
    assert _failed(result, "local_prefix_is_global")


def test_report_count_tampering_is_caught(run, tmp_path):
    def bump(root):
        path = root / "report" / "report.json"
        rep = json.loads(path.read_text())
        rep["horizons"][str(H)]["main"]["E_all"]["geo"]["binary"]["counts"]["tp"] += 1
        path.write_text(json.dumps(rep))
    result = _tamper(run, tmp_path, bump)
    assert _failed(result, "report.counts")


def test_model_target_order_swap_is_caught(run, tmp_path):
    def swap(root):
        for path in (root / "models").glob("*/*/record.json"):
            record = json.loads(path.read_text())
            if record["identity"]["scope"] == "stage3-global":
                fr = record["fit_records"]
                fr["q2"]["y_sha256"], fr["q5"]["y_sha256"] = fr["q5"]["y_sha256"], fr["q2"]["y_sha256"]
                path.write_text(json.dumps(record))
                return
    result = _tamper(run, tmp_path, swap)
    assert _failed(result, "models.target_order")
