"""Supervisor P2/P3 review (pinned 791b4fd): five contract gaps and bounded numeric edges."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import ipcch_climate_geoxgb
from ipcch_climate_geoxgb import cli, learnmap, metrics, predict, projection, quartet, stage3
from ipcch_climate_geoxgb.contract import load_experiment_contract
from ipcch_climate_geoxgb.errors import ContractError, TechnicalError
from ipcch_climate_geoxgb.modelstore import ModelStore, array_digest, target_digests
from ipcch_climate_geoxgb.quartet import config_digest

CONTRACT = load_experiment_contract()


# ------------------------------------------------------------ 1. cache record integrity


def _store_entry(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    Y = np.sort(rng.random((80, 4)), axis=1)[:, ::-1]
    gp, _ = quartet.global_params(CONTRACT, "G1")
    identity = {"scope": "stage3-global", "X": array_digest(X), "params": gp, "rounds": 5,
                "n_rows": 80, "y_sha256": target_digests(Y)}
    store = ModelStore(tmp_path / "m", tmp_path / "l.jsonl")
    _, entry = store.get_or_fit(identity, lambda: quartet.fit_global_quartet(X, Y, gp, 5), {})
    d = entry["identity_sha256"]
    return store, identity, tmp_path / "m" / d[:2] / d / "record.json"


def _set_resolved(rec, path_keys, value, redigest=True):
    node = rec["resolved_config"]
    for k in path_keys[:-1]:
        node = node[k]
    node[path_keys[-1]] = value
    if redigest:
        rec["resolved_config_sha256"] = config_digest(rec["resolved_config"])


TAMPERS = {
    "requested_max_depth_99": lambda r: r["params"].update(max_depth=99),
    "resolved_max_depth_99_redigested": lambda r: _set_resolved(r, ["learner", "gradient_booster", "tree_train_param", "max_depth"], "99"),
    "resolved_eta_redigested": lambda r: _set_resolved(r, ["learner", "gradient_booster", "tree_train_param", "eta"], "0.3"),
    "resolved_seed_redigested": lambda r: _set_resolved(r, ["learner", "generic_param", "seed"], "7"),
    "resolved_edit_without_redigest": lambda r: _set_resolved(r, ["learner", "gradient_booster", "tree_train_param", "max_bin"], "256", False) or r["resolved_config"]["learner"].update(extra=1),
    "resolved_reduced": lambda r: r.update(resolved_config={"learner": {"objective": r["resolved_config"]["learner"]["objective"], "learner_model_param": r["resolved_config"]["learner"]["learner_model_param"]}}, resolved_config_sha256=None) or r.update(resolved_config_sha256=config_digest(r["resolved_config"])),
    "resolved_none": lambda r: r.update(resolved_config=None),
    "rows_negative": lambda r: r.update(rows=-1),
    "weights_nonunit": lambda r: r.update(weights="nonunit"),
    "y_sha_empty": lambda r: r.update(y_sha256=""),
    "constant_inconsistent": lambda r: r.update(constant=True),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_inconsistent_cached_fit_record_stops_without_refit(tmp_path, name):
    store, identity, path = _store_entry(tmp_path)
    record = json.loads(path.read_text())
    TAMPERS[name](record["fit_records"]["q3"])
    path.write_text(json.dumps(record))
    with pytest.raises(TechnicalError):
        store.get_or_fit(identity, lambda: pytest.fail("must not refit"), {})


def test_untampered_entry_still_hits(tmp_path):
    store, identity, _ = _store_entry(tmp_path)
    _, entry = store.get_or_fit(identity, lambda: pytest.fail("must not refit"), {})
    assert entry["status"] == "hit"


@pytest.mark.parametrize("field", ["params", "rounds", "n_rows", "y_sha256"])
def test_identity_must_bind_required_fit_fields(tmp_path, field):
    store, identity, _ = _store_entry(tmp_path)
    bad = {k: v for k, v in identity.items() if k != field}
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(40, 4)), np.sort(rng.random((40, 4)), axis=1)[:, ::-1]
    gp, _ = quartet.global_params(CONTRACT, "G1")
    with pytest.raises(TechnicalError, match="identity lacks"):
        store.get_or_fit(bad, lambda: quartet.fit_global_quartet(X, Y, gp, 5), {})


# ------------------------------------------------------------ 2/3/5 via the Stage1 end-to-end world


@pytest.fixture(scope="module")
def learned(tmp_path_factory):
    import test_p3_learnmap as p3  # same synthetic world as the P3 integration test

    run = tmp_path_factory.mktemp("rev") / "run"
    p3._prepared(run)
    mp = pytest.MonkeyPatch()
    mp.setattr(learnmap, "load_experiment_contract", p3._contract)
    learnmap.run_learn_map(run)
    mp.undo()
    return run


def test_every_child_fit_has_recoverable_members_and_rows(learned):
    run = learned
    keys = pd.read_csv(run / "prepared" / "keys_h01.csv.gz")
    split = pd.read_csv(run / "prepared" / "stage1_split.csv.gz")
    fit_keys = set(map(tuple, split.loc[split.split_role == "fit", ["admin_code", "month_ord"]].to_numpy()))
    ledger = [json.loads(x) for x in (run / "stage1" / "model_requests.jsonl").read_text().splitlines()]
    children = [e for e in ledger if e["purpose"] == "child_local"]
    assert children
    terminal_nodes = set()
    for cand in {e["candidate"] for e in children}:
        terminal_nodes |= {(cand, n) for n in pd.read_csv(run / "stage1" / "h01" / cand / "terminal_map.csv", dtype={"node_id": str})["node_id"]}
    rejected = [e for e in children if (e["candidate"], e["node_id"]) not in terminal_nodes]
    assert rejected, "the synthetic search should attempt at least one child that is not a final terminal"
    for e in children:
        rows = keys.iloc[e["prepared_rows"]]
        assert set(rows["admin_code"]) == set(e["members"])  # members resolve the exact fit rows
        assert set(map(tuple, rows[["admin_code", "target_ord"]].to_numpy())) == {
            k for k in fit_keys if k[0] in set(e["members"])}
        decisions = json.loads((run / "stage1" / "h01" / e["candidate"] / "decisions.json").read_text())["decisions"]
        parent = next(d for d in decisions if d["node_id"] == e["parent_node_id"])
        side = parent["child_ids"].index(e["node_id"])
        assert parent["child_members"][side] == e["members"]
        assert e["keys_artifact_sha256"] == json.loads((run / "prepared" / "prepared-manifest.json").read_text())["artifacts_sha256"]["keys_h01.csv.gz"]


def test_frozen_connectivity_records_full_component_sizes(learned):
    frozen = json.loads((learned / "stage1" / "frozen_h01.json").read_text())
    for region in frozen["connectivity"].values():
        sizes = region["component_sizes"]
        assert sizes == sorted(sizes, reverse=True) and sum(sizes) == region["areas"]
        assert len(sizes) == region["components"] and sizes[0] == region["largest_component"]


def test_stage1_technical_failure_leaves_durable_incomplete_record(tmp_path, monkeypatch):
    import test_p3_learnmap as p3

    run = tmp_path / "run"
    p3._prepared(run)
    monkeypatch.setattr(learnmap, "load_experiment_contract", p3._contract)

    def boom(*args, **kwargs):
        raise TechnicalError("injected global fit failure")

    monkeypatch.setattr(learnmap.quartet, "fit_global_quartet", boom)
    with pytest.raises(TechnicalError):
        learnmap.run_learn_map(run)
    incomplete = json.loads((run / "stage1" / "INCOMPLETE.json").read_text())
    assert incomplete["status"] == "incomplete" and incomplete["context"]["H"] == 1 and incomplete["context"]["G"] == "G1"
    assert "injected global fit failure" in incomplete["error"] and (run / "RUN_INCOMPLETE.json").exists()
    failed = [json.loads(x) for x in (run / "stage1" / "model_requests.jsonl").read_text().splitlines()]
    assert failed[-1]["status"] == "failed" and failed[-1]["purpose"] == "root_global" and failed[-1]["H"] == 1
    assert "stage1/model_requests.jsonl" in incomplete["partial_evidence"]
    monkeypatch.setattr(ipcch_climate_geoxgb, "RUNS_DIR", tmp_path)
    with pytest.raises(ContractError, match="incomplete"):
        cli.existing_run("run")


def test_stage3_technical_failure_records_fold_region_and_failed_request(tmp_path, monkeypatch):
    import test_p4_stage3 as p4

    ctx = p4._ctx(tmp_path)

    def boom(*args, **kwargs):
        raise TechnicalError("injected local failure")

    monkeypatch.setattr(stage3.quartet, "continue_local_quartet", boom)
    with pytest.raises(TechnicalError) as caught:
        stage3.run_fold(ctx, p4._fold(p4.M0 + 50))
    assert any("region" in n and "gate month" in n for n in caught.value.__notes__)
    entries = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    failed = [e for e in entries if e["status"] == "failed"]
    assert failed and failed[0]["purpose"] == "local" and "region" in failed[0] and failed[0]["fold"] == f"t_{p4.M0 + 50}"


def test_predict_failure_records_fold_context(tmp_path, monkeypatch):
    import test_p5_e2e as p5

    run = tmp_path / "run"
    p5._write_prepared(run)
    monkeypatch.setattr(learnmap, "load_experiment_contract", p5._contract)
    monkeypatch.setattr(predict, "load_experiment_contract", p5._contract)
    learnmap.run_learn_map(run)
    calls = {"n": 0}
    real = stage3.run_fold

    def flaky(ctx, fold):
        calls["n"] += 1
        if calls["n"] == 2:
            raise TechnicalError("injected fold failure")
        return real(ctx, fold)

    monkeypatch.setattr(predict.stage3, "run_fold", flaky)
    with pytest.raises(TechnicalError):
        predict.run_predict(run)
    incomplete = json.loads((run / "stage3" / "INCOMPLETE.json").read_text())
    assert incomplete["context"]["fold_id"] == "main_h03_2003-06" and incomplete["context"]["H"] == 3
    assert any(p.endswith("gate_decisions.jsonl") for p in incomplete["partial_evidence"])


# ------------------------------------------------------------ 4. prepared inventory


def test_prepared_inventory_omission_or_extra_stops(tmp_path):
    import test_p3_learnmap as p3

    run = tmp_path / "run"
    p3._prepared(run)
    manifest_path = run / "prepared" / "prepared-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert learnmap.verify_prepared(run / "prepared", [1])
    omitted = dict(manifest, artifacts_sha256={k: v for k, v in manifest["artifacts_sha256"].items() if k != "keys_h01.csv.gz"})
    manifest_path.write_text(json.dumps(omitted))
    (run / "prepared" / "keys_h01.csv.gz").write_bytes(b"changed")
    with pytest.raises(TechnicalError, match="missing \\['keys_h01.csv.gz'\\]"):
        learnmap.verify_prepared(run / "prepared", [1])
    extra = dict(manifest, artifacts_sha256={**manifest["artifacts_sha256"], "other.csv": "0" * 64})
    manifest_path.write_text(json.dumps(extra))
    with pytest.raises(TechnicalError, match="unexpected"):
        learnmap.verify_prepared(run / "prepared", [1])


# ------------------------------------------------------------ bounded numeric/digest edges


def test_structured_object_dtype_cannot_be_digested():
    with pytest.raises(TechnicalError):
        array_digest(np.array([(1, "a")], dtype=[("i", "i8"), ("o", "O")]))


def test_projection_overflowing_sum_has_bounded_answer():
    z = projection.project(np.array([[1e308, 1.5e308, 0.9, 0.1]]))
    np.testing.assert_allclose(z[0], [1.0, 1.0, 0.9, 0.1])
    ordinary = np.array([[0.3, 0.5, 0.1, 0.0]])  # ordinary values unchanged
    np.testing.assert_allclose(projection.project(ordinary)[0], [0.4, 0.4, 0.1, 0.0])


@pytest.mark.parametrize("truth, pred", [([0.0, 1e-170], [0.0, 0.0]), ([0.1, 0.3], [0.1, 1e160])])
def test_r2_out_of_range_is_na_not_inf(truth, pred):
    value, reason = metrics.r_squared(truth, pred)
    assert value is None and "float64" in reason
    assert metrics.r_squared([0.1, 0.2, 0.3], [0.3, 0.1, 0.5])[0] < 0  # ordinary negative kept
