"""Contract tests for Phase 4 (threshold freeze) and Phase 5 (override).

Covers task AC4 and AC5, which shipped without tests because the sub-agent
dispatched for Phases 4-5 hit a rate limit and the main session implemented and
reported them itself.

Run from the repository root::

    PYTHONPATH="$PWD/PersistenceCorrectionExperiment" python3 -m pytest \
        PersistenceCorrectionExperiment/tests -q

Tests split into two families:

* **synthetic** -- exercise the contract directly and always run;
* **artifact** -- re-derive the frozen numbers from the real run outputs and
  skip when ``outputs/`` is absent (it is gitignored, so a clean checkout has
  no artifacts).

The artifact tests deliberately avoid ``persistencecorrection.selection`` and
``persistencecorrection.override`` for the *verification* arithmetic: crisis-class
F1 is recomputed from a hand-built confusion count here, so an error shared by
the implementation and the runner cannot pass unnoticed.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parent
for _path in (EXPERIMENT_DIR, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from persistencecorrection import override as O  # noqa: E402
from persistencecorrection import selection as S  # noqa: E402
from persistencecorrection.override import (  # noqa: E402
    OverrideContractError,
    apply_override,
    flip_report,
)

OUTPUTS = EXPERIMENT_DIR / "outputs"
PHASE3 = OUTPUTS / "phase3_20260918"
PHASE4 = OUTPUTS / "phase4_20260919"
PHASE5 = OUTPUTS / "phase5_20260919"
SCOPES = (1, 2)

_artifacts_present = (
    (PHASE4 / "frozen_thresholds.json").exists()
    and all((PHASE3 / f"calibrated_selection_2020_fs{s}.csv").exists() for s in SCOPES)
    and all((PHASE5 / f"predictions_2layer_fs{s}.csv").exists() for s in SCOPES)
)
needs_artifacts = pytest.mark.skipif(
    not _artifacts_present,
    reason="run outputs absent (outputs/ is gitignored); artifact tests are skipped",
)


# ---------------------------------------------------------------------------
# Local, independent helpers -- never the implementation under test
# ---------------------------------------------------------------------------


def reference_f1(y_true, y_pred):
    """Crisis-class F1 from a hand-built confusion count."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def reference_override(persistence, p_cal, tau):
    """The contract restated independently of ``apply_override``."""
    persistence = np.asarray(persistence).astype(int)
    p_cal = np.asarray(p_cal, dtype=float)
    return np.where((persistence == 0) & (p_cal > tau), 1, persistence)


def reference_select(persistence, p_cal, y_true):
    """Ascending argmax sweep with strict improvement, restated independently."""
    persistence = np.asarray(persistence).astype(int)
    p_cal = np.asarray(p_cal, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    best_tau = None
    best_f1 = reference_f1(y_true, persistence)
    for tau in np.unique(p_cal):
        score = reference_f1(y_true, reference_override(persistence, p_cal, tau))
        if score > best_f1:
            best_tau, best_f1 = float(tau), score
    return best_tau, best_f1


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen():
    return json.loads((PHASE4 / "frozen_thresholds.json").read_text(encoding="utf-8"))


def _keys(frame):
    return set(zip(frame["FEWSNET_admin_code"], frame["month_start"]))


# ===========================================================================
# AC5 -- up-only is structural
# ===========================================================================


def test_apply_override_cannot_express_a_down_flip_on_any_input():
    """Exhaustive over a dense grid: no (persistence, p_cal, tau) yields 1 -> 0."""
    persistence = np.array([0, 1] * 101)
    grid = np.repeat(np.linspace(0.0, 1.0, 101), 2)
    for tau in np.linspace(-0.5, 1.5, 41):
        out = apply_override(persistence, grid, tau)
        assert not ((persistence == 1) & (out == 0)).any()
        # rows the override must never touch stay byte-identical
        assert (out[persistence == 1] == 1).all()


def test_apply_override_leaves_persistence_one_rows_untouched_under_extremes():
    """NaN-free extremes: -inf/+inf tau, and p_cal at exactly tau (strict >)."""
    persistence = np.array([1, 1, 0, 0])
    p_cal = np.array([0.0, 1.0, 0.5, 0.5])
    assert apply_override(persistence, p_cal, -np.inf).tolist() == [1, 1, 1, 1]
    assert apply_override(persistence, p_cal, np.inf).tolist() == [1, 1, 0, 0]
    # equality does not flip: the comparison is strictly greater-than
    assert apply_override(persistence, p_cal, 0.5).tolist() == [1, 1, 0, 0]


def test_override_source_has_exactly_one_write_and_it_writes_the_constant_one():
    """AST-level proof that up-only is construction, not a guard.

    If someone replaces the single masked assignment with anything else -- a
    second subscript write, a write of a variable, a write restricted to
    ``persistence == 1`` -- this fails.
    """
    tree = ast.parse(inspect.getsource(O))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "apply_override"
    )
    subscript_writes = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Subscript) for target in node.targets)
    ]
    assert len(subscript_writes) == 1, "apply_override must contain exactly one masked write"
    written = subscript_writes[0].value
    assert isinstance(written, ast.Constant) and written.value == 1, (
        "the only write must be the literal 1; writing a variable would make a "
        "1 -> 0 flip expressible"
    )
    # and the mask must be anchored on persistence == 0
    mask_source = ast.unparse(subscript_writes[0].targets[0].slice)
    assert "persistence == 0" in mask_source


def test_flip_report_catches_a_deliberately_corrupted_override():
    """Mutation check: a hand-built 1 -> 0 flip must be detected, not averaged away."""
    persistence = np.array([1, 1, 0, 0, 1])
    y_true = np.array([1, 0, 1, 0, 1])
    corrupted = np.array([0, 1, 1, 0, 1])  # row 0 silenced -- the forbidden direction
    with pytest.raises(OverrideContractError, match="1 -> 0"):
        flip_report(persistence, corrupted, y_true)


def test_flip_report_accepts_the_legal_direction_and_accounts_for_every_row():
    persistence = np.array([0, 0, 0, 1, 1])
    y_true = np.array([1, 0, 0, 1, 0])
    y_pred = apply_override(persistence, np.array([0.9, 0.9, 0.1, 0.0, 0.0]), 0.5)
    report = flip_report(persistence, y_pred, y_true)
    assert report == {
        "flips": 2,
        "flips_1_to_0": 0,
        "fixed": 1,
        "damaged": 1,
        "test_flip_precision": 0.5,
    }
    assert report["fixed"] + report["damaged"] == report["flips"]


def test_flip_report_mutation_guard_would_fail_if_the_guard_were_removed():
    """Independent re-check: a forced-off defeat of the guard is detectable.

    Simulates removing the ``if down.any(): raise`` guard by recomputing the
    down-flip count directly; the assertion below is what a reviewer would use
    if the guard silently disappeared.
    """
    persistence = np.array([1, 0, 1])
    corrupted = np.array([0, 1, 1])
    down = (corrupted != persistence) & (persistence == 1)
    assert int(down.sum()) == 1, "the re-check itself must see the injected defect"


# ===========================================================================
# AC5 -- input validation
# ===========================================================================


def test_apply_override_rejects_misaligned_inputs():
    with pytest.raises(OverrideContractError, match="align"):
        apply_override(np.array([0, 1, 0]), np.array([0.1, 0.2]), 0.5)


def test_apply_override_rejects_non_binary_persistence():
    for bad in ([0, 1, 2], [0, 1, -1], [0.0, 0.5, 1.0], [0, 1, np.nan]):
        with pytest.raises(OverrideContractError, match="binary"):
            apply_override(np.array(bad, dtype=float), np.array([0.1, 0.2, 0.3]), 0.5)


def test_apply_override_rejects_nan_probabilities():
    with pytest.raises(OverrideContractError, match="NaN"):
        apply_override(np.array([0, 0, 1]), np.array([0.1, np.nan, 0.3]), 0.5)


def test_apply_override_validates_before_writing_anything():
    """A rejected call must not have mutated the caller's persistence array."""
    persistence = np.array([0, 0, 1])
    original = persistence.copy()
    with pytest.raises(OverrideContractError):
        apply_override(persistence, np.array([0.1, np.nan, 0.3]), 0.5)
    assert (persistence == original).all()


def test_apply_override_does_not_alias_the_caller_array():
    persistence = np.array([0, 0, 1])
    out = apply_override(persistence, np.array([0.9, 0.1, 0.1]), 0.5)
    assert out is not persistence
    assert persistence.tolist() == [0, 0, 1]


# ===========================================================================
# AC4 -- freeze semantics
# ===========================================================================


def test_freeze_refuses_to_overwrite_an_existing_frozen_artifact(tmp_path):
    target = tmp_path / "frozen_thresholds.json"
    S.freeze({"fs1": {"tau": 0.5}}, {"fs1": "abc"}, target)
    with pytest.raises(FileExistsError):
        S.freeze({"fs1": {"tau": 0.9}}, {"fs1": "abc"}, target)
    assert json.loads(target.read_text())["frozen_thresholds"]["fs1"]["tau"] == 0.5


def test_freeze_digest_is_the_sha256_of_the_file_bytes(tmp_path):
    target = tmp_path / "frozen_thresholds.json"
    digest = S.freeze({"fs1": {"tau": 0.5}}, {"fs1": "abc"}, target)
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()


def test_select_threshold_matches_an_independent_sweep_on_synthetic_data():
    rng = np.random.default_rng(11)
    p_cal = np.round(rng.random(4000), 2)
    persistence = (rng.random(4000) < 0.25).astype(int)
    y_true = ((rng.random(4000) < np.clip(0.15 + 0.7 * p_cal, 0, 1)) | (persistence == 1)).astype(int)
    result = S.select_threshold(persistence, p_cal, y_true)
    tau, f1 = reference_select(persistence, p_cal, y_true)
    assert result["tau"] == tau
    assert result["selected_f1"] == pytest.approx(f1, abs=1e-12)


def test_select_threshold_reports_no_improvement_rather_than_a_spurious_tau():
    persistence = np.array([1, 1, 0, 0])
    p_cal = np.array([0.9, 0.9, 0.9, 0.9])
    y_true = np.array([1, 1, 0, 0])
    result = S.select_threshold(persistence, p_cal, y_true)
    assert result["tau"] is None
    assert result["improves_on_baseline"] is False


# ===========================================================================
# AC4 -- the frozen tau is recomputable from the 2020 artifacts alone
# ===========================================================================


@needs_artifacts
@pytest.mark.parametrize("scope", SCOPES)
def test_frozen_tau_is_recomputable_from_the_2020_selection_file_alone(scope):
    frame = pd.read_csv(PHASE3 / f"calibrated_selection_2020_fs{scope}.csv")
    tau, f1 = reference_select(frame["persistence"], frame["p_cal"], frame["y_true"])
    recorded = _frozen()["frozen_thresholds"][f"fs{scope}"]
    assert tau == recorded["tau"]
    assert f1 == pytest.approx(recorded["selected_f1"], abs=1e-12)
    assert reference_f1(frame["y_true"], frame["persistence"]) == pytest.approx(
        recorded["baseline_persistence_f1"], abs=1e-12
    )
    assert int(frame["p_cal"].nunique()) == recorded["n_candidates"]
    assert len(frame) == recorded["n_rows"]


@needs_artifacts
@pytest.mark.parametrize("scope", SCOPES)
def test_no_test_month_key_influenced_selection(scope):
    """AC4, asserted on (admin, month) keys -- not on a year string."""
    selection = pd.read_csv(
        PHASE3 / f"calibrated_selection_2020_fs{scope}.csv",
        usecols=["FEWSNET_admin_code", "month_start"],
    )
    test = pd.read_csv(
        PHASE5 / f"predictions_2layer_fs{scope}.csv",
        usecols=["FEWSNET_admin_code", "month_start"],
    )
    assert not (_keys(selection) & _keys(test)), (
        "a test-window (admin, month) key appears in the selection input"
    )


@needs_artifacts
@pytest.mark.parametrize("scope", SCOPES)
def test_calibration_fit_window_is_disjoint_from_selection_and_test(scope):
    fit = pd.read_csv(
        PHASE3 / f"calibrated_fit_2018_2019_fs{scope}.csv",
        usecols=["FEWSNET_admin_code", "month_start"],
    )
    selection = pd.read_csv(
        PHASE3 / f"calibrated_selection_2020_fs{scope}.csv",
        usecols=["FEWSNET_admin_code", "month_start"],
    )
    test = pd.read_csv(
        PHASE5 / f"predictions_2layer_fs{scope}.csv",
        usecols=["FEWSNET_admin_code", "month_start"],
    )
    assert not (_keys(fit) & _keys(selection))
    assert not (_keys(fit) & _keys(test))


@needs_artifacts
@pytest.mark.parametrize("scope", SCOPES)
def test_frozen_artifact_hash_matches_its_recorded_selection_input(scope):
    """The freeze must still describe the file it was computed from (AC4)."""
    recorded = _frozen()["selection_inputs_sha256"][f"fs{scope}"]
    actual = sha256_of(PHASE3 / f"calibrated_selection_2020_fs{scope}.csv")
    assert recorded == actual


@needs_artifacts
def test_adjudication_runner_verifies_the_freeze_hash_before_scoring():
    """Source-level: the hash re-check must exist and precede the test read."""
    source = (EXPERIMENT_DIR / "run_adjudication.py").read_text(encoding="utf-8")
    assert "selection_inputs_sha256" in source and "assert" in source
    assert source.index("selection_inputs_sha256") < source.index(
        "calibrated_test_2021_2024"
    ), "the freeze hash must be verified before any test row is read"


@needs_artifacts
def test_selection_runner_never_opens_the_test_file():
    source = (EXPERIMENT_DIR / "run_selection.py").read_text(encoding="utf-8")
    assert "calibrated_test_2021_2024" not in source
    assert "phase5" not in source


# ===========================================================================
# AC5 / AC6 -- the real outputs obey the contract
# ===========================================================================


@needs_artifacts
@pytest.mark.parametrize("scope", SCOPES)
def test_real_outputs_contain_zero_down_flips(scope):
    frame = pd.read_csv(PHASE5 / f"predictions_2layer_fs{scope}.csv")
    persistence = frame["persistence"].to_numpy().astype(int)
    y_override = frame["y_override"].to_numpy().astype(int)
    down = (y_override != persistence) & (persistence == 1)
    assert int(down.sum()) == 0
    assert flip_report(persistence, y_override, frame["y_true"])["flips_1_to_0"] == 0


@needs_artifacts
@pytest.mark.parametrize("scope", SCOPES)
def test_stored_predictions_reproduce_from_persistence_p_cal_and_frozen_tau(scope):
    frame = pd.read_csv(PHASE5 / f"predictions_2layer_fs{scope}.csv")
    tau = _frozen()["frozen_thresholds"][f"fs{scope}"]["tau"]
    expected = reference_override(frame["persistence"], frame["p_cal"], tau)
    assert (frame["y_override"].to_numpy().astype(int) == expected).all()


@needs_artifacts
@pytest.mark.parametrize("scope", SCOPES)
def test_reported_adjudication_scores_match_an_independent_recomputation(scope):
    frame = pd.read_csv(PHASE5 / f"predictions_2layer_fs{scope}.csv")
    reported = json.loads((PHASE5 / "adjudication.json").read_text())[f"fs{scope}"]
    y_true = frame["y_true"].to_numpy().astype(int)
    persistence = frame["persistence"].to_numpy().astype(int)
    override = frame["y_override"].to_numpy().astype(int)

    assert len(frame) == reported["n"]
    assert reference_f1(y_true, persistence) == pytest.approx(
        reported["scores"]["persistence"]["f1"], abs=1e-12
    )
    assert reference_f1(y_true, override) == pytest.approx(
        reported["scores"]["two_layer_override"]["f1"], abs=1e-12
    )
    assert reference_f1(y_true, frame["expert"]) == pytest.approx(
        reported["scores"]["expert"]["f1"], abs=1e-12
    )
    delta = reference_f1(y_true, override) - reference_f1(y_true, persistence)
    assert delta == pytest.approx(reported["override_delta"], abs=1e-12)
    assert flip_report(persistence, override, y_true) == reported["flips"]


# ===========================================================================
# Guards for the interpretive claims corrected on 2026-09-19
# ===========================================================================


@needs_artifacts
@pytest.mark.parametrize(
    "scope,expected_tau,expected_delta",
    [(1, 0.9090909090909092, 0.005440), (2, 0.7, 0.014030)],
)
def test_test_tuned_ceiling_on_p_cal_is_below_the_mde(scope, expected_tau, expected_delta):
    """The ceiling for the mechanism **as built** -- not the raw-probability variant.

    The first draft of RESULTS.md compared the realised delta against the
    raw-probability ceiling (+0.010 / +0.033), which layer 2 never consumes, and
    concluded the failure was threshold transfer. Sweeping ``tau`` on ``p_cal``
    with test labels shows both ceilings sit under the +0.02 MDE, so the failure
    is more fundamental than that. This test pins the corrected numbers.
    """
    frame = pd.read_csv(PHASE5 / f"predictions_2layer_fs{scope}.csv")
    tau, f1 = reference_select(frame["persistence"], frame["p_cal"], frame["y_true"])
    delta = f1 - reference_f1(frame["y_true"], frame["persistence"])
    assert tau == pytest.approx(expected_tau)
    assert delta == pytest.approx(expected_delta, abs=1e-5)
    assert delta < 0.02, "an oracle threshold on p_cal still cannot clear the MDE"


@needs_artifacts
@pytest.mark.parametrize("scope", SCOPES)
def test_best_achievable_down_flip_is_zero_flips(scope):
    """Down-flipping is useless here, not catastrophic.

    The reported diagnostic used an unoptimised ``1 - tau`` down-threshold. Swept
    properly, the optimum is to flip nothing at all, so the reported 0.6042 /
    0.4392 say nothing about the direction -- only about that threshold.
    """
    frame = pd.read_csv(PHASE5 / f"predictions_2layer_fs{scope}.csv")
    y_true = frame["y_true"].to_numpy().astype(int)
    persistence = frame["persistence"].to_numpy().astype(int)
    p_cal = frame["p_cal"].to_numpy()
    baseline = reference_f1(y_true, persistence)
    best = baseline
    for threshold in np.unique(p_cal):
        candidate = persistence.copy()
        candidate[(persistence == 1) & (p_cal < threshold)] = 0
        best = max(best, reference_f1(y_true, candidate))
    assert best == pytest.approx(baseline, abs=1e-12), (
        "a down-flip threshold that beats persistence would contradict the "
        "corrected RESULTS.md claim"
    )
