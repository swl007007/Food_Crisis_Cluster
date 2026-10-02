"""The frozen first-round experiment table (task experiment-plan.md v1.0, D24/D25).

Every number here is a pre-declared design constant, not a tuned value. Changing any
of them is a new experiment, not a repair of this one.
"""
from __future__ import annotations

from fractions import Fraction

HORIZONS = (4, 8, 12)

#: D26 (2026-10-01): native four-class probabilities, binary crisis evaluation
#: (argmax collapsed to IPC >= 3); fixed-four macro F1 is secondary.
ENDPOINT = "crisis_f1"
#: D26: Stage 2/3 metric alignment awaits user review of the Stage 1 diagnostics.
DOWNSTREAM_ALIGNED = False
SCOPE_OF = {4: 1, 8: 2, 12: 3}
N_CLASSES = 4

#: Actual label window [O-59, O) for every global, parent and local fit (D9 revised).
WINDOW = 59

#: Native xgb.train settings shared by every booster (plan section 2).
XGB_BASE = {
    "booster": "gbtree", "objective": "multi:softprob", "num_class": N_CLASSES,
    "multi_strategy": "one_output_per_tree", "num_parallel_tree": 1, "tree_method": "hist",
    "device": "cpu", "seed": 42, "nthread": 4,
}
_G = {"eta": 0.05, "min_child_weight": 10, "reg_lambda": 10, "reg_alpha": 0, "subsample": 0.8,
      "colsample_bytree": 0.8}
_L = {"eta": 0.05, "min_child_weight": 20, "reg_lambda": 20, "reg_alpha": 1, "subsample": 1,
      "colsample_bytree": 1}
G_CONFIGS = {
    "G1": {"max_depth": 3, "rounds": 200, **_G},
    "G2": {"max_depth": 3, "rounds": 400, **_G},
    "G3": {"max_depth": 4, "rounds": 200, **_G},
    "G4": {"max_depth": 4, "rounds": 400, **_G},
}
L_CONFIGS = {
    "L1": {"max_depth": 1, "rounds": 20, **_L},
    "L2": {"max_depth": 2, "rounds": 40, **_L},
}
#: Stage 1 path ceiling: rounds appended after the global root along one path.
PATH_ROUND_CAP = 80

#: Stage 1 within-area random split: tag -> validation share (D20).
SPLIT_RATIOS = {"r80": 0.20, "r50": 0.50}
SPLIT_SEEDS = (42, 43, 44)
#: D27 (experiment-plan A2): the time-block split identity. In the root's legal
#: [O-59, O) pool, the latest TIME_BLOCK_MONTHS distinct observed label months are the
#: common E1/E2 validation block (same months for every area); every earlier legal row
#: is fitting. Not a random split and not an internal rolling-origin replay.
TIME_BLOCK = "tb3"
TIME_BLOCK_MONTHS = 3
#: The bounded D27 contrast: 2 targets x 3 H, L1 + gt0 only, split/model seed 42.
TB3_TARGETS = ("2018-02", "2020-10")
TB3_LOCAL = "L1"
TB3_FAMILY = "gt0"
TB3_SEED = 42
#: D27 locks the D26 development crisis-F1 G selection (no reselection for tb3).
TB3_G = {"4": "G1", "8": "G4", "12": "G2"}
#: A2 pre-checked validation months per (H, T); a computed block that differs is an
#: identity/calendar error, never a reason to pick another block.
TB3_VALIDATION_MONTHS = {
    (4, "2018-02"): ("2016-10", "2017-02", "2017-06"),
    (8, "2018-02"): ("2016-06", "2016-10", "2017-02"),
    (12, "2018-02"): ("2016-02", "2016-06", "2016-10"),
    (4, "2020-10"): ("2019-06", "2019-10", "2020-02"),
    (8, "2020-10"): ("2019-02", "2019-06", "2019-10"),
    (12, "2020-10"): ("2018-10", "2019-02", "2019-06"),
}
#: D23: two E2 acceptance families, strict gain above the threshold; parent wins ties.
THRESHOLD_FAMILIES = {"gt0": Fraction(0), "gt001": Fraction(1, 100)}
#: D13: Stage 3 local activation, strict gain above .01 on the pooled gate dates.
STAGE3_GAIN = Fraction(1, 100)

#: Real-row support floors (plan section 3). Engineering floors, not sufficiency.
FIT_SUPPORT = {"rows": 500, "areas": 50, "dates": 6, "classes": 2}
STAGE1_VAL_SUPPORT = {"rows": 100, "areas": 20, "dates": 3}
STAGE3_GATE_SUPPORT = {"rows": 100, "areas": 20, "dates": 3, "local_fit_dates": 3}
GATE_DATES = 6

#: Stage 1 candidate targets: the labelled months of 2018-2020 (Feb/Jun/Oct).
STAGE1_TARGETS = tuple(f"{y}-{m:02d}" for y in (2018, 2019, 2020) for m in (2, 6, 10))
#: Development outer targets (G screening and the 24 full-pipeline schemes).
DEV_TARGETS = tuple(f"{y}-{m:02d}" for y in (2019, 2020) for m in (2, 6, 10))
PARTITION_INFO_CUTOFF = "2020-12"
FINAL_TARGETS = {4: ("2021-05", "2024-12"), 8: ("2021-09", "2024-12"), 12: ("2022-01", "2024-12")}

#: The 24 development schemes: one L per horizon, and one threshold strategy.
STRATEGIES = ("strict-only", "loose-only", "merged")
STRATEGY_FAMILIES = {"strict-only": ("gt001",), "loose-only": ("gt0",), "merged": ("gt0", "gt001")}


def l_vectors():
    """The 2**3 per-horizon local choices in a fixed numbering (index = vector id)."""
    out = []
    for a in ("L1", "L2"):
        for b in ("L1", "L2"):
            for c in ("L1", "L2"):
                out.append({4: a, 8: b, 12: c})
    return out


def schemes():
    """24 = 8 L vectors x 3 strategies, each with a stable id."""
    rows = []
    for v, vector in enumerate(l_vectors()):
        for strategy in STRATEGIES:
            rows.append({"scheme": f"v{v}_{strategy}", "l_vector_id": v, "l_vector": vector,
                         "strategy": strategy})
    return rows


def candidate_name(h, target, g, l, ratio, seed, family):
    return f"h{h}_{target}_{g}_{l}_{ratio}_s{seed}_{family}"


def root_name(h, target, g, ratio, seed):
    return f"h{h}_{target}_{g}_{ratio}_s{seed}"


def booster_params(config: dict) -> tuple[dict, int]:
    """(xgb.train params, rounds) for one G or L configuration."""
    params = dict(XGB_BASE)
    params.update({k: v for k, v in config.items() if k != "rounds"})
    return params, int(config["rounds"])


def g_tiebreak_key(name: str):
    """Exact-tie order for G selection: fewer rounds, shallower trees, config number."""
    c = G_CONFIGS[name]
    return (c["rounds"], c["max_depth"], int(name[1:]))
