"""Readable vocabulary for the IPCCH MLflow store (task 10-09-mlflow-readable-naming).

One place for every user-facing name: families, arms, lead times, periods, cohorts, metric
keys, run/model/dataset names and description text. Extraction (extract.py) keeps the source
reports' own names; ``metric_name`` translates them, and the old name stays in provenance.
Facts in the tables below were checked against the source code, configs and accepted reports
(see the task's research notes); the importer re-checks the period spans against predictions.
"""

from __future__ import annotations

import re

from extract import SourceConflict

# ------------------------------------------------------------------ families

STANDARD_PRIMARY = {"1": "2023-02..2025-10", "3": "2023-04..2025-10", "6": "2023-07..2025-10", "12": "2024-01..2025-10"}
HOLDOUT = {h: "2026-01..2026-04" for h in ("1", "3", "6", "12")}
SPLIT_PRIMARY = {"1": "2025-02..2025-10", "3": "2025-04..2025-10", "6": "2025-07..2025-10", "12": None}
SPLIT_COMBINED = {"1": "2025-02..2026-04", "3": "2025-04..2026-04", "6": "2025-07..2026-04", "12": "2026-01..2026-04"}
STANDARD_PERIODS = {"main": ("primary", STANDARD_PRIMARY), "supplementary": ("holdout", HOLDOUT)}

TRUTH = ("IPC phase decoded from the area's population shares: phase = highest p with cumulative share "
         "q_p >= 0.20 after a bounded isotonic projection; crisis = phase 3 or worse.")
MODEL_RECIPE = ("Four XGBoost regressors predict the cumulative population shares q2..q5 of each area; "
                "the predictions are projected to be monotone and decoded to an IPC phase with the 0.20 rule.")
REFERENCE_MAPS = "frozen reference maps (learned in GeoXGB reference on target months 2014-01..2022-12)"
MONTHLY_36 = "monthly refit; 36-month training window ending at the forecast origin; unit weights"

FAMILIES = {
    "p6_geoxgb": {
        "slug": "geoxgb_reference", "short": "GeoXGB reference",
        "long": "GeoXGB reference (monthly refit, maps learned on 2014-2022)",
        "model_type": "GeoXGB", "status": "supervisor_accepted", "features": "rich561",
        "maps": "learned in this run on target months 2014-01..2022-12",
        "training_window": MONTHLY_36, "periods": STANDARD_PERIODS, "seeds": ("42",),
        "what": (f"{MODEL_RECIPE} A spatial partition map, learned in this run on target months 2014-01..2022-12, "
                 "splits the areas into regions. Each region may get its own regional model; a historical gate "
                 f"decides per region whether to use it or the pooled model. {MONTHLY_36.capitalize()}."),
        "question": ("Does a statistically learned spatial partition improve crisis prediction over one pooled model "
                     "and over persistence? Does any gain come from the regressors or from the spatial layer?"),
        "conclusion": ('"The spatial layer shows no demonstrated benefit over pooled." "Against persistence, recall/F2 '
                       'and continuous q3 improve, while precision, binary accuracy and four-class macro F1 fall. Do '
                       'not describe general superiority." (supervisor acceptance, task 10-04)'),
        "status_text": ("Supervisor accepted (task 10-04-ipcch-cumulative-share-geoxgb). Close and spot audits were "
                        "waived by the user - this is not an audit pass."),
    },
    "yearly_geoxgb": {
        "slug": "geoxgb_yearly_refit", "short": "GeoXGB yearly refit",
        "long": "GeoXGB yearly refit (full history, 24-month decay, reference maps)",
        "model_type": "GeoXGB", "status": "supervisor_accepted", "features": "rich561", "maps": REFERENCE_MAPS,
        "training_window": ("yearly refit; all history up to the forecast origin; sample weights halve every 24 "
                            "months; gate decided once per region and year"),
        "periods": STANDARD_PERIODS, "seeds": ("42",),
        "what": (f"{MODEL_RECIPE} Uses the {REFERENCE_MAPS}. Models are refit once per year on all history up "
                 "to the forecast origin, with sample weights that halve every 24 months; the historical gate is "
                 "decided once per region and year."),
        "question": ("Do the frozen spatial partitions give useful regional XGB adaptation when fitting follows a "
                     "yearly pooled protocol?"),
        "conclusion": ('"No clear geographic gain under the annual protocol. G-P point estimates are near zero ... in '
                       'every main horizon, and all intervals include zero. This is not evidence of an exact zero or '
                       'of equivalence." The yearly pooled model "is .026 lower at H1, within +/-.005 at H3-H12. This '
                       'is a bundled change ... and cannot be attributed to one ingredient." (task 10-07 results)'),
        "status_text": ("Supervisor PASS at 76fdedf. The lifecycle close audit was queued, not passed. Exploratory: "
                        "2023-2026 had already been inspected."),
    },
    "climate_perturbation": {
        "slug": "geoxgb_climate_swap", "short": "GeoXGB climate swap",
        "long": "GeoXGB climate swap (rich601 climate inputs, maps relearned on 2014-2022)",
        "model_type": "GeoXGB", "status": "exploratory",
        "features": "rich601 (shared-folder monthly and growing-season climate replace the monthly climate columns)",
        "maps": "relearned in this run on rich601, target months 2014-01..2022-12",
        "training_window": MONTHLY_36, "periods": STANDARD_PERIODS, "seeds": ("42",),
        "what": (f"{MODEL_RECIPE} Same design as GeoXGB reference, but the monthly climate inputs are replaced by "
                 "shared-folder monthly and growing-season climate features (rich601; climate covariates start "
                 "2015-01), and the partition maps are relearned on these features (2014-01..2022-12)."),
        "question": "Does more complete climate data improve prediction?",
        "conclusion": ('"Replacing the monthly climate inputs and adding growing-season features does not change '
                       'main-period crisis F1 beyond country-sampling noise. Pooled delta is -0.0007 to +0.0063 and '
                       'every interval includes zero." "The spatial layer still adds nothing: Geo-pooled stays within '
                       '+/-0.0006 at every H." (task 10-05 results)'),
        "status_text": "Exploratory; the 2023-2026 period had already been inspected. No audit (task 10-05).",
    },
    "split2024_sensitivity": {
        "slug": "geoxgb_maps_2024", "short": "GeoXGB maps to 2024",
        "long": "GeoXGB maps to 2024 (maps relearned on 2014-2024, evaluated 2025-2026)",
        "model_type": "GeoXGB", "status": "exploratory", "features": "rich561",
        "maps": "relearned in this run on target months 2014-01..2024-12",
        "training_window": MONTHLY_36, "seeds": ("42",),
        "periods": {"main": ("primary", SPLIT_PRIMARY), "supplementary": ("holdout", HOLDOUT),
                    "combined": ("combined", SPLIT_COMBINED), "y2025": ("year_2025", SPLIT_PRIMARY),
                    "y2026": ("year_2026", HOLDOUT)},
        "what": (f"{MODEL_RECIPE} Same design as GeoXGB reference, but the partition maps are relearned on target "
                 "months 2014-01..2024-12 and evaluation starts in 2025. The primary period is 2025 only (empty at "
                 "the 12-month lead); year_2025 / year_2026 / combined come from a separately recomputed report "
                 "and agree with primary / holdout to floating-point precision."),
        "question": "Can more recent partition-learning data and wider map coverage recover the spatial gain?",
        "conclusion": ("More development data and wider map coverage still did not recover a partitioned-over-pooled "
                       "F1 gain; at the 12-month lead the pooled model also rose from 0.7747 to 0.7870, so the overall "
                       "gain cannot be credited to the geographic layer. (translated from the user's 2026-10-04 "
                       "group-meeting note; no supervisor sign-off)"),
        "status_text": ("Exploratory post-hoc robustness check, user-approved, no Trellis task; 2025-2026 had "
                        "already been seen. Replay passed (29,848 checks)."),
    },
    "history_window_sensitivity": {
        "slug": "geoxgb_window_probe", "short": "GeoXGB window probe",
        "long": "GeoXGB window probe (36-month vs full-history training window, selected months)",
        "model_type": "GeoXGB", "status": "exploratory", "features": "rich561", "maps": REFERENCE_MAPS,
        "training_window": "36-month window vs all history from 2014-01, both ending at the forecast origin",
        "periods": {}, "seeds": ("42",),
        "what": (f"{MODEL_RECIPE} Uses the {REFERENCE_MAPS} and recipes. For a few selected target months, the "
                 "pooled and regional models are fit once on a 36-month window and once on all history from "
                 "2014-01. The historical gate is not recomputed or applied: the regional arms are ungated and fall "
                 "back to the pooled arm of the same window where a region lacks local support or an area is "
                 "unmapped."),
        "question": "Can the limited regional gain be improved by adding older historical training data?",
        "conclusion": ("With the recipes fixed, full history did not generally improve pooled F1 and did not recover "
                       "the geographic increment; adding older history alone is not supported as a fix. The gate was "
                       "not recomputed or applied, so this is not a full deployed-GeoXGB result. (translated from "
                       "the user's 2026-10-04 group-meeting note; no supervisor sign-off)"),
        "status_text": ("Exploratory read-only diagnostic on selected months; no significance test; no Trellis "
                        "task."),
    },
    "mlp_fixed_map": {
        "slug": "mlp_residual_fixed_maps", "short": "MLP residual on fixed maps",
        "long": "MLP residual on fixed maps (global MLP + residual correction on the reference maps, 3 seeds)",
        "model_type": "MLP", "status": "user_accepted",
        "features": "rich561 plus 561 missingness flags (1,122 inputs)", "maps": REFERENCE_MAPS,
        "training_window": MONTHLY_36, "periods": STANDARD_PERIODS, "seeds": ("42", "43", "44"),
        "what": ("A global MLP (global_base) predicts the cumulative population shares, decoded with the 0.20 "
                 "rule. pooled = global_base + one pooled residual model; regional_ungated = global_base + a "
                 f"regional residual model on the {REFERENCE_MAPS}; partitioned_gated uses the regional residual "
                 f"where the historical gate passes, else pooled. {MONTHLY_36.capitalize()}. Three seeds."),
        "question": ("Does an MLP improve IPCCH population-share prediction, and does regional residual correction "
                     "on the existing maps add value beyond a matched pooled residual?"),
        "conclusion": ('"The MLP learner is weaker than XGB here. Pooled MLP (B) is .021-.057 crisis F1 below the '
                       'matched pooled XGB." "Regional residual correction adds nothing measurable on the fixed XGB '
                       'maps." "The MLP system is below persistence." (task 10-05 MLP results)'),
        "status_text": ("Completed under user supervision (the user acted as supervisor); the Trellis audit was "
                        "waived by the user - not an audit pass. Exploratory: 2023-2026 had already been inspected. "
                        "Replay passed."),
    },
}

# Training pool: the prepared feature matrix + key table every refit draws from (one per lead).
# "from" = the family whose archived manifests hold the file hashes ("note" says why it differs).
_P6_POOL = {"from": "p6_geoxgb", "x": "prepared/X_rich561_h{H:02d}.npy", "keys": "prepared/keys_h{H:02d}.csv.gz",
            "features": "rich561"}
TRAINING_POOL = {
    "p6_geoxgb": _P6_POOL,
    "climate_perturbation": {"from": "climate_perturbation", "x": "prepared/X_rich601_h{H:02d}.npy",
                             "keys": "prepared/keys_h{H:02d}.csv.gz", "features": "rich601"},
    "split2024_sensitivity": {"from": "split2024_sensitivity", "features": "rich561",
                              "x": "IPCCHGeoXGBExperiment/runs/split2024-20261005/prepared/X_rich561_h{H:02d}.npy",
                              "keys": "IPCCHGeoXGBExperiment/runs/split2024-20261005/prepared/keys_h{H:02d}.csv.gz"},
    "yearly_geoxgb": {"from": "yearly_geoxgb", "x": "inputs/run/prepared/X_rich561_h{H:02d}.npy",
                      "keys": "inputs/run/prepared/keys_h{H:02d}.csv.gz", "features": "rich561"},
    "mlp_fixed_map": {**_P6_POOL, "note": "reads the GeoXGB reference prepared files in place (pinned by SHA256 "
                                          "in IPCCHMLPExperiment/config/inputs.json)"},
    "history_window_sensitivity": {**_P6_POOL, "note": "reads the GeoXGB reference prepared files in place "
                                                       "(pinned by SHA256 in the probe's start.json)"},
}
TRAINING_POOL_TEXT = ("Candidate training rows: every QC-valid area x target month from 2014-01, with features "
                      "observed at or before its own forecast origin. Each refit uses the subset allowed by the "
                      "family's training window; this descriptor names the pool, not the exact rows of one refit.")

STATUS_MEANING = {"supervisor_accepted": "results accepted by the supervisor",
                  "user_accepted": "results accepted by the user acting as supervisor",
                  "exploratory": "exploratory; no formal acceptance"}

# ------------------------------------------------------------------ arms

ROLE = {"persistence_baseline": "baseline", "reused_comparator": "baseline", "fresh_trained": "candidate",
        "diagnostic_local": "diagnostic"}
ROLE_MEANING = {"baseline": "not trained in this run (persistence, or a model reused for comparison)",
                "candidate": "trained in this run; part of the main comparison",
                "diagnostic": "trained in this run for diagnosis only (not a deployable arm)"}

_XGB = {"pool": ("pooled", None, "fresh_trained"), "geo": ("partitioned_gated", None, "fresh_trained"),
        "persistence": ("persistence", None, "persistence_baseline")}
ARMS = {
    "p6_geoxgb": _XGB, "climate_perturbation": _XGB, "split2024_sensitivity": _XGB,
    "yearly_geoxgb": {**_XGB, "local": ("regional_ungated", None, "diagnostic_local")},
    "mlp_fixed_map": {"base": ("global_base", None, "fresh_trained"), **_XGB,
                      "local": ("regional_ungated", None, "diagnostic_local")},
    "history_window_sensitivity": {"base_global": ("pooled", "36_month", "reused_comparator"),
                                   "exp_global": ("pooled", "full_history", "fresh_trained"),
                                   "base_local": ("regional_ungated", "36_month", "reused_comparator"),
                                   "exp_local": ("regional_ungated", "full_history", "fresh_trained")},
}
WINDOW_LABEL = {"36_month": "36-month window", "full_history": "full-history window"}

ARM_MEANING = {
    "persistence": ("No model: predicts the phase observed most recently at or before the forecast origin in the "
                    "same area. Defined only on persistence_available rows."),
    "pooled": "One model for all areas.",
    "partitioned_gated": ("Regional model where the historical gate accepted it for that region, pooled model "
                          "elsewhere."),
    "regional_ungated": ("The regional model's own prediction wherever a regional model was fitted, whatever the "
                         "gate decided. Diagnostic: scored on regional_model_fitted rows only."),
    "global_base": "The global MLP alone, without a residual model.",
}
MLP_ARM_MEANING = {
    "pooled": "global_base plus one pooled residual model for all areas.",
    "partitioned_gated": ("global_base plus the regional residual where the historical gate passed for that region, "
                          "pooled elsewhere."),
    "regional_ungated": ("global_base plus the regional residual, wherever a regional residual was fitted, whatever "
                         "the gate decided. Diagnostic: scored on regional_model_fitted rows only."),
}
WINDOW_ARM_MEANING = {
    "pooled": "One model for all areas.",
    "regional_ungated": ("Regional model fitted without a gate where the region has local support; elsewhere the "
                         "pooled arm of the same window."),
}

# tokens used inside deltas / contrasts / comparator panels
OPERAND = {"geo": "partitioned_gated", "pool": "pooled", "local": "regional_ungated", "base": "global_base",
           "persistence": "persistence", "p6geo": "reference_partitioned_gated", "xgbgeo": "reference_partitioned_gated",
           "p6pool": "reference_pooled", "xgbpool": "reference_pooled"}

# ------------------------------------------------------------------ cohorts

COHORTS = {"E_all": "all_scored", "all": "all_scored", "E_persist": "persistence_available",
           "local_eligible": "regional_model_fitted",
           "local_persist_matched": "regional_model_fitted_and_persistence_available",
           "mapped": "in_partition_map", "common_local_support": "regional_model_fitted_both_windows",
           "new_local_support": "regional_model_fitted_full_history_only"}
GATES = {"adopted": "gate_used_regional", "gain_rejected": "gate_rejected_no_gain",
         "historical_support_rejected": "gate_rejected_too_little_validation"}
# cohorts whose rows depend on the family's own regional fits: dataset names carry the family
FAMILY_DEPENDENT = {"regional_model_fitted", "regional_model_fitted_and_persistence_available",
                    "regional_model_fitted_both_windows", "regional_model_fitted_full_history_only"}
COHORT_MEANING = {
    "all_scored": "Every scored row: one area x one target month in the period with a valid IPCCH truth value.",
    "persistence_available": ("Rows whose area has at least one valid IPCCH observation at or before the forecast "
                              "origin (target month minus lead). Only these rows have a persistence forecast, so "
                              "every comparison with persistence uses this cohort."),
    "regional_model_fitted": ("Rows whose region got a regional model at that refit (training data with at least "
                              "500 rows, 50 areas and 6 target months), whatever the gate later decided."),
    "regional_model_fitted_and_persistence_available": ("regional_model_fitted rows that also have a persistence "
                                                        "forecast; used to compare the regional model with "
                                                        "persistence on exactly the same rows."),
    "in_partition_map": "Rows whose area is covered by the frozen partition map.",
    "regional_model_fitted_both_windows": ("Rows in regions that can fit a regional model under both the 36-month "
                                           "window and full history; the only rows where the two regional arms "
                                           "are both real regional fits."),
    "regional_model_fitted_full_history_only": ("Rows in regions with too little data for a regional model in the "
                                                "36-month window but enough with full history (from 2014-01); "
                                                "shows what a longer window adds."),
    "gate_used_regional": "regional_model_fitted rows where the gate used the regional model.",
    "gate_rejected_no_gain": ("regional_model_fitted rows where the gate rejected the regional model because it "
                              "did not beat the pooled model on past validation months."),
    "gate_rejected_too_little_validation": ("regional_model_fitted rows where the gate rejected the regional model "
                                            "because past validation data were too thin (needs 100 rows, 20 areas, "
                                            "3 months, 20 crisis and 20 non-crisis rows)."),
}
PERIOD_ROLE_MEANING = {
    "primary": "main evaluation period of the family",
    "holdout": "later months reported separately (2026-01..2026-04, 4 target months, point estimates)",
    "combined": "primary and holdout together (GeoXGB maps to 2024 only)",
    "year_2025": "2025 block of the separately recomputed report (GeoXGB maps to 2024 only)",
    "year_2026": "2026 block of the separately recomputed report (GeoXGB maps to 2024 only)",
    "selected_months": "selected target months only (GeoXGB window probe), not a full period",
}

# ------------------------------------------------------------------ metric names

LEAF = {"n": "n_rows", "n_keys": "n_rows_reported", "q3_r2_projected": "share_phase3plus_r2",
        "q3_r2_raw": "share_phase3plus_r2_raw", "q3_mse.star": "share_phase3plus_mse",
        "q3_mse.raw": "share_phase3plus_mse_raw"}
_PANEL = re.compile(r"^(binary\.(accuracy|precision|recall|f1|f2|count\.(tp|fp|fn|tn))|four_class\.(accuracy|macro_f1))$")
BOOT = {"point_delta": "delta", "ci_lower": "ci_low", "ci_upper": "ci_high", "K": "countries", "draws": "draws",
        "defined_draws": "defined_draws", "undefined_draws": "undefined_draws", "seed": "rng_seed",
        "ci": "ci"}   # NA entry: no interval saved
_SELECTED_DATE = re.compile(r"^selected_date_(\d{4}-\d{2})$")


def _family(family: str) -> dict:
    if family not in FAMILIES:
        raise SourceConflict(f"unknown family {family}")
    return FAMILIES[family]


def _leaf(parts: list, old: str) -> str:
    s = ".".join(parts)
    if s in LEAF:
        return LEAF[s]
    if _PANEL.match(s):
        return s
    raise SourceConflict(f"no naming rule for metric {old!r} (leaf {s!r})")


def _operand(tok: str, old: str) -> str:
    if tok not in OPERAND:
        raise SourceConflict(f"no naming rule for arm token {tok!r} in {old!r}")
    return OPERAND[tok]


def _pair(tok: str, old: str) -> str:
    m = re.match(r"^new_minus_old_(\w+)$", tok)
    if m:
        a = _operand(m.group(1), old)
        return f"{a}_minus_reference_{a}"
    for sep in ("_minus_", "_vs_"):
        if sep in tok:
            a, b = tok.split(sep, 1)
            return f"{_operand(a, old)}_minus_{_operand(b, old)}"
    raise SourceConflict(f"no naming rule for contrast {tok!r} in {old!r}")


def _boot(parts: list, old: str) -> str:
    if len(parts) != 2 or parts[1] not in BOOT:
        raise SourceConflict(f"no naming rule for bootstrap metric {old!r}")
    return f"bootstrap.{_pair(parts[0], old)}.binary.f1.{BOOT[parts[1]]}"


def _cohort_tail(parts: list, old: str) -> str:
    head = parts[0] if parts else ""
    if head in GATES:
        return f"{GATES[head]}.{_cohort_tail(parts[1:], old)}"
    if head in ("comparator", "matched_old") and len(parts) > 2 and parts[1] in OPERAND:
        return f"{_operand(parts[1], old)}.{_leaf(parts[2:], old)}"
    if head == "matched_old" and len(parts) > 1:
        return _cohort_tail(parts[1:], old)
    if head in ("delta", "comparator_delta") and len(parts) > 2:
        return f"delta.{_pair(parts[1], old)}.{_leaf(parts[2:], old)}"
    if head == "contrast":
        return _boot(parts[1:], old)
    return _leaf(parts, old)


def _tree(parts: list) -> str:
    out = []
    for p in parts:
        p = p.replace("E_all", "all_scored").replace("E_persist", "persistence_available")
        p = p.replace("local_eligible", "regional_model_fitted")
        out.append("_".join("rows" if w == "keys" else w for w in p.split("_")))
    return ".".join(out)


def period_role(family: str, old_period: str) -> str:
    fam = _family(family)
    if old_period not in fam["periods"]:
        raise SourceConflict(f"no period {old_period!r} in family {family}")
    return fam["periods"][old_period][0]


def period_span(family: str, old_period: str, H: str) -> str | None:
    fam = _family(family)
    if old_period not in fam["periods"]:
        raise SourceConflict(f"no period {old_period!r} in family {family}")
    return fam["periods"][old_period][1][H]


def metric_name(family: str, old: str) -> str:
    """Translate one metric key of the accepted import into the readable vocabulary."""
    _family(family)
    parts = old.split(".")
    if parts[0] == "seed_summary" and len(parts) == 4:
        m = re.match(r"^(\w+?)_h(\d{2})$", parts[1])
        if not m:
            raise SourceConflict(f"no naming rule for {old!r}")
        stat = parts[2]
        if stat.startswith("f1_"):
            what = f"{_operand(stat[3:], old)}.binary.f1"
        elif stat.endswith("_f1"):
            what = f"{_pair(stat[:-3], old)}.binary.f1"
        else:
            raise SourceConflict(f"no naming rule for {old!r}")
        return f"seed_summary.{period_role(family, m.group(1))}.lead_{m.group(2)}.{what}.{parts[3]}"
    if parts[0] == "selected_dates":
        if len(parts) < 3 or parts[1] not in COHORTS:
            raise SourceConflict(f"no naming rule for {old!r}")
        return f"selected_months.{COHORTS[parts[1]]}.{_cohort_tail(parts[2:], old)}"
    d = _SELECTED_DATE.match(parts[0])
    if d:
        if parts[1:] == ["fit_n"]:
            return f"target_month_{d.group(1)}.training_rows"
        if len(parts) < 3 or parts[1] not in COHORTS:
            raise SourceConflict(f"no naming rule for {old!r}")
        return f"target_month_{d.group(1)}.{COHORTS[parts[1]]}.{_cohort_tail(parts[2:], old)}"
    role = period_role(family, parts[0])
    rest = parts[1:]
    if not rest:
        raise SourceConflict(f"no naming rule for {old!r}")
    head = rest[0]
    if head in COHORTS:
        return f"{role}.{COHORTS[head]}.{_cohort_tail(rest[1:], old)}"
    if head == "coverage":
        return f"{role}.coverage.{_tree(rest[1:])}"
    if head == "routes":                       # gate-route names are the source report's own labels
        return f"{role}.routes.{'.'.join(rest[1:])}"
    if head == "label_flips_geo_vs_pool" and len(rest) == 2:
        return f"{role}.label_flips.partitioned_gated_vs_pooled.{rest[1]}"
    if head == "label_flips_vs_p6" and len(rest) == 3:
        return f"{role}.label_flips.{_operand(rest[1], old)}_vs_reference.{rest[2]}"
    if rest == ["local_rows"]:
        return f"{role}.regional_routed_rows"
    if rest == ["unmapped_rows"]:
        return f"{role}.unmapped_rows"
    raise SourceConflict(f"no naming rule for metric {old!r}")


def check_one_to_one(family: str, olds) -> dict:
    """Old->new map for one record; refuses two old names that land on one new name."""
    out, seen = {}, {}
    for o in olds:
        n = metric_name(family, o)
        if n in seen and seen[n] != o:
            raise SourceConflict(f"{family}: {seen[n]!r} and {o!r} map to the same new name {n!r}")
        seen[n] = o
        out[o] = n
    return out


def namespace(family: str, old_ns: str) -> str:
    """Old cohort namespace (e.g. ``main.E_persist``) -> new (``primary.persistence_available``)."""
    return metric_name(family, f"{old_ns}.n").rsplit(".", 1)[0]


# ------------------------------------------------------------------ names

def arm(family: str, old_arm: str) -> tuple:
    """(new arm, window or None, arm_role)."""
    arms = ARMS.get(family) or {}
    if old_arm not in arms:
        raise SourceConflict(f"no arm {old_arm!r} in family {family}")
    new, window, kind = arms[old_arm]
    return new, window, ROLE[kind]


def lead_label(H: str) -> str:
    return f"{int(H)}-month"


def lead_tag(H: str) -> str:
    return f"{int(H):02d}"


def _arm_parts(family: str, old_arm: str) -> list:
    new, window, _ = arm(family, old_arm)
    return [new] + ([WINDOW_LABEL[window]] if window else [])


def multi_seed(family: str) -> bool:
    return len(_family(family)["seeds"]) > 1


def run_name(family: str, old_arm: str, H: str, seed: str) -> str:
    parts = [_family(family)["short"], *_arm_parts(family, old_arm), lead_label(H)]
    if multi_seed(family) and seed not in ("none", "mean"):
        parts.append(f"seed {seed}")
    if seed == "mean":
        parts.append("mean of 3 seeds")
    return " | ".join(parts)


def registered_model_name(family: str, old_arm: str, H: str) -> str:
    return " | ".join([f"IPCCH {_family(family)['short']}", *_arm_parts(family, old_arm), lead_label(H)])


def logged_model_name(family: str, old_arm: str, H: str, seed: str) -> str:
    return " | ".join([_family(family)["short"], *_arm_parts(family, old_arm), lead_label(H), f"seed {seed}"])


def record_key(family: str, old_arm: str, H: str, seed: str) -> str:
    new, window, _ = arm(family, old_arm)
    return "/".join([_family(family)["slug"], f"lead{lead_tag(H)}", new] + ([window] if window else []) + [f"seed{seed}"])


def span_label(family: str, H: str, old_period: str, dates) -> str:
    if old_period == "selected_dates":
        return "target months " + ", ".join(dates)
    span = period_span(family, old_period, H)
    if span is None:
        raise SourceConflict(f"{family}: no scored months in period {old_period} at lead {H}")
    return span


def eval_dataset_name(family: str, H: str, old_period: str, old_cohort: str, dates) -> str:
    cohort = COHORTS[old_cohort]
    name = f"IPCCH eval | {lead_label(H)} | {span_label(family, H, old_period, dates)} | {cohort}"
    if cohort in FAMILY_DEPENDENT:
        name += f" ({_family(family)['short']})"
    return name


def training_dataset_name(features: str, H: str) -> str:
    return f"IPCCH training pool | {features} | {lead_label(H)}"


# ------------------------------------------------------------------ description text

def periods_text(family: str) -> str:
    """Evaluation periods of a family, by role and lead (tag ``periods`` on the family run)."""
    f = _family(family)
    if not f["periods"]:
        return "selected target months only: 2023-09, 2024-09, 2025-09 (12-month: 2024-09, 2025-01)"
    out = []
    for role, spans in f["periods"].values():
        by_span = {}
        for h in ("1", "3", "6", "12"):
            by_span.setdefault(spans[h] or "none scored", []).append(lead_label(h))
        if len(by_span) == 1:
            out.append(f"{role} {next(iter(by_span))}")
        else:
            out.append(f"{role} " + ", ".join(f"{sp} ({'/'.join(ls)})" for sp, ls in by_span.items()))
    return "; ".join(out)


def arm_meaning(family: str, old_arm: str) -> str:
    new = arm(family, old_arm)[0]
    if family == "mlp_fixed_map" and new in MLP_ARM_MEANING:
        return MLP_ARM_MEANING[new]
    if family == "history_window_sensitivity":
        return WINDOW_ARM_MEANING[new]
    return ARM_MEANING[new]


def family_description(family: str, source_run_id: str) -> str:
    f = _family(family)
    return "\n\n".join([
        f"**{f['long']}**",
        f"**What:** {f['what']}",
        f"**Question:** {f['question']}",
        f"**Accepted conclusion:** {f['conclusion']}",
        f"**Status:** {f['status_text']}",
        f"**Truth:** {TRUTH}",
        ("**Original:** source run `" + source_run_id + "`; source files, manifests and the models.tar bundle are "
         "archived on this run. Child runs hold one arm x lead time each. Values are copied from the saved reports; "
         "MLflow times are import times, not fit times."),
    ])


def compare_with(family: str, old_arm: str) -> str:
    new = arm(family, old_arm)[0]
    if family == "history_window_sensitivity":
        return ("the same arm under the other training window, on the same selected months and cohort; "
                "regional arms only on regional_model_fitted_both_windows for a like-for-like regional comparison.")
    if new == "persistence":
        return "any model arm on persistence_available rows."
    out = "pooled on all_scored and persistence on persistence_available, same lead time and period only."
    if new == "pooled":
        out = "persistence on persistence_available rows, same lead time and period only."
    if new == "regional_ungated":
        out = "pooled on regional_model_fitted rows (and persistence on regional_model_fitted_and_persistence_available)."
    if family in ("yearly_geoxgb", "mlp_fixed_map", "climate_perturbation", "split2024_sensitivity") and new != "persistence":
        out += " reference_* panels are the matching GeoXGB reference arm on the same rows."
    return out


def view_description(family: str, old_arm: str, H: str, seed: str, source_run_id: str) -> str:
    f = _family(family)
    new, window, role = arm(family, old_arm)
    seed_txt = "" if seed == "none" else f" Seed {seed}."
    return "\n\n".join([
        f"**{run_name(family, old_arm, H, seed)}**",
        (f"**What:** {arm_meaning(family, old_arm)} {lead_label(H)} lead (forecast origin = target month minus "
         f"{int(H)}). Family: {f['long']}. Role: {role} ({ROLE_MEANING[role]}).{seed_txt}"
         + (f" Training window: {WINDOW_LABEL[window]}." if window else "")),
        f"**Compare with:** {compare_with(family, old_arm)} Compare values only within one cohort and period.",
        f"**Status:** {f['status_text']}",
        ("**Caveats + original:** values copied from the saved report of source run `" + source_run_id + "`; "
         "undefined values are not logged (see view/na.json). Metric keys read "
         "<period_role>.<cohort>.<metric>; view/evaluation_view.json maps every key to its original name and "
         "source path."),
    ])
