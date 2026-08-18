from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree


matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


JSONDict = dict[str, Any]
FIGURE_LAYOUT = {
    "width_inches": 28,
    "individual_width_inches": 14,
    "height_inches": 11,
    "tree_fontsize": 12,
    "title_fontsize": 18,
    "suptitle_fontsize": 14,
    "condition_wrap_width_chars": 18,
    "dpi": 300,
    "uses_constrained_layout": True,
}

PAPER_FEATURE_LABELS = {
    "EVI": "EVI",
    "EVI_lag4m": "EVI (4 mo prior)",
    "Rainf_zscore": "Rainfall anomaly (z-score)",
    "event_count_explosions_w5": "Explosion events\n(nearest-5 mean)",
    "event_count_explosions_w10": "Explosion events\n(nearest-10 mean)",
    "event_count_violence_w5": "Violent events\n(nearest-5 mean)",
    "pop": "Population",
    "lat": "Latitude",
    "market_distance_lag4m": "Market distance\n(4 mo prior)",
    "distance_to_nearest_acled": "Distance to nearest ACLED event",
    "distance_to_nearest_acled_lag4m": "Nearest ACLED-event distance\n(4 mo prior)",
    "sg_phh2o_5-15cm": "Soil pH (H2O), 5-15 cm",
}

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def _paper_feature_label(raw_name: str) -> str:
    """Return a paper-friendly feature label without changing model values."""
    if raw_name in PAPER_FEATURE_LABELS:
        return PAPER_FEATURE_LABELS[raw_name]

    ipc_match = re.fullmatch(r"fews_ipc_lag_(\d+)(?:_lag(\d+)m)?", raw_name)
    if ipc_match:
        lag_months = int(ipc_match.group(1)) + int(ipc_match.group(2) or 0)
        return f"IPC phase ({lag_months} mo prior)"

    crisis_match = re.fullmatch(
        r"fews_ipc_crisis_lag_(\d+)(?:_lag(\d+)m)?", raw_name
    )
    if crisis_match:
        lag_months = int(crisis_match.group(1)) + int(
            crisis_match.group(2) or 0
        )
        return f"Crisis status ({lag_months} mo prior)"

    evi_match = re.fullmatch(r"EVI_l(\d+)(?:_lag(\d+)m)?", raw_name)
    if evi_match:
        lag_months = int(evi_match.group(1)) + int(evi_match.group(2) or 0)
        return f"EVI ({lag_months} mo prior)"

    month_match = re.fullmatch(r"month_(\d+)", raw_name)
    if month_match:
        month_number = int(month_match.group(1))
        if month_number in MONTH_NAMES:
            return f"{MONTH_NAMES[month_number]} indicator"

    aez_match = re.fullmatch(r"AEZ_(.+)", raw_name)
    if aez_match:
        return f"AEZ {aez_match.group(1)} indicator"

    return re.sub(r"\s+", " ", raw_name.replace("_", " ")).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare branch-specific GeoDT local DecisionTree checkpoints."
    )
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--archive-list")
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--artifact-provider-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--png-only", action="store_true")
    parser.add_argument("--pdf-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--reproduce-from", type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-plot-depth", type=int, default=None)
    return parser


def _archive_identity(path: Path) -> tuple[str, str]:
    match = re.search(r"fs(?P<fs>\d+)_(?P<month>\d{4}-\d{2})", path.name)
    if match:
        return match.group("month"), f"fs{match.group('fs')}"
    return "unknown-month", "unknown-fs"


def _archive_discovery_input(args: argparse.Namespace) -> JSONDict:
    if args.archive_path is not None:
        return {"type": "archive-path", "value": str(args.archive_path)}
    if args.archive_list:
        return {"type": "archive-list", "value": args.archive_list}
    if args.archive_root is not None:
        return {"type": "archive-root", "value": str(args.archive_root)}
    return {"type": "unspecified", "value": None}


def _candidate_paths(args: argparse.Namespace) -> tuple[list[Path], str | None]:
    if args.archive_path is not None:
        return [args.archive_path], None
    if args.archive_list:
        return [
            Path(raw.strip()) for raw in args.archive_list.split(",") if raw.strip()
        ], None
    if args.archive_root is not None:
        if not args.archive_root.is_dir():
            return (
                [],
                f"Archive root is not a bounded existing directory: {args.archive_root}",
            )
        return sorted(
            path
            for path in args.archive_root.iterdir()
            if path.is_dir() and path.name.startswith("result_GeoDT")
        ), None
    return (
        [],
        "No bounded archive input was provided. Use --archive-path, --archive-list, or --archive-root.",
    )


def _artifact_roots(path: Path, provider: Path | None = None) -> list[Path]:
    roots = [path]
    if provider is not None and provider.is_dir():
        roots.append(provider)
    return roots


def _checkpoint_record(checkpoint_path: str) -> JSONDict:
    path = Path(checkpoint_path)
    suffix = path.name.removeprefix("dt_")
    parsed_branch_id = suffix
    if path.name == "dt_":
        classification = "root/global"
        parsed_branch_id = ""
    elif re.fullmatch(r"[01]+", suffix):
        classification = "branch-specific candidate"
    else:
        classification = "unknown"

    record: JSONDict = {
        "path": str(path),
        "filename": path.name,
        "parsed_branch_id": parsed_branch_id,
        "classification": classification,
        "parsing_method": "DTmodel dt_<normalized-branch-id> filename convention",
        "load_status": "not attempted",
        "mismatch_reason": None,
    }
    try:
        with path.open("rb") as handle:
            pickle.load(handle)
        record["load_status"] = "loadable"
    except Exception as exc:
        record["classification"] = "unusable"
        record["load_status"] = "failed"
        record["mismatch_reason"] = str(exc)
    return record


def _assignment_counts(
    assignment_source: dict[str, str] | None,
) -> tuple[dict[str, int], str]:
    if assignment_source is None:
        return {}, "none"

    source_path = Path(assignment_source["path"])
    source_type = assignment_source["type"]
    if source_type == "X_branch_id.npy":
        values = [
            str(value) for value in np.load(source_path, allow_pickle=True).tolist()
        ]
        return dict(
            Counter(value for value in values if value != "")
        ), "prediction-row count"
    if source_type == "s_branch.pkl":
        frame = pd.read_pickle(source_path)
        counts: dict[str, int] = {}
        for column in frame.columns:
            counts[str(column)] = int(
                sum(int(value) >= 0 for value in frame[column].dropna())
            )
        return counts, "assigned admin/group count"
    if source_type == "correspondence_table":
        frame = pd.read_csv(source_path)
        if "branch_id" in frame.columns:
            return {
                str(key): int(value)
                for key, value in frame["branch_id"].astype(str).value_counts().items()
            }, "assigned admin/group count"
        if "partition_id" in frame.columns:
            return {
                str(key): int(value)
                for key, value in frame["partition_id"]
                .astype(str)
                .value_counts()
                .items()
            }, "assigned admin/group count"
    return {}, "unknown"


def _tree_node_depths(clf: Any) -> list[int]:
    tree = clf.tree_
    depths = [0] * int(tree.node_count)
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        if left >= 0:
            stack.append((left, depth + 1))
        if right >= 0:
            stack.append((right, depth + 1))
    return depths


def extract_branch_signature(
    clf: Any,
    *,
    feature_names: list[str],
    k: int = 3,
    branch_id: str | None = None,
) -> JSONDict:
    tree = clf.tree_
    depths = _tree_node_depths(clf)
    split_features_by_depth: dict[int, list[str]] = {}
    threshold_direction_summary: dict[str, list[str]] = {}
    leaf_class_summaries: list[JSONDict] = []

    for node_id, depth in enumerate(depths):
        feature_index = int(tree.feature[node_id])
        if feature_index >= 0:
            if depth < k:
                feature_name = feature_names[feature_index]
                split_features_by_depth.setdefault(depth, []).append(feature_name)
                threshold = float(tree.threshold[node_id])
                threshold_direction_summary.setdefault(feature_name, []).extend(
                    [f"<= {threshold:.6g}", f"> {threshold:.6g}"]
                )
            continue

        values = tree.value[node_id][0]
        class_index = int(np.argmax(values))
        samples = int(tree.n_node_samples[node_id])
        leaf_class_summaries.append(
            {
                "node_id": node_id,
                "depth": depth,
                "class_summary": f"class {class_index}",
                "samples": samples,
            }
        )

    split_feature_set = {
        feature
        for features_at_depth in split_features_by_depth.values()
        for feature in features_at_depth
    }
    split_depths = [
        depth for node_id, depth in enumerate(depths) if int(tree.feature[node_id]) >= 0
    ]
    return {
        "branch_id": branch_id,
        "k": k,
        "actual_available_depth": max(split_depths) if split_depths else 0,
        "split_features_by_depth": split_features_by_depth,
        "split_feature_set": split_feature_set,
        "threshold_direction_summary": threshold_direction_summary,
        "leaf_class_summaries": leaf_class_summaries,
    }


def _contrast_descriptor(jaccard_distance: float) -> str:
    if jaccard_distance >= 0.75:
        return "high"
    if jaccard_distance >= 0.40:
        return "moderate"
    return "low"


def _signature_count(signature: JSONDict, key: str) -> int | None:
    value = signature.get(key)
    if value is None:
        return None
    return int(value)


def score_branch_pairs(
    signatures: dict[str, JSONDict],
) -> list[JSONDict]:
    records: list[JSONDict] = []
    branch_ids = sorted(str(branch_id) for branch_id in signatures)
    for left_index, branch_a in enumerate(branch_ids):
        for branch_b in branch_ids[left_index + 1 :]:
            signature_a = signatures[branch_a]
            signature_b = signatures[branch_b]
            features_a = set(signature_a.get("split_feature_set", set()))
            features_b = set(signature_b.get("split_feature_set", set()))
            union = features_a | features_b
            if not union:
                continue
            intersection = features_a & features_b
            jaccard_distance = 1 - (len(intersection) / len(union))

            assigned_counts = [
                _signature_count(signature_a, "assigned_count"),
                _signature_count(signature_b, "assigned_count"),
            ]
            assigned_counts_present = [
                count for count in assigned_counts if count is not None
            ]
            prediction_counts = [
                _signature_count(signature_a, "prediction_row_count"),
                _signature_count(signature_b, "prediction_row_count"),
            ]
            prediction_counts_present = [
                count for count in prediction_counts if count is not None
            ]
            min_prediction_row_count = (
                min(prediction_counts_present)
                if len(prediction_counts_present) == 2
                else None
            )

            records.append(
                {
                    "branch_pair": (branch_a, branch_b),
                    "jaccard_distance": jaccard_distance,
                    "contrast_descriptor": _contrast_descriptor(jaccard_distance),
                    "ranking_formula": "top_k_split_feature_jaccard_distance",
                    "threshold_direction_fields_used_for_ranking": False,
                    "split_feature_intersection": sorted(intersection),
                    "split_feature_union": sorted(union),
                    "min_assigned_count": min(assigned_counts_present)
                    if len(assigned_counts_present) == 2
                    else 0,
                    "total_assigned_count": sum(assigned_counts_present),
                    "min_prediction_row_count": min_prediction_row_count,
                    "tie_break_fields": {
                        "min_assigned_count": min(assigned_counts_present)
                        if len(assigned_counts_present) == 2
                        else 0,
                        "total_assigned_count": sum(assigned_counts_present),
                        "min_prediction_row_count": min_prediction_row_count,
                        "lexicographic_branch_pair": (branch_a, branch_b),
                    },
                }
            )

    return sorted(
        records,
        key=lambda record: (
            -float(record["jaccard_distance"]),
            -int(record["min_assigned_count"]),
            -int(record["total_assigned_count"]),
            -(
                int(record["min_prediction_row_count"])
                if record["min_prediction_row_count"] is not None
                else -1
            ),
            tuple(record["branch_pair"]),
        ),
    )


def _branch_pair(value: Any) -> tuple[str, str]:
    parts = tuple(str(part) for part in value)
    if len(parts) != 2:
        raise ValueError(f"Expected a two-branch pair, got {parts}")
    return parts[0], parts[1]


def select_readable_pair(
    scored_pairs: list[JSONDict],
    readability: dict[tuple[str, str], JSONDict],
) -> JSONDict:
    rejected_higher_scoring_pairs: list[JSONDict] = []
    sorted_pairs = sorted(
        scored_pairs,
        key=lambda record: (
            -float(record["jaccard_distance"]),
            -int(record.get("min_assigned_count", 0)),
            -int(record.get("total_assigned_count", 0)),
            -(
                int(record["min_prediction_row_count"])
                if record.get("min_prediction_row_count") is not None
                else -1
            ),
            tuple(record["branch_pair"]),
        ),
    )

    for record in sorted_pairs:
        branch_pair = _branch_pair(record["branch_pair"])
        readability_record = readability.get(branch_pair)
        if readability_record is None:
            readability_record = {
                "passes": False,
                "reason": "readability record missing",
            }
        if readability_record.get("passes"):
            return {
                "selected_pair": branch_pair,
                "selected_pair_score": record["jaccard_distance"],
                "selected_pair_record": record,
                "readability_result": readability_record,
                "tie_break_result": {
                    "min_assigned_count": record.get("min_assigned_count", 0),
                    "total_assigned_count": record.get("total_assigned_count", 0),
                    "min_prediction_row_count": record.get("min_prediction_row_count"),
                    "lexicographic_branch_pair": branch_pair,
                },
                "rejected_higher_scoring_pairs": rejected_higher_scoring_pairs,
            }

        rejected_higher_scoring_pairs.append(
            {
                "branch_pair": branch_pair,
                "jaccard_distance": record["jaccard_distance"],
                "reason": f"readability gate failed: {readability_record.get('reason', 'not readable')}",
                "readability_result": readability_record,
            }
        )

    raise ValueError("No readable branch pair")


def _branch_eligibility_records(
    checkpoint_records: list[JSONDict],
    assignment_source: dict[str, str] | None,
    feature_errors: list[str],
) -> list[JSONDict]:
    counts, count_type = _assignment_counts(assignment_source)
    records: list[JSONDict] = []
    for checkpoint in checkpoint_records:
        branch_id = str(checkpoint["parsed_branch_id"])
        reasons: list[str] = []
        assigned_count = counts.get(branch_id, 0)
        if checkpoint["classification"] == "root/global":
            reasons.append(
                "root/global checkpoint is excluded unless explicitly terminal"
            )
        if checkpoint["classification"] != "branch-specific candidate":
            reasons.append(
                f"checkpoint classification is {checkpoint['classification']}"
            )
        if checkpoint["load_status"] != "loadable":
            reasons.append("checkpoint is not loadable")
        if assigned_count <= 0:
            reasons.append("branch has no assignment count in selected source")
        if feature_errors:
            reasons.append("feature source is incompatible")
        records.append(
            {
                "branch_id": branch_id,
                "checkpoint_path": checkpoint["path"],
                "checkpoint_classification": checkpoint["classification"],
                "eligible": not reasons,
                "assigned_count": assigned_count,
                "assigned_count_type": count_type,
                "prediction_row_count": assigned_count
                if count_type == "prediction-row count"
                else None,
                "assigned_admin_group_count": assigned_count
                if count_type == "assigned admin/group count"
                else None,
                "training_sample_count": None,
                "exclusion_reasons": reasons,
            }
        )
    return records


def _read_feature_names_from_source(path: Path) -> tuple[list[str], str]:
    if path.name == "feature_name_reference.csv":
        feature_reference = pd.read_csv(path)
        feature_reference = feature_reference.sort_values("feature_index")
        return feature_reference["feature_name"].astype(str).tolist(), path.name
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ], path.name


def _feature_source_status(
    feature_name_candidates: list[str], branch_checkpoint_files: list[str]
) -> tuple[JSONDict | None, list[str]]:
    if not feature_name_candidates:
        return None, ["feature-name source missing"]

    selected_path = Path(feature_name_candidates[0])
    feature_names, source_type = _read_feature_names_from_source(selected_path)
    errors: list[str] = []

    for checkpoint_path in branch_checkpoint_files:
        with Path(checkpoint_path).open("rb") as handle:
            clf = pickle.load(handle)
        checkpoint_feature_count = int(
            getattr(clf, "n_features_in_", 0) or getattr(clf.tree_, "n_features", 0)
        )
        if len(feature_names) != checkpoint_feature_count:
            errors.append(
                f"feature count mismatch for {Path(checkpoint_path).name}: "
                f"{len(feature_names)} names vs checkpoint count {checkpoint_feature_count}"
            )
        split_indices = [int(index) for index in clf.tree_.feature if int(index) >= 0]
        out_of_bounds = [
            index for index in split_indices if index >= len(feature_names)
        ]
        if out_of_bounds:
            errors.append(
                f"feature split index bounds mismatch for {Path(checkpoint_path).name}: "
                f"indices {out_of_bounds} exceed feature count {len(feature_names)}"
            )

    return {
        "path": str(selected_path),
        "type": source_type,
        "feature_name_count": len(feature_names),
        "status": "selected" if not errors else "unusable",
    }, errors


def _characterize_artifacts(path: Path, provider: Path | None = None) -> JSONDict:
    roots = _artifact_roots(path, provider)
    checkpoint_files: list[str] = []
    branch_checkpoint_files: list[str] = []
    partition_artifacts: list[str] = []
    correspondence_tables: list[str] = []
    feature_name_candidates: list[str] = []

    for root in roots:
        checkpoints = root / "checkpoints"
        if checkpoints.is_dir():
            for checkpoint in sorted(checkpoints.glob("dt_*")):
                checkpoint_files.append(str(checkpoint))
                if checkpoint.name != "dt_":
                    branch_checkpoint_files.append(str(checkpoint))

        space = root / "space_partitions"
        for artifact_name in ("X_branch_id.npy", "s_branch.pkl", "branch_table.npy"):
            artifact_path = space / artifact_name
            if artifact_path.is_file():
                partition_artifacts.append(str(artifact_path))

        correspondence_tables.extend(
            str(table) for table in sorted(root.glob("correspondence_table_*.csv"))
        )
        for feature_filename in ("feature_names.txt", "feature_name_reference.csv"):
            feature_path = root / feature_filename
            if feature_path.is_file():
                feature_name_candidates.append(str(feature_path))

    assignment_source = None
    rejected_assignment_sources: list[dict[str, str]] = []
    x_branch = next(
        (item for item in partition_artifacts if Path(item).name == "X_branch_id.npy"),
        None,
    )
    s_branch = next(
        (item for item in partition_artifacts if Path(item).name == "s_branch.pkl"),
        None,
    )
    correspondence = correspondence_tables[0] if correspondence_tables else None
    if x_branch is not None:
        assignment_source = {
            "path": x_branch,
            "type": "X_branch_id.npy",
            "status": "selected",
        }
        if s_branch is not None:
            rejected_assignment_sources.append(
                {"path": s_branch, "reason": "lower precedence than X_branch_id.npy"}
            )
        if correspondence is not None:
            rejected_assignment_sources.append(
                {
                    "path": correspondence,
                    "reason": "lower precedence than X_branch_id.npy",
                }
            )
    elif s_branch is not None:
        assignment_source = {
            "path": s_branch,
            "type": "s_branch.pkl",
            "status": "selected",
        }
        if correspondence is not None:
            rejected_assignment_sources.append(
                {"path": correspondence, "reason": "lower precedence than s_branch.pkl"}
            )
    elif correspondence is not None:
        assignment_source = {
            "path": correspondence,
            "type": "correspondence_table",
            "status": "selected",
        }

    feature_source, feature_errors = _feature_source_status(
        feature_name_candidates, branch_checkpoint_files
    )
    checkpoint_records = [
        _checkpoint_record(checkpoint) for checkpoint in checkpoint_files
    ]
    branch_eligibility = _branch_eligibility_records(
        checkpoint_records, assignment_source, feature_errors
    )

    return {
        "artifact_roots": [str(root) for root in roots],
        "checkpoint_files": checkpoint_files,
        "checkpoint_records": checkpoint_records,
        "root_global_checkpoint": next(
            (item for item in checkpoint_files if Path(item).name == "dt_"), None
        ),
        "branch_checkpoint_files": branch_checkpoint_files,
        "partition_artifacts": partition_artifacts,
        "correspondence_tables": correspondence_tables,
        "assignment_source": assignment_source,
        "rejected_assignment_sources": rejected_assignment_sources,
        "feature_name_candidates": feature_name_candidates,
        "feature_name_source": feature_source,
        "feature_name_errors": feature_errors,
        "branch_eligibility": branch_eligibility,
        "dt_rules_boundary": "root/global dt_rules are not branch-specific checkpoint evidence"
        if (path / "dt_rules").is_dir()
        else None,
    }


def _minimum_completeness(
    path: Path, provider: Path | None = None
) -> tuple[bool, list[str], JSONDict]:
    if not path.is_dir():
        return False, ["archive directory missing"], {"artifact_roots": [str(path)]}

    characterization = _characterize_artifacts(path, provider)
    missing: list[str] = []

    loadable_branch_records = [
        record
        for record in characterization["checkpoint_records"]
        if record["classification"] == "branch-specific candidate"
        and record["load_status"] == "loadable"
    ]
    if len(loadable_branch_records) < 2:
        missing.append("at least two loadable branch-specific checkpoints")
    eligible_branch_records = [
        record
        for record in characterization["branch_eligibility"]
        if record["eligible"]
    ]
    if len(eligible_branch_records) < 2:
        missing.append("at least two eligible assigned branch checkpoints")

    if characterization["assignment_source"] is None:
        missing.append("branch assignment source")

    if characterization["feature_name_source"] is None:
        missing.append("feature-name source")
    if characterization["feature_name_errors"]:
        missing.extend(characterization["feature_name_errors"])

    if (path / "dt_rules").is_dir() and not characterization["checkpoint_files"]:
        missing.append(
            "root/global dt_rules only; branch checkpoint archive incomplete"
        )

    return not missing, missing, characterization


def _write_audit(
    output_dir: Path,
    selected: Path,
    candidates: list[JSONDict],
    characterization: JSONDict,
    selection: JSONDict,
    signatures: dict[str, JSONDict],
    provider: Path | None = None,
) -> Path:
    month, fs = _archive_identity(selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = (
        output_dir
        / f"geodt_branch_tree_compare_{month}_{fs}_archive_selection_audit.json"
    )
    pair = tuple(str(part) for part in selection["selected_pair"])
    audit = {
        "workflow_mode": "audit-only",
        "selected_archive": str(selected),
        "artifact_provider_path": str(provider) if provider is not None else None,
        "artifact_provider_compatibility": "explicit provider path used"
        if provider is not None
        else None,
        "minimum_required_artifact_completeness": True,
        "candidate_archive_decisions": candidates,
        "fallback_selection_reason": "first candidate satisfying minimum required artifact set",
        "branch_assignment_source": characterization["assignment_source"],
        "feature_name_source": characterization["feature_name_source"],
        "selected_branch_ids": list(pair),
        "selected_pair_score": selection["selected_pair_score"],
        "selected_pair_record": selection["selected_pair_record"],
        "readability_result": selection["readability_result"],
        "tie_break_result": selection["tie_break_result"],
        "rejected_higher_scoring_pairs": selection["rejected_higher_scoring_pairs"],
        "selection_pipeline_source": "shared",
        "selected_signatures": {
            branch_id: {
                "split_feature_set": sorted(signature["split_feature_set"]),
                "split_features_by_depth": signature["split_features_by_depth"],
                "threshold_direction_summary": signature["threshold_direction_summary"],
                "leaf_class_summaries": signature["leaf_class_summaries"],
            }
            for branch_id, signature in signatures.items()
            if branch_id in pair
        },
        "output_figure_paths": [],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit_path


def _run_audit_only(args: argparse.Namespace) -> int:
    paths, error = _candidate_paths(args)
    if error is not None:
        print(error, file=sys.stderr)
        return 1

    decisions: list[JSONDict] = []
    selected: Path | None = None
    provider = args.artifact_provider_path
    for path in paths:
        candidate_provider = (
            provider if provider is not None and path == paths[0] else None
        )
        complete, missing, characterization = _minimum_completeness(
            path, candidate_provider
        )
        decisions.append(
            {
                "path": str(path),
                "artifact_provider_path": str(candidate_provider)
                if candidate_provider is not None
                else None,
                "minimum_required_artifact_completeness": complete,
                "missing_or_invalid_artifacts": missing,
                "artifact_characterization": characterization,
                "decision": "accepted" if complete and selected is None else "rejected",
            }
        )
        if complete and selected is None:
            selected = path

    if selected is None:
        print(
            "No archive satisfied the minimum required branch-specific artifact set; "
            "root/global dt_rules archives are incomplete for branch diagnostics.",
            file=sys.stderr,
        )
        print(
            json.dumps({"candidate_archive_decisions": decisions}, indent=2),
            file=sys.stderr,
        )
        return 1

    output_dir = args.output_dir or selected / "diagnostics" / "output"
    max_plot_depth = args.max_plot_depth if args.max_plot_depth is not None else args.k
    selected_decision = next(
        decision for decision in decisions if decision["path"] == str(selected)
    )
    characterization = selected_decision["artifact_characterization"]
    selection, signatures, _checkpoints = _build_selection(
        characterization, args.k, max_plot_depth
    )
    audit_path = _write_audit(
        output_dir,
        selected,
        decisions,
        characterization,
        selection,
        signatures,
        provider,
    )
    print(f"Archive selection audit written: {audit_path}")
    print(f"assignment source selected: {characterization['assignment_source']}")
    print(f"feature source selected: {characterization['feature_name_source']}")
    if characterization["feature_name_errors"]:
        print(
            f"feature compatibility errors: {characterization['feature_name_errors']}"
        )
    print(
        f"branch checkpoints: {', '.join(Path(item).name for item in characterization['branch_checkpoint_files'])}"
    )
    print(f"checkpoint records: {characterization['checkpoint_records']}")
    print("root/global checkpoint excluded from branch eligibility")
    print(f"eligible branch records: {characterization['branch_eligibility']}")
    print("assigned branch counts recorded from selected assignment source")
    return 0


def _load_feature_names(characterization: JSONDict) -> list[str]:
    feature_source = characterization["feature_name_source"]
    if not isinstance(feature_source, dict):
        raise ValueError("feature-name source missing")
    feature_names, _ = _read_feature_names_from_source(Path(str(feature_source["path"])))
    return feature_names


def _load_checkpoint(path: str) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _eligible_checkpoint_map(
    characterization: JSONDict,
) -> dict[str, JSONDict]:
    return {
        str(record["branch_id"]): record
        for record in characterization["branch_eligibility"]
        if record["eligible"]
    }


def _readability_records(
    signatures: dict[str, JSONDict], max_plot_depth: int
) -> dict[tuple[str, str], JSONDict]:
    records: dict[tuple[str, str], JSONDict] = {}
    branch_ids = sorted(signatures)
    for left_index, branch_a in enumerate(branch_ids):
        for branch_b in branch_ids[left_index + 1 :]:
            has_left_split = any(
                int(depth) < max_plot_depth and features
                for depth, features in signatures[branch_a][
                    "split_features_by_depth"
                ].items()
            )
            has_right_split = any(
                int(depth) < max_plot_depth and features
                for depth, features in signatures[branch_b][
                    "split_features_by_depth"
                ].items()
            )
            passes = bool(has_left_split and has_right_split)
            records[(branch_a, branch_b)] = {
                "passes": passes,
                "reason": "readable"
                if passes
                else "no non-leaf split within plotted depth",
                "plotted_max_depth": max_plot_depth,
                "matched_depth": True,
                "matched_style": True,
                "label_handling": (
                    "paper-friendly feature names with condition-only node text"
                ),
                "minimum_font_size_pt": FIGURE_LAYOUT["tree_fontsize"],
            }
    return records


def _build_selection(
    characterization: JSONDict, k: int, max_plot_depth: int
) -> tuple[JSONDict, JSONDict, JSONDict]:
    feature_names = _load_feature_names(characterization)
    eligible = _eligible_checkpoint_map(characterization)
    signatures: dict[str, JSONDict] = {}
    loaded_checkpoints: JSONDict = {}
    for branch_id, record in eligible.items():
        clf = _load_checkpoint(str(record["checkpoint_path"]))
        loaded_checkpoints[branch_id] = clf
        signature = extract_branch_signature(
            clf, feature_names=feature_names, k=k, branch_id=branch_id
        )
        signature["assigned_count"] = record["assigned_count"]
        signature["prediction_row_count"] = record["prediction_row_count"]
        signatures[branch_id] = signature

    scored_pairs = score_branch_pairs(signatures)
    readability = _readability_records(signatures, max_plot_depth)
    selection = select_readable_pair(scored_pairs, readability)
    return selection, signatures, loaded_checkpoints


def _output_stem(selected: Path, pair: tuple[str, str]) -> str:
    month, fs = _archive_identity(selected)
    return f"geodt_branch_tree_compare_{month}_{fs}_{pair[0]}_vs_{pair[1]}"


def _output_paths(
    output_dir: Path, selected: Path, pair: tuple[str, str]
) -> dict[str, Path]:
    stem = _output_stem(selected, pair)
    return {
        "png": output_dir / f"{stem}.png",
        "metadata": output_dir / f"{stem}_metadata.json",
        "failure_summary": output_dir / f"{stem}_failure_summary.json",
    }


def _existing_output_conflicts(paths: dict[str, Path]) -> list[Path]:
    return [
        path
        for key, path in paths.items()
        if key != "failure_summary" and path.exists()
    ]


def _write_failure_summary(
    output_dir: Path, selected: Path, pair: tuple[str, str], summary: JSONDict
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _output_paths(output_dir, selected, pair)["failure_summary"]
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _render_pair(
    output_paths: dict[str, Path],
    selected: Path,
    pair: tuple[str, str],
    checkpoints: JSONDict,
    feature_names: list[str],
    max_plot_depth: int,
) -> Path:
    png_path = output_paths["png"]
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_LAYOUT["width_inches"], FIGURE_LAYOUT["height_inches"]),
        dpi=FIGURE_LAYOUT["dpi"],
        constrained_layout=FIGURE_LAYOUT["uses_constrained_layout"],
    )
    for axis, branch_id in zip(axes, pair):
        _plot_condition_only_tree(
            axis=axis,
            checkpoint=checkpoints[branch_id],
            feature_names=feature_names,
            max_plot_depth=max_plot_depth,
        )
        axis.set_title(
            f"Branch {branch_id}",
            fontsize=FIGURE_LAYOUT["title_fontsize"],
        )
    fig.savefig(png_path, dpi=FIGURE_LAYOUT["dpi"])
    plt.close(fig)
    return png_path


def _plot_condition_only_tree(
    *,
    axis: Any,
    checkpoint: Any,
    feature_names: list[str],
    max_plot_depth: int,
) -> list[Any]:
    """Render a tree with only interpretable split conditions in node boxes."""
    artists = plot_tree(
        checkpoint,
        feature_names=[_paper_feature_label(name) for name in feature_names],
        class_names=None,
        max_depth=max_plot_depth,
        filled=True,
        rounded=True,
        impurity=False,
        precision=2,
        fontsize=FIGURE_LAYOUT["tree_fontsize"],
        ax=axis,
    )
    diagnostic_pattern = re.compile(
        r"^(?:gini|entropy|log_loss|samples|value|class)\s*="
    )
    preserved_labels = {"true", "false", "(...)"}
    for artist in artists:
        if not hasattr(artist, "get_text") or not hasattr(artist, "set_text"):
            continue
        original_text = artist.get_text()
        kept_lines: list[str] = []
        for line in original_text.splitlines():
            normalized = line.strip().lower()
            if diagnostic_pattern.match(normalized):
                break
            kept_lines.append(line)

        cleaned_text = "\n".join(kept_lines).strip()
        if cleaned_text:
            if " <= " in cleaned_text:
                feature_label, threshold = cleaned_text.rsplit(" <= ", maxsplit=1)
                wrapped_feature_lines: list[str] = []
                for feature_line in feature_label.splitlines():
                    wrapped_feature_lines.extend(
                        textwrap.wrap(
                            feature_line,
                            width=FIGURE_LAYOUT["condition_wrap_width_chars"],
                            break_long_words=False,
                            break_on_hyphens=False,
                        )
                        or [feature_line]
                    )
                cleaned_text = "\n".join(
                    [*wrapped_feature_lines, f"≤ {threshold}"]
                )
            artist.set_text(cleaned_text)
        elif original_text.strip().lower() in preserved_labels:
            artist.set_text(original_text.strip())
        else:
            artist.set_text("Leaf")
    return artists


def write_individual_tree_figure(
    checkpoint_path: Path,
    feature_names_path: Path,
    branch_id: str,
    output_path: Path,
    max_plot_depth: int = 3,
) -> Path:
    """Render one archived branch-specific DecisionTree for a paper figure."""
    checkpoint = _load_checkpoint(str(checkpoint_path))
    feature_names, _ = _read_feature_names_from_source(feature_names_path)
    checkpoint_feature_count = int(
        getattr(checkpoint, "n_features_in_", 0)
        or getattr(checkpoint.tree_, "n_features", 0)
    )
    if len(feature_names) != checkpoint_feature_count:
        raise ValueError(
            f"Feature count mismatch for {checkpoint_path.name}: "
            f"{len(feature_names)} names vs {checkpoint_feature_count} checkpoint features"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(
        figsize=(
            FIGURE_LAYOUT["individual_width_inches"],
            FIGURE_LAYOUT["height_inches"],
        ),
        dpi=FIGURE_LAYOUT["dpi"],
        constrained_layout=FIGURE_LAYOUT["uses_constrained_layout"],
    )
    _plot_condition_only_tree(
        axis=axis,
        checkpoint=checkpoint,
        feature_names=feature_names,
        max_plot_depth=max_plot_depth,
    )
    axis.set_title(f"Branch {branch_id}", fontsize=FIGURE_LAYOUT["title_fontsize"])
    fig.savefig(output_path, dpi=FIGURE_LAYOUT["dpi"], bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_figure_metadata(
    output_paths: dict[str, Path],
    selected: Path,
    provider: Path | None,
    characterization: JSONDict,
    pair: tuple[str, str],
    selection: JSONDict,
    signatures: dict[str, JSONDict],
    png_path: Path,
    k: int,
    max_plot_depth: int,
    archive_discovery_input: JSONDict,
    candidate_archive_decisions: list[JSONDict],
) -> Path:
    month, fs = _archive_identity(selected)
    checkpoint_records = {
        str(record["parsed_branch_id"]): record
        for record in characterization["checkpoint_records"]
    }
    eligibility_records = {
        str(record["branch_id"]): record
        for record in characterization["branch_eligibility"]
    }
    metadata_path = output_paths["metadata"]
    feature_source = dict(characterization["feature_name_source"])
    feature_source["checkpoint_feature_count"] = feature_source.get(
        "feature_name_count"
    )
    feature_source["feature_name_compatibility_result"] = "selected"
    feature_source["evidence_validation_status"] = "confirmed"
    checkpoint_loading_notes = [
        {
            "path": record["path"],
            "load_status": record["load_status"],
            "classification": record["classification"],
            "mismatch_reason": record["mismatch_reason"],
        }
        for record in characterization["checkpoint_records"]
    ]
    evidence_validation_status = {
        "branch_assignment_source": "confirmed"
        if characterization["assignment_source"] is not None
        else "rejected",
        "feature_name_source": "confirmed",
        "checkpoint_branch_id_parsing": "confirmed",
        "root_global_dt_rules_boundary": "confirmed",
        "class_label_mapping": "not_applicable",
    }
    metadata = {
        "workflow_mode": "figure-generation",
        "archive_discovery_input": archive_discovery_input,
        "archive_discovery_root": archive_discovery_input["value"]
        if archive_discovery_input["type"] == "archive-root"
        else None,
        "archive_explicit_path_or_list": archive_discovery_input["value"]
        if archive_discovery_input["type"] in {"archive-path", "archive-list"}
        else None,
        "discovered_archive_candidate_count": len(candidate_archive_decisions),
        "candidate_archive_decisions": candidate_archive_decisions,
        "selected_archive": str(selected),
        "artifact_provider_path": str(provider) if provider is not None else None,
        "artifact_provider_compatibility": "explicit provider path used"
        if provider is not None
        else "not_applicable",
        "selected_run_month": month,
        "forecasting_scope": fs,
        "model_family": "GeoDT",
        "minimum_required_artifact_completeness": True,
        "fallback_selection_reason": "first candidate satisfying minimum required artifact set",
        "branch_assignment_source": characterization["assignment_source"],
        "rejected_assignment_sources": characterization["rejected_assignment_sources"],
        "feature_name_source": feature_source,
        "rejected_feature_name_sources": [],
        "feature_name_compatibility_result": "selected",
        "evidence_validation_status": evidence_validation_status,
        "selected_branch_ids": list(pair),
        "checkpoint_paths_used": {
            branch_id: str(eligibility_records[branch_id]["checkpoint_path"])
            for branch_id in pair
        },
        "checkpoint_records_used": {
            branch_id: checkpoint_records.get(branch_id) for branch_id in pair
        },
        "checkpoint_feature_count": feature_source["checkpoint_feature_count"],
        "checkpoint_loading_notes": checkpoint_loading_notes,
        "checkpoint_branch_id_parsing_evidence_status": "confirmed",
        "root_global_dt_rules_boundary_evidence_status": "confirmed",
        "branch_counts": {
            branch_id: {
                "assigned_count": eligibility_records[branch_id]["assigned_count"],
                "assigned_count_type": eligibility_records[branch_id][
                    "assigned_count_type"
                ],
                "prediction_row_count": eligibility_records[branch_id][
                    "prediction_row_count"
                ],
                "assigned_admin_group_count": eligibility_records[branch_id][
                    "assigned_admin_group_count"
                ],
                "training_sample_count": eligibility_records[branch_id][
                    "training_sample_count"
                ],
            }
            for branch_id in pair
        },
        "root_global_exclusion_decision": "root/global checkpoint excluded from branch eligibility unless explicitly terminal",
        "dt_rules_boundary": "branch-specific local DecisionTree checkpoints used; root/global dt_rules are not used",
        "k": k,
        "actual_plotted_depth": max_plot_depth,
        "dissimilarity_formula_name": "top_k_split_feature_jaccard_distance",
        "dissimilarity_formula_version": "1",
        "selected_pair_score": selection["selected_pair_score"],
        "contrast_descriptor": selection["selected_pair_record"].get(
            "contrast_descriptor"
        ),
        "selected_pair_record": selection["selected_pair_record"],
        "readability_result": selection["readability_result"],
        "figure_layout": FIGURE_LAYOUT,
        "tie_break_result": selection["tie_break_result"],
        "rejected_higher_scoring_pairs": selection["rejected_higher_scoring_pairs"],
        "node_label_policy": (
            "internal nodes show interpretable split conditions only; impurity, "
            "samples, values, and predicted class are omitted"
        ),
        "feature_display_name_policy": (
            "paper-friendly display labels derived from raw feature names; "
            "thresholds remain on stored model scale"
        ),
        "class_label_policy": (
            "class labels omitted from node text; sklearn fill shading retained"
        ),
        "class_output_availability": "available from DecisionTree values; semantic class mapping unavailable",
        "selected_signatures": {
            branch_id: {
                "split_feature_set": sorted(signature["split_feature_set"]),
                "split_features_by_depth": signature["split_features_by_depth"],
                "threshold_direction_summary": signature["threshold_direction_summary"],
                "leaf_class_summaries": signature["leaf_class_summaries"],
            }
            for branch_id, signature in signatures.items()
            if branch_id in pair
        },
        "output_figure_paths": [str(png_path)],
        "audit_summary_paths": [],
        "reproduction_report_paths": [],
        "failure_summary_path": None,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata_path


def _write_reproduction_failure_summary(
    output_dir: Path,
    metadata_path: Path,
    summary: JSONDict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        output_dir / f"{metadata_path.stem}_reproduction_failure_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary_path


def _run_reproduction(args: argparse.Namespace) -> int:
    metadata_path = args.reproduce_from
    if metadata_path is None:
        print("No reproduction metadata path was provided.", file=sys.stderr)
        return 1
    if not metadata_path.is_file():
        print(f"Reproduction metadata missing: {metadata_path}", file=sys.stderr)
        return 1

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir or metadata_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_archive = metadata["selected_archive"]
    selected_branch_ids = [
        str(branch_id) for branch_id in metadata["selected_branch_ids"]
    ]
    checkpoint_paths = {
        str(branch_id): Path(path)
        for branch_id, path in metadata.get("checkpoint_paths_used", {}).items()
    }
    feature_source = metadata.get("feature_name_source") or {}
    feature_source_path = Path(str(feature_source.get("path", "")))
    missing_paths = [
        str(path) for path in checkpoint_paths.values() if not path.is_file()
    ]
    if not feature_source_path.is_file():
        missing_paths.append(str(feature_source_path))
    if missing_paths:
        summary = {
            "failure_stage": "reproduction-mismatch",
            "mismatch_type": "missing recorded artifact",
            "missing_artifacts": missing_paths,
            "evidence_mismatches": [],
            "source_metadata_path": str(metadata_path),
            "reselected_pair": False,
        }
        summary_path = _write_reproduction_failure_summary(
            output_dir, metadata_path, summary
        )
        print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
        print(f"Reproduction failure summary written: {summary_path}", file=sys.stderr)
        return 1

    feature_names = [
        line.strip()
        for line in feature_source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    k = int(metadata.get("k", 3))
    recovered_signatures: dict[str, JSONDict] = {}
    evidence_mismatches: list[JSONDict] = []
    for branch_id in selected_branch_ids:
        checkpoint_path = checkpoint_paths[branch_id]
        clf = _load_checkpoint(str(checkpoint_path))
        recovered_signature = extract_branch_signature(
            clf, feature_names=feature_names, k=k, branch_id=branch_id
        )
        recovered_signatures[branch_id] = recovered_signature
        recorded_features = set(
            metadata.get("selected_signatures", {})
            .get(branch_id, {})
            .get("split_feature_set", [])
        )
        recovered_features = set(recovered_signature["split_feature_set"])
        if recovered_features != recorded_features:
            evidence_mismatches.append(
                {
                    "field": f"selected_signatures.{branch_id}.split_feature_set",
                    "recorded": sorted(recorded_features),
                    "recovered": sorted(recovered_features),
                }
            )

    scored_pairs = score_branch_pairs(recovered_signatures)
    selected_pair = tuple(selected_branch_ids)
    recovered_pair_record = next(
        (
            record
            for record in scored_pairs
            if _branch_pair(record["branch_pair"]) == selected_pair
        ),
        None,
    )
    if recovered_pair_record is None:
        evidence_mismatches.append(
            {
                "field": "selected_pair_score",
                "recorded": metadata.get("selected_pair_score"),
                "recovered": None,
            }
        )
        recovered_score = None
    else:
        recovered_score = float(recovered_pair_record["jaccard_distance"])
        recorded_score = float(metadata["selected_pair_score"])
        if not np.isclose(recovered_score, recorded_score):
            evidence_mismatches.append(
                {
                    "field": "selected_pair_score",
                    "recorded": recorded_score,
                    "recovered": recovered_score,
                }
            )

    if evidence_mismatches:
        summary = {
            "failure_stage": "reproduction-mismatch",
            "mismatch_type": "changed recorded artifact",
            "missing_artifacts": [],
            "evidence_mismatches": evidence_mismatches,
            "source_metadata_path": str(metadata_path),
            "reselected_pair": False,
        }
        summary_path = _write_reproduction_failure_summary(
            output_dir, metadata_path, summary
        )
        print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
        print(f"Reproduction failure summary written: {summary_path}", file=sys.stderr)
        return 1

    report_path = output_dir / f"{metadata_path.stem}_reproduction_report.json"
    report = {
        "workflow_mode": "reproduction",
        "source_metadata_path": str(metadata_path),
        "selected_archive": selected_archive,
        "artifact_provider_path": metadata.get("artifact_provider_path"),
        "selected_branch_ids": selected_branch_ids,
        "checkpoint_paths_used": metadata.get("checkpoint_paths_used", {}),
        "feature_name_source": feature_source,
        "k": k,
        "actual_plotted_depth": metadata.get("actual_plotted_depth"),
        "selected_signatures": metadata.get("selected_signatures", {}),
        "recovered_signatures": {
            branch_id: {
                "split_feature_set": sorted(signature["split_feature_set"]),
                "split_features_by_depth": signature["split_features_by_depth"],
            }
            for branch_id, signature in recovered_signatures.items()
        },
        "selected_pair_score": metadata["selected_pair_score"],
        "recovered_pair_score": recovered_score,
        "dissimilarity_formula_name": metadata.get("dissimilarity_formula_name"),
        "dissimilarity_formula_version": metadata.get("dissimilarity_formula_version"),
        "reselected_pair": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Reproduction report written: {report_path}")
    return 0


def _run_figure_generation(args: argparse.Namespace) -> int:
    paths, error = _candidate_paths(args)
    if error is not None:
        print(error, file=sys.stderr)
        return 1

    provider = args.artifact_provider_path
    output_dir = args.output_dir or paths[0] / "diagnostics" / "output"
    decisions: list[JSONDict] = []
    selected: Path | None = None
    characterization: JSONDict | None = None
    for path in paths:
        candidate_provider = (
            provider if provider is not None and path == paths[0] else None
        )
        complete, missing, candidate_characterization = _minimum_completeness(
            path, candidate_provider
        )
        decision = {
            "path": str(path),
            "artifact_provider_path": str(candidate_provider)
            if candidate_provider is not None
            else None,
            "minimum_required_artifact_completeness": complete,
            "missing_or_invalid_artifacts": missing,
            "artifact_characterization": candidate_characterization,
            "decision": "accepted" if complete and selected is None else "rejected",
            "reason": "first candidate satisfying minimum required artifact set"
            if complete and selected is None
            else "minimum required artifact set incomplete",
        }
        decisions.append(decision)
        if complete and selected is None:
            selected = path
            characterization = candidate_characterization
            break
        print(
            f"Archive rejected: {path}; missing or invalid artifacts: {missing}",
            file=sys.stderr,
        )

    if selected is None or characterization is None:
        missing = [
            artifact
            for decision in decisions
            for artifact in decision["missing_or_invalid_artifacts"]
        ]
        summary = {
            "failure_stage": "preflight",
            "missing_or_invalid_artifacts": missing,
            "evidence_mismatch": None,
            "candidate_archive_decisions": decisions,
            "recommended_next_action": "provide a bounded GeoDT archive path/root with branch checkpoints, assignments, and feature names",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = (
            output_dir / "geodt_branch_tree_compare_preflight_failure_summary.json"
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
        print(f"Failure summary written: {summary_path}", file=sys.stderr)
        return 1

    max_plot_depth = args.max_plot_depth if args.max_plot_depth is not None else args.k
    selection, signatures, checkpoints = _build_selection(
        characterization, args.k, max_plot_depth
    )
    pair = _branch_pair(selection["selected_pair"])
    output_paths = _output_paths(output_dir, selected, pair)
    conflicts = _existing_output_conflicts(output_paths)
    if conflicts and not args.overwrite:
        summary = {
            "failure_stage": "output-conflict",
            "missing_or_invalid_artifacts": [],
            "evidence_mismatch": None,
            "candidate_archive_decisions": decisions,
            "conflicting_output_paths": [str(path) for path in conflicts],
            "recommended_next_action": "rerun with --overwrite or choose an empty output directory",
        }
        failure_summary_path = _write_failure_summary(
            output_dir, selected, pair, summary
        )
        print(
            f"Output path exists and overwrite is disabled: {conflicts[0]}",
            file=sys.stderr,
        )
        print(f"Failure summary written: {failure_summary_path}", file=sys.stderr)
        return 1

    feature_names = _load_feature_names(characterization)
    png_path = _render_pair(
        output_paths, selected, pair, checkpoints, feature_names, max_plot_depth
    )
    metadata_path = _write_figure_metadata(
        output_paths,
        selected,
        provider,
        characterization,
        pair,
        selection,
        signatures,
        png_path,
        args.k,
        max_plot_depth,
        _archive_discovery_input(args),
        decisions,
    )
    print(f"Figure written: {png_path}")
    print(f"Metadata written: {metadata_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.reproduce_from is not None:
        return _run_reproduction(args)
    if args.audit_only:
        return _run_audit_only(args)
    return _run_figure_generation(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
