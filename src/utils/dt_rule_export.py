"""Utilities for exporting trained DecisionTreeClassifier artifacts to tabular CSVs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


_TREE_LEAF = -2


def _normalize_feature_names(clf, feature_names: Sequence[str] | None) -> list[str]:
    n_features = int(getattr(clf, "n_features_in_", 0) or getattr(clf.tree_, "n_features", 0))
    names = [str(name) for name in feature_names] if feature_names is not None else []
    if len(names) < n_features:
        names.extend([f"feature_{idx}" for idx in range(len(names), n_features)])
    return names[:n_features]


def _normalize_class_names(clf, class_names: Sequence[str] | None) -> list[str]:
    n_classes = int(clf.tree_.value.shape[2])
    if class_names is not None:
        names = [str(name) for name in class_names]
    elif hasattr(clf, "classes_"):
        names = [str(name) for name in clf.classes_]
    else:
        names = []
    if len(names) < n_classes:
        names.extend([f"class_{idx}" for idx in range(len(names), n_classes)])
    return names[:n_classes]


def tree_to_dataframe(clf, feature_names: Sequence[str] | None = None, class_names: Sequence[str] | None = None) -> pd.DataFrame:
    """Convert a trained decision tree into a rule-level DataFrame."""

    if clf is None or not hasattr(clf, "tree_"):
        raise TypeError("Expected a fitted DecisionTreeClassifier-like object with tree_.")

    tree_ = clf.tree_
    resolved_feature_names = _normalize_feature_names(clf, feature_names)
    resolved_class_names = _normalize_class_names(clf, class_names)

    rows: list[dict[str, object]] = []

    def recurse(node: int, path_conditions: list[str]) -> None:
        feature_idx = int(tree_.feature[node])
        if feature_idx == _TREE_LEAF:
            value = np.asarray(tree_.value[node][0], dtype=float)
            total_weight = float(value.sum())
            n_samples = int(tree_.n_node_samples[node])
            class_idx = int(np.argmax(value)) if value.size else 0
            predicted_class = resolved_class_names[class_idx] if class_idx < len(resolved_class_names) else str(class_idx)
            confidence = (float(value[class_idx]) / total_weight) if total_weight > 0 else 0.0
            rows.append(
                {
                    "Rule": " AND ".join(path_conditions) if path_conditions else "ROOT",
                    "Class": predicted_class,
                    "Confidence": round(confidence, 4),
                    "Samples": n_samples,
                    "Conditions_List": json.dumps(path_conditions, ensure_ascii=False),
                }
            )
            return

        feature_name = resolved_feature_names[feature_idx] if 0 <= feature_idx < len(resolved_feature_names) else f"feature_{feature_idx}"
        threshold = float(tree_.threshold[node])
        recurse(int(tree_.children_left[node]), path_conditions + [f"{feature_name} <= {threshold:.6f}"])
        recurse(int(tree_.children_right[node]), path_conditions + [f"{feature_name} > {threshold:.6f}"])

    recurse(0, [])
    if not rows:
        return pd.DataFrame(columns=["Rule", "Class", "Confidence", "Samples", "Conditions_List"])
    return pd.DataFrame(rows).sort_values(by="Samples", ascending=False, kind="mergesort").reset_index(drop=True)


def tree_nodes_to_dataframe(clf, feature_names: Sequence[str] | None = None, class_names: Sequence[str] | None = None) -> pd.DataFrame:
    """Export node-level decision tree structure as a tabular DataFrame."""

    if clf is None or not hasattr(clf, "tree_"):
        raise TypeError("Expected a fitted DecisionTreeClassifier-like object with tree_.")

    tree_ = clf.tree_
    resolved_feature_names = _normalize_feature_names(clf, feature_names)
    resolved_class_names = _normalize_class_names(clf, class_names)
    rows: list[dict[str, object]] = []

    stack: list[tuple[int, int, int]] = [(0, -1, 0)]  # node_id, parent_id, depth
    while stack:
        node_id, parent_id, depth = stack.pop()
        feature_idx = int(tree_.feature[node_id])
        is_leaf = feature_idx == _TREE_LEAF
        left_child = int(tree_.children_left[node_id])
        right_child = int(tree_.children_right[node_id])
        value = np.asarray(tree_.value[node_id][0], dtype=float)
        total_weight = float(value.sum())
        class_idx = int(np.argmax(value)) if value.size else 0
        predicted_class = resolved_class_names[class_idx] if class_idx < len(resolved_class_names) else str(class_idx)
        confidence = (float(value[class_idx]) / total_weight) if total_weight > 0 else 0.0
        class_distribution = {
            resolved_class_names[idx] if idx < len(resolved_class_names) else str(idx): float(v)
            for idx, v in enumerate(value.tolist())
        }

        rows.append(
            {
                "Node_ID": node_id,
                "Parent_Node_ID": parent_id,
                "Depth": depth,
                "Is_Leaf": int(is_leaf),
                "Left_Child": left_child if not is_leaf else -1,
                "Right_Child": right_child if not is_leaf else -1,
                "Feature_Index": feature_idx if not is_leaf else -1,
                "Feature": resolved_feature_names[feature_idx] if (not is_leaf and 0 <= feature_idx < len(resolved_feature_names)) else "",
                "Threshold": float(tree_.threshold[node_id]) if not is_leaf else np.nan,
                "Impurity": float(tree_.impurity[node_id]),
                "Samples": int(tree_.n_node_samples[node_id]),
                "Weighted_Samples": float(tree_.weighted_n_node_samples[node_id]),
                "Predicted_Class": predicted_class,
                "Confidence": round(confidence, 4),
                "Class_Distribution": json.dumps(class_distribution, ensure_ascii=False),
            }
        )

        if not is_leaf:
            stack.append((right_child, node_id, depth + 1))
            stack.append((left_child, node_id, depth + 1))

    if not rows:
        return pd.DataFrame(
            columns=[
                "Node_ID",
                "Parent_Node_ID",
                "Depth",
                "Is_Leaf",
                "Left_Child",
                "Right_Child",
                "Feature_Index",
                "Feature",
                "Threshold",
                "Impurity",
                "Samples",
                "Weighted_Samples",
                "Predicted_Class",
                "Confidence",
                "Class_Distribution",
            ]
        )
    return pd.DataFrame(rows).sort_values(by="Node_ID", ascending=True, kind="mergesort").reset_index(drop=True)


def export_dt_rules(clf, feature_names: Sequence[str] | None, class_names: Sequence[str] | None, out_csv: str | Path) -> int:
    """Export tree rules to CSV and return number of exported rules."""

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rules_df = tree_to_dataframe(clf, feature_names=feature_names, class_names=class_names)
    rules_df.to_csv(out_path, index=False)
    return int(len(rules_df))


def export_dt_node_dump(clf, feature_names: Sequence[str] | None, class_names: Sequence[str] | None, out_csv: str | Path) -> int:
    """Export node-level tree dump to CSV and return number of exported nodes."""

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nodes_df = tree_nodes_to_dataframe(clf, feature_names=feature_names, class_names=class_names)
    nodes_df.to_csv(out_path, index=False)
    return int(len(nodes_df))


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 for a file path."""

    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
