#!/usr/bin/env python3
"""Diagnose GeoRF partition lineage and visualization scope issues using only stdlib.

This script builds branch scopes from per-branch partition CSVs in result_GeoRF/vis,
compares them with the current correspondence table, computes metrics, and emits
diagnostics/reports required by the visualization fix task.

It avoids non-stdlib dependencies so it can run inside restricted environments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

ResultDir = str
BranchId = str
Uid = str


@dataclass
class BranchInfo:
    round_id: int
    branch_id: BranchId
    parent_id: Optional[BranchId]
    csv_path: str
    scope: Set[Uid] = field(default_factory=set)
    child_map: Dict[BranchId, Set[Uid]] = field(default_factory=lambda: defaultdict(set))

    def child_labels(self) -> Sequence[BranchId]:
        base = "" if self.branch_id in ("", "root") else self.branch_id
        return [f"{base}0", f"{base}1"]


def parse_branch_files(vis_dir: str) -> Dict[BranchId, BranchInfo]:
    branch_infos: Dict[BranchId, BranchInfo] = {}
    for name in sorted(os.listdir(vis_dir)):
        if not name.startswith("final_partitions_round_") or not name.endswith(".csv"):
            continue
        parts = name.replace(".csv", "").split("_")
        try:
            round_idx = int(parts[3])
        except (IndexError, ValueError):
            continue
        try:
            branch_token = parts[5]
        except IndexError:
            continue
        branch_id = "" if branch_token == "root" else branch_token
        parent_id = None if branch_id == "" else branch_id[:-1]
        csv_path = os.path.join(vis_dir, name)
        info = BranchInfo(round_id=round_idx, branch_id=branch_id or "root", parent_id=parent_id, csv_path=csv_path)
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                uid_raw = row.get("FEWSNET_admin_code")
                if uid_raw is None:
                    continue
                uid = uid_raw.strip()
                if uid == "":
                    continue
                info.scope.add(uid)
                s0 = _safe_int(row.get("s0_partition"))
                s1 = _safe_int(row.get("s1_partition"))
                base = "" if info.branch_id in ("", "root") else info.branch_id
                child0 = f"{base}0"
                child1 = f"{base}1"
                if s0:
                    info.child_map[child0].add(uid)
                if s1:
                    info.child_map[child1].add(uid)
        branch_infos[info.branch_id] = info
    return branch_infos


def _safe_int(value: Optional[str]) -> int:
    if value is None:
        return 0
    value = value.strip()
    if value == "":
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def gather_all_uids(branch_infos: Dict[BranchId, BranchInfo]) -> Set[Uid]:
    all_uids: Set[Uid] = set()
    for info in branch_infos.values():
        all_uids.update(info.scope)
    return all_uids


def build_expected_terminals(branch_infos: Dict[BranchId, BranchInfo]) -> Dict[Uid, BranchId]:
    if "root" not in branch_infos:
        raise RuntimeError("Missing root branch information (final_partitions_round_0_branch_root.csv)")

    assignments: Dict[Uid, BranchId] = {}

    def recurse(branch_id: BranchId, uids: Set[Uid]):
        info = branch_infos.get(branch_id if branch_id else "root")
        if info is None:
            for uid in uids:
                assignments[uid] = branch_id
            return
        child0, child1 = info.child_labels()
        child0_uids = info.child_map.get(child0, set())
        child1_uids = info.child_map.get(child1, set())
        used: Set[Uid] = set()
        if child0_uids:
            recurse(child0, child0_uids)
            used.update(child0_uids)
        if child1_uids:
            recurse(child1, child1_uids)
            used.update(child1_uids)
        leftover = set(uids) - used
        if leftover:
            for uid in leftover:
                assignments[uid] = branch_id

    root_info = branch_infos["root"]
    recurse("", set(root_info.scope))
    return assignments


def load_current_correspondence(corr_path: str) -> Dict[Uid, Set[str]]:
    if not os.path.exists(corr_path):
        raise FileNotFoundError(corr_path)
    mapping: Dict[Uid, Set[str]] = defaultdict(set)
    with open(corr_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            uid_raw = row.get("FEWSNET_admin_code")
            pid_raw = row.get("partition_id")
            if uid_raw is None or pid_raw is None:
                continue
            uid = uid_raw.strip()
            pid = pid_raw.strip()
            if uid == "" or pid == "":
                continue
            mapping[uid].add(pid)
    return mapping


def invert_mapping(mapping: Dict[Uid, Set[str]]) -> Dict[str, Set[Uid]]:
    inverted: Dict[str, Set[Uid]] = defaultdict(set)
    for uid, label_set in mapping.items():
        for label in label_set:
            inverted[label].add(uid)
    return inverted


def compute_iou(expected: Dict[str, Set[Uid]], actual: Dict[str, Set[Uid]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for label in sorted(expected.keys()):
        exp_set = expected[label]
        act_set = actual.get(label, set())
        intersection = exp_set & act_set
        union = exp_set | act_set
        iou = len(intersection) / len(union) if union else 1.0
        rows.append({
            "label": label if label != "" else "root",
            "expected_count": len(exp_set),
            "actual_count": len(act_set),
            "intersection_count": len(intersection),
            "union_count": len(union),
            "iou": f"{iou:.6f}",
        })
    for label in sorted(actual.keys()):
        if label in expected:
            continue
        act_set = actual[label]
        rows.append({
            "label": label,
            "expected_count": 0,
            "actual_count": len(act_set),
            "intersection_count": 0,
            "union_count": len(act_set),
            "iou": "0.000000",
        })
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, object]], header: Optional[Sequence[str]] = None) -> None:
    if not rows:
        return
    if header is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        header = keys
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def hash_uid_set(uids: Iterable[Uid]) -> str:
    digest = hashlib.sha256()
    for uid in sorted(uids):
        digest.update(uid.encode("utf-8"))
    return digest.hexdigest()


def generate_scope_report(
    branch_infos: Dict[BranchId, BranchInfo],
    all_uids: Set[Uid],
    output_path: str,
    assume_scoped: bool,
) -> Dict[str, int]:
    metrics: Dict[str, int] = {}
    lines: List[str] = []
    for info in sorted(branch_infos.values(), key=lambda x: (x.round_id, x.branch_id)):
        if info.branch_id == "root":
            continue
        if assume_scoped:
            mismatch: Set[Uid] = set()
            actual_scope = info.scope
        else:
            mismatch = all_uids - info.scope
            actual_scope = all_uids
        metrics_key = f"round{info.round_id}_branch{info.branch_id or 'root'}"
        metrics[metrics_key] = len(mismatch)
        lines.append(
            json.dumps({
                "round": info.round_id,
                "branch": info.branch_id or "root",
                "expected_scope": len(info.scope),
                "actual_scope": len(actual_scope),
                "mismatched": len(mismatch),
                "mismatched_uids": sorted(mismatch),
            })
        )
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return metrics


def label_frequencies(branch_infos: Dict[BranchId, BranchInfo], output_path: str) -> None:
    rows: List[Dict[str, object]] = []
    for info in sorted(branch_infos.values(), key=lambda x: (x.round_id, x.branch_id)):
        child0, child1 = info.child_labels()
        rows.append({
            "round": info.round_id,
            "branch": info.branch_id or "root",
            "child": child0,
            "count": len(info.child_map.get(child0, set())),
        })
        rows.append({
            "round": info.round_id,
            "branch": info.branch_id or "root",
            "child": child1,
            "count": len(info.child_map.get(child1, set())),
        })
    write_csv(output_path, rows, header=["round", "branch", "child", "count"])


def lineage_trace(branch_infos: Dict[BranchId, BranchInfo],
                  expected_assignments: Dict[Uid, BranchId],
                  actual_mapping: Dict[Uid, Set[str]],
                  iou_rows: Sequence[Dict[str, object]],
                  scope_metrics: Dict[str, int],
                  output_path: str) -> None:
    lines: List[str] = []
    lines.append("# Partition Lineage Trace")
    lines.append(f"total_uids={len(expected_assignments)}")
    dup_uids = {uid: labels for uid, labels in actual_mapping.items() if len(labels) > 1}
    lines.append(f"duplicate_uid_labels={len(dup_uids)}")
    if dup_uids:
        sample = list(sorted(dup_uids.items()))[:10]
        lines.append(f"duplicate_uid_sample={sample}")
    lines.append("## branch_scopes")
    for info in sorted(branch_infos.values(), key=lambda x: (x.round_id, x.branch_id)):
        child0, child1 = info.child_labels()
        lines.append(
            json.dumps({
                "round": info.round_id,
                "branch": info.branch_id or "root",
                "scope": len(info.scope),
                "child0": child0,
                "child0_count": len(info.child_map.get(child0, set())),
                "child1": child1,
                "child1_count": len(info.child_map.get(child1, set())),
                "scope_hash": hash_uid_set(info.scope),
            })
        )
    lines.append("## terminal_vs_expected_iou")
    for row in iou_rows:
        lines.append(json.dumps(row))
    lines.append("## scope_mismatches")
    for key, value in scope_metrics.items():
        lines.append(f"{key}={value}")
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def hash_compare(branch_infos: Dict[BranchId, BranchInfo], output_path: str) -> None:
    lines: List[str] = []
    for info in sorted(branch_infos.values(), key=lambda x: (x.round_id, x.branch_id)):
        scope_hash = hash_uid_set(info.scope)
        child_hashes = {
            child: hash_uid_set(uids)
            for child, uids in sorted(info.child_map.items())
        }
        lines.append(json.dumps({
            "round": info.round_id,
            "branch": info.branch_id or "root",
            "scope_hash": scope_hash,
            "children_hash": child_hashes,
        }))
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def invert_expected(assignments: Dict[Uid, BranchId]) -> Dict[BranchId, Set[Uid]]:
    inverted: Dict[BranchId, Set[Uid]] = defaultdict(set)
    for uid, branch in assignments.items():
        inverted[branch].add(uid)
    return inverted


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose GeoRF partition lineage")
    parser.add_argument("result_dir", help="Path to result_GeoRF directory")
    parser.add_argument("--correspondence", default=None,
                        help="Optional explicit path to correspondence table")
    parser.add_argument(
        "--write-corrected-correspondence",
        action="store_true",
        help="Emit a corrected correspondence CSV based on expected terminal labels",
    )
    parser.add_argument(
        "--assume-scoped",
        action="store_true",
        help="Assume scoped per-round maps (post-fix validation)",
    )
    args = parser.parse_args(argv)

    vis_dir = os.path.join(args.result_dir, "vis")
    if not os.path.isdir(vis_dir):
        print(f"error: {vis_dir} not found", file=sys.stderr)
        return 1

    branch_infos = parse_branch_files(vis_dir)
    if not branch_infos:
        print("error: no branch files found", file=sys.stderr)
        return 1

    all_uids = gather_all_uids(branch_infos)
    expected_assignments = build_expected_terminals(branch_infos)
    expected_labels = invert_expected(expected_assignments)

    corr_path = args.correspondence or os.path.join(args.result_dir, "correspondence_table_Q4_2015.csv")
    actual_mapping = load_current_correspondence(corr_path)
    actual_labels = invert_mapping(actual_mapping)

    scope_report_path = os.path.join(args.result_dir, "scope_mismatch_uids_round_k.txt")
    label_freqs_path = os.path.join(args.result_dir, "label_freqs_by_round.csv")
    iou_path = os.path.join(args.result_dir, "terminal_vs_expected_iou.csv")
    lineage_path = os.path.join(args.result_dir, "vis", "lineage_trace.txt")
    hash_path = os.path.join(args.result_dir, "vis", "hash_compare.txt")

    scope_metrics = generate_scope_report(
        branch_infos,
        all_uids,
        scope_report_path,
        assume_scoped=args.assume_scoped,
    )
    label_frequencies(branch_infos, label_freqs_path)

    iou_rows = compute_iou(expected_labels, actual_labels)
    write_csv(iou_path, iou_rows, header=["label", "expected_count", "actual_count", "intersection_count", "union_count", "iou"])

    lineage_trace(branch_infos, expected_assignments, actual_mapping, iou_rows, scope_metrics, lineage_path)
    hash_compare(branch_infos, hash_path)

    summary = OrderedDict()
    summary["total_uids"] = len(all_uids)
    summary["n_branches"] = len(branch_infos)
    summary["n_expected_labels"] = len(expected_labels)
    summary["n_actual_labels"] = len(actual_labels)
    summary["n_duplicate_uid_assignments"] = sum(1 for labels in actual_mapping.values() if len(labels) > 1)
    summary["n_missing_expected_labels"] = sum(1 for label in expected_labels if label not in actual_labels)
    summary["n_out_of_scope_on_round_maps"] = scope_metrics

    if args.write_corrected_correspondence:
        def _uid_sort_key(item: tuple[str, str]) -> tuple[int, str]:
            uid, _ = item
            try:
                return (0, str(int(float(uid))))
            except ValueError:
                return (1, uid)

        corrected_rows = [
            {
                "FEWSNET_admin_code": uid,
                "partition_id": label if label != "" else "root",
                "branch_id": label if label != "" else "root",
            }
            for uid, label in sorted(expected_assignments.items(), key=_uid_sort_key)
        ]
        corrected_path = os.path.join(args.result_dir, "correspondence_table_Q4_2015_corrected.csv")
        write_csv(corrected_path, corrected_rows, header=["FEWSNET_admin_code", "partition_id", "branch_id"])
        summary["corrected_correspondence"] = corrected_path

    stats_path = os.path.join(args.result_dir, "diagnostics_summary.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
