#!/usr/bin/env python3
"""Render lightweight PNG summaries for partition scopes without external deps."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import struct
import zlib
from typing import Dict, Iterable, List, Sequence, Tuple

Uid = str
Label = str

LEFT_COLOR = (31, 119, 180)
RIGHT_COLOR = (255, 127, 14)
NODATA_COLOR = (221, 221, 221)


def parse_branch_files(vis_dir: str) -> Dict[Label, Dict[str, object]]:
    branches: Dict[Label, Dict[str, object]] = {}
    for name in sorted(os.listdir(vis_dir)):
        if not name.startswith("final_partitions_round_") or not name.endswith(".csv"):
            continue
        path = os.path.join(vis_dir, name)
        parts = name[:-4].split("_")
        round_id = int(parts[3])
        branch_token = parts[5]
        branch_id = "" if branch_token == "root" else branch_token
        child0 = (branch_id + "0") if branch_id else "0"
        child1 = (branch_id + "1") if branch_id else "1"
        scope: List[Uid] = []
        child_map = {child0: [], child1: []}
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                uid = row.get("FEWSNET_admin_code")
                if uid is None:
                    continue
                uid = uid.strip()
                if not uid:
                    continue
                scope.append(uid)
                s0 = int(float(row.get("s0_partition", "0") or 0))
                s1 = int(float(row.get("s1_partition", "0") or 0))
                if s0:
                    child_map[child0].append(uid)
                if s1:
                    child_map[child1].append(uid)
        branches[branch_id or "root"] = {
            "round": round_id,
            "scope": scope,
            "children": child_map,
        }
    return branches


def read_final_correspondence(path: str) -> List[Tuple[Uid, Label]]:
    rows: List[Tuple[Uid, Label]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            uid = row.get("FEWSNET_admin_code")
            label = row.get("partition_id")
            if uid is None or label is None:
                continue
            uid = uid.strip()
            label = label.strip()
            if not uid or not label:
                continue
            rows.append((uid, label))
    return rows


def palette_for_final(label: Label) -> Tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    r, g, b = digest[0], digest[8], digest[16]
    # Soften colors for readability
    return tuple(int(0.6 * c + 0.4 * 255) for c in (r, g, b))


def write_png(path: str, pixels: List[List[Tuple[int, int, int]]]) -> None:
    height = len(pixels)
    width = len(pixels[0]) if height > 0 else 0
    raw = bytearray()
    for row in pixels:
        raw.append(0)  # filter type 0
        for r, g, b in row:
            raw.extend([r, g, b])
    compressed = zlib.compress(bytes(raw), level=9)
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        # IHDR
        ihdr = struct.pack(
            ">IIBBBBB",
            width,
            height,
            8,
            2,
            0,
            0,
            0,
        )
        fh.write(_chunk(b"IHDR", ihdr))
        fh.write(_chunk(b"IDAT", compressed))
        fh.write(_chunk(b"IEND", b""))


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(
        ">I", len(data)
    ) + chunk_type + data + struct.pack(
        ">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    )


def arrange_grid(items: Sequence[Tuple[Uid, Label]], colors: Dict[Label, Tuple[int, int, int]]) -> List[List[Tuple[int, int, int]]]:
    total = len(items)
    if total == 0:
        return [[NODATA_COLOR]]
    width = max(1, int(math.sqrt(total)))
    height = math.ceil(total / width)
    pixels: List[List[Tuple[int, int, int]]] = []
    idx = 0
    for _ in range(height):
        row: List[Tuple[int, int, int]] = []
        for _ in range(width):
            if idx < total:
                uid, label = items[idx]
                row.append(colors.get(label, NODATA_COLOR))
            else:
                row.append(NODATA_COLOR)
            idx += 1
        pixels.append(row)
    return pixels


def render_round_maps(result_dir: str, branches: Dict[Label, Dict[str, object]]) -> List[str]:
    vis_dir = os.path.join(result_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    outputs: List[str] = []
    for branch_label, meta in sorted(branches.items(), key=lambda item: (item[1]["round"], item[0])):
        if branch_label == "root":
            continue
        round_id = meta["round"]
        child_map = meta["children"]
        child_labels = sorted(child_map.keys())
        colors = {
            child_labels[0]: LEFT_COLOR,
            child_labels[1]: RIGHT_COLOR,
        }
        ordered: List[Tuple[Uid, Label]] = []
        for uid in sorted(meta["scope"], key=lambda x: (len(x), x)):
            if uid in child_map[child_labels[0]]:
                ordered.append((uid, child_labels[0]))
            elif uid in child_map[child_labels[1]]:
                ordered.append((uid, child_labels[1]))
            else:
                ordered.append((uid, "__nodata__"))
        colors["__nodata__"] = NODATA_COLOR
        pixels = arrange_grid(ordered, colors)
        save_path = os.path.join(
            vis_dir,
            f"partition_map_round_{round_id}_branch_{branch_label}_scoped.png",
        )
        write_png(save_path, pixels)
        outputs.append(save_path)
        # Write UID summary for diagnostics
        uid_csv = os.path.splitext(save_path)[0] + "_uids.csv"
        with open(uid_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["FEWSNET_admin_code", "partition_id"])
            for uid, label in ordered:
                writer.writerow([uid, label if label != "__nodata__" else ""])
    return outputs


def render_final_map(result_dir: str, correspondence_rows: List[Tuple[Uid, Label]]) -> str:
    vis_dir = os.path.join(result_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    colors: Dict[Label, Tuple[int, int, int]] = {}
    ordered = sorted(correspondence_rows, key=lambda x: (len(x[1]), x[1], int(float(x[0]))))
    for _, label in ordered:
        colors.setdefault(label, palette_for_final(label))
    pixels = arrange_grid(ordered, colors)
    save_path = os.path.join(vis_dir, "final_partition_map.png")
    write_png(save_path, pixels)
    uid_csv = os.path.splitext(save_path)[0] + "_uids.csv"
    with open(uid_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["FEWSNET_admin_code", "partition_id"])
        for uid, label in ordered:
            writer.writerow([uid, label])
    return save_path


def main() -> None:
    result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result_GeoRF"))
    vis_dir = os.path.join(result_dir, "vis")
    branches = parse_branch_files(vis_dir)
    correspondence_rows = read_final_correspondence(os.path.join(result_dir, "correspondence_table_Q4_2015.csv"))
    render_round_maps(result_dir, branches)
    render_final_map(result_dir, correspondence_rows)


if __name__ == "__main__":
    main()
