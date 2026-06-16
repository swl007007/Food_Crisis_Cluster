"""Run Stage 3 fixed-partition GeoRF ablations for feature-exclude datasets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_SOURCE_ROOT = (
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data"
)
POSIX_SOURCE_ROOT = (
    "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data"
)
WINDOWS_POLYGONS_PATH = (
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data"
    r"\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp"
)
POSIX_POLYGONS_PATH = (
    "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data"
    "/Outcome/FEWSNET_IPC/FEWS NET Admin Boundaries/FEWS_Admin_LZ_v3.shp"
)

RUN_ROOT = REPO_ROOT / "main_ablation_exclude_updated_stage3_fixed_partitions"
DEFAULT_DATA_ROOT = RUN_ROOT / "input_datasets"
MANIFEST_PATH = REPO_ROOT / "GeoRFExperiment" / "knn_sparsification_results" / "cluster_mapping_manifest.json"
ADJACENCY_CACHE = REPO_ROOT / "src" / "adjacency" / "polygon_adjacency_cache.pkl"
REFINE_SCRIPT = REPO_ROOT / "scripts" / "refine_partitions_contiguity.py"
COMPARISON_SCRIPT = REPO_ROOT / "scripts" / "compare_partitioned_vs_pooled_rf_k40_nc4.py"

FEATURE_DATASETS = {
    "weather_exclude": "weather_exclude.csv",
    "agri_exclude": "agri_exclude.csv",
    "conflict_exclude": "conflict_exclude.csv",
    "econ_exclude": "econ_exclude.csv",
    "food_prices_exclude": "food_prices_exclude.csv",
    "geographic_exclude": "geographic_exclude.csv",
    "secondary_exclude": "secondary_exclude.csv",
    "lag_exclude": "lag_exclude.csv",
}

FEATURE_ENV_OVERRIDES = {
    "lag_exclude": {
        "ENABLE_LAG_FEATURES": "false",
    },
}

SCOPES = (1, 2, 3)
REFINE_ITERS = 3
START_MONTH = "2021-01"
END_MONTH = "2024-12"
TRAIN_WINDOW = 36


def source_root(data_root: Path | None = None) -> Path:
    if data_root is not None:
        return data_root
    if DEFAULT_DATA_ROOT.exists():
        return DEFAULT_DATA_ROOT
    return Path(WINDOWS_SOURCE_ROOT if os.name == "nt" else POSIX_SOURCE_ROOT)


def polygons_path() -> Path:
    return Path(WINDOWS_POLYGONS_PATH if os.name == "nt" else POSIX_POLYGONS_PATH)


def windows_path_to_current_os(value: str) -> Path:
    """Convert manifest Windows paths for WSL dry-runs; keep native Windows paths on Windows."""
    if os.name == "nt":
        return Path(value)
    normalized = value.replace("\\", "/")
    if len(normalized) >= 3 and normalized[1:3] == ":/":
        drive = normalized[0].lower()
        return Path("/mnt") / drive / normalized[3:]
    return Path(normalized)


def load_partition_maps() -> dict[str, Path]:
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)

    keys = {"general": "general", "m02": "m2", "m06": "m6", "m10": "m10"}
    maps: dict[str, Path] = {}
    for manifest_key, output_key in keys.items():
        entry = manifest.get(manifest_key)
        if not isinstance(entry, dict) or not entry.get("path"):
            raise ValueError(f"Missing partition map '{manifest_key}' in {MANIFEST_PATH}")
        path = windows_path_to_current_os(entry["path"])
        if not path.exists():
            raise FileNotFoundError(f"Partition map not found: {path}")
        maps[output_key] = path
    return maps


def refined_path(input_path: Path, refined_dir: Path) -> Path:
    return refined_dir / f"{input_path.stem}_refined_contig{REFINE_ITERS}.csv"


def command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def scan_log(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    ignored_warning_fragments = (
        "polygons have invalid partition assignments (will be skipped)",
    )
    hits = []
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            upper = line.upper()
            if any(fragment in line for fragment in ignored_warning_fragments):
                continue
            if "WARNING" in upper or "ERROR" in upper or "TRACEBACK" in upper:
                hits.append(line.rstrip())
    return hits


def print_log_tail(log_path: Path, n_lines: int = 80) -> None:
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        return
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    for line in lines[-n_lines:]:
        print(line.rstrip())


def run_logged_command(
    command: list[str],
    dry_run: bool,
    log_path: Path,
    env: dict[str, str] | None = None,
    env_overrides: dict[str, str] | None = None,
) -> None:
    if dry_run:
        print(command_text(command))
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(command_text(command) + "\n\n")
        if env_overrides:
            log_handle.write("Environment overrides:\n")
            for key in sorted(env_overrides):
                log_handle.write(f"{key}={env_overrides[key]}\n")
            log_handle.write("\n")
        log_handle.flush()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
    warnings = scan_log(log_path)
    if warnings:
        print(f"WARNING/ERROR lines in {log_path}:")
        for line in warnings[:8]:
            print(f"  {line}")
        if len(warnings) > 8:
            print(f"  ... {len(warnings) - 8} more warning/error lines in log")
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}: {command_text(command)}")
        print(f"Log tail: {log_path}")
        print_log_tail(log_path)
        raise subprocess.CalledProcessError(result.returncode, command)


def select_groups(group: str | None, force_value: str | None) -> list[str]:
    selected = list(FEATURE_DATASETS)
    if group:
        selected = [group]
    if force_value and force_value != "__all__":
        selected = [force_value]
    for item in selected:
        if item not in FEATURE_DATASETS:
            raise ValueError(f"Unknown feature group: {item}")
    return selected


def select_scopes(scope: int | None) -> tuple[int, ...]:
    if scope is None:
        return SCOPES
    if scope not in SCOPES:
        raise ValueError(f"--scope must be one of {SCOPES}; got {scope}")
    return (scope,)


def refine_maps(base_maps: dict[str, Path], refined_dir: Path, dry_run: bool) -> dict[str, Path]:
    if not dry_run:
        refined_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for key in ("general", "m2", "m6", "m10"):
        source = base_maps[key]
        target = refined_path(source, refined_dir)
        outputs[key] = target
        if target.exists():
            print(f"SKIP refine existing: {target}")
            continue
        command = [
            sys.executable,
            str(REFINE_SCRIPT),
            "--adj",
            str(ADJACENCY_CACHE),
            "--in",
            str(source),
            "--out",
            str(refined_dir),
            "--iters",
            str(REFINE_ITERS),
        ]
        run_logged_command(command, dry_run, refined_dir / "command_logs" / f"refine_{key}.log")
    return outputs


def comparison_command(
    dataset_path: Path,
    out_dir: Path,
    scope: int,
    refined_maps: dict[str, Path],
) -> list[str]:
    return [
        sys.executable,
        str(COMPARISON_SCRIPT),
        "--data",
        str(dataset_path),
        "--partition-map",
        str(refined_maps["general"]),
        "--partition-map-m2",
        str(refined_maps["m2"]),
        "--partition-map-m6",
        str(refined_maps["m6"]),
        "--partition-map-m10",
        str(refined_maps["m10"]),
        "--polygons",
        str(polygons_path()),
        "--out-dir",
        str(out_dir),
        "--start-month",
        START_MONTH,
        "--end-month",
        END_MONTH,
        "--train-window",
        str(TRAIN_WINDOW),
        "--forecasting-scope",
        str(scope),
        "--month-ind",
    ]


def run_combo(
    group: str,
    scope: int,
    base_maps: dict[str, Path],
    data_root: Path,
    dry_run: bool,
    force: bool,
) -> dict[str, str | int | bool]:
    dataset_path = source_root(data_root) / FEATURE_DATASETS[group]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    out_dir = RUN_ROOT / group / f"result_partition_k40_compare_GF_fs{scope}"
    log_dir = out_dir / "command_logs"
    metrics_path = out_dir / "metrics_monthly.csv"
    if force and out_dir.exists() and not dry_run:
        for filename in (
            "metrics_monthly.csv",
            "predictions_monthly.csv",
            "metrics_polygon_overall.csv",
            "run_manifest.json",
        ):
            path = out_dir / filename
            if path.exists():
                path.unlink()
    if metrics_path.exists() and not force:
        print(f"SKIP existing: {metrics_path}")
        return {
            "group": group,
            "scope": scope,
            "status": "skipped_existing",
            "out_dir": str(out_dir),
        }

    refined_maps = refine_maps(base_maps, out_dir / "refined", dry_run)
    env = os.environ.copy()
    env_overrides = {
        "NO_LEAK_PARTITION_LEARNING_YEARS": "2018-2020",
        "NO_LEAK_EVALUATION_YEARS": "2021-2024",
    }
    env_overrides.update(FEATURE_ENV_OVERRIDES.get(group, {}))
    env.update(env_overrides)
    command = comparison_command(dataset_path, out_dir, scope, refined_maps)

    print("=" * 100)
    print(f"RUN group={group} scope={scope} out={out_dir}")
    if env_overrides:
        print(f"Env overrides: {env_overrides}")
    if dry_run:
        print(command_text(command))
    else:
        run_logged_command(
            command,
            dry_run,
            log_dir / "compare.log",
            env=env,
            env_overrides=env_overrides,
        )

    return {
        "group": group,
        "scope": scope,
        "status": "dry_run" if dry_run else "completed",
        "dataset": str(dataset_path),
        "out_dir": str(out_dir),
        "env_overrides": env_overrides,
    }


def write_manifest(records: Iterable[dict[str, str | int | bool]], dry_run: bool) -> None:
    if dry_run:
        return
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = RUN_ROOT / "ablation_run_manifest.json"
    existing_records: list[dict[str, str | int | bool]] = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        existing_records = existing_manifest.get("records", [])

    merged_records = {
        (record.get("group"), record.get("scope")): record
        for record in existing_records
    }
    for record in records:
        merged_records[(record.get("group"), record.get("scope"))] = record

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "stage": "Stage 3 fixed-partition feature-exclude ablation",
        "base_model": "GeoRF/GF",
        "start_month": START_MONTH,
        "end_month": END_MONTH,
        "train_window": TRAIN_WINDOW,
        "month_ind": True,
        "contiguity_refinement": f"cont{REFINE_ITERS}",
        "records": list(merged_records.values()),
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved manifest: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(FEATURE_DATASETS), help="Run one feature group.")
    parser.add_argument("--scope", type=int, choices=SCOPES, help="Run one forecasting scope.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--force",
        nargs="?",
        const="__all__",
        help="Re-run outputs. Optional value can name a single feature group.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = select_groups(args.group, args.force)
    scopes = select_scopes(args.scope)
    force = args.force is not None
    base_maps = load_partition_maps()
    data_root = args.data_root

    print(f"Run root: {RUN_ROOT}")
    print(f"Data root: {data_root}")
    print(f"Groups: {groups}")
    print(f"Scopes: {scopes}")
    print(f"Force: {force}")
    print(f"Dry-run: {args.dry_run}")

    records = []
    for group in groups:
        for scope in scopes:
            records.append(run_combo(group, scope, base_maps, data_root, args.dry_run, force))
    write_manifest(records, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
