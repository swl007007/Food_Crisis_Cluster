"""Utility script to rename XGB result CSVs with an explicit prefix."""
from pathlib import Path, PureWindowsPath
import os
import sys

# Windows-style directory containing the XGB result CSVs.
WINDOWS_TARGET_DIR = PureWindowsPath(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\xgb"
)

# Filename prefix to inject into matching CSVs.
SOURCE_PREFIX = "results_df_gp_fs"
TARGET_PREFIX = "results_df_xgb_gp_fs"
PATTERN = f"{SOURCE_PREFIX}*_*_*.csv"


def resolve_target_dir() -> Path:
    """Resolve the target directory across Windows and WSL environments."""

    direct_path = Path(str(WINDOWS_TARGET_DIR))
    if direct_path.exists():
        return direct_path

    drive = WINDOWS_TARGET_DIR.drive.rstrip(":").lower()
    if drive:
        wsl_path = Path("/mnt") / drive / Path(*WINDOWS_TARGET_DIR.parts[1:])
        if wsl_path.exists():
            return wsl_path

    raise FileNotFoundError(f"Directory not found (Windows/WSL): {WINDOWS_TARGET_DIR}")


def main() -> None:
    target_dir = resolve_target_dir()

    os.chdir(target_dir)
    renamed_any = False

    for csv_path in Path(".").glob(PATTERN):
        target_name = csv_path.name.replace(SOURCE_PREFIX, TARGET_PREFIX, 1)
        target_path = csv_path.with_name(target_name)

        if target_path.exists():
            print(f"Skipping {csv_path.name}: {target_name} already exists", file=sys.stderr)
            continue

        csv_path.rename(target_path)
        print(f"Renamed {csv_path.name} -> {target_name}")
        renamed_any = True

    if not renamed_any:
        print("No files matched the pattern; nothing renamed.")


if __name__ == "__main__":
    main()
