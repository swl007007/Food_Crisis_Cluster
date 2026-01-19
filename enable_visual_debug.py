#!/usr/bin/env python3
"""
Helper script to enable visual debug settings for GeoRF and XGBoost batch runs.

This script automatically modifies:
1. config.py: VIS_DEBUG_MODE = True
2. app/main_model_GF.py: track_partition_metrics = True, enable_metrics_maps = True
3. app/main_model_XGB.py: track_partition_metrics = True, enable_metrics_maps = True

Usage:
    python enable_visual_debug.py          # Enable visual debug
    python enable_visual_debug.py disable  # Disable visual debug
"""

import os
import sys
import re
from pathlib import Path


def modify_file_setting(file_path, pattern, replacement, description):
    """
    Modify a specific setting in a file using regex.

    Args:
        file_path: Path to the file to modify
        pattern: Regex pattern to match the line to replace
        replacement: Replacement string
        description: Description of the change for logging

    Returns:
        bool: True if modification was successful
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if pattern exists
        if not re.search(pattern, content):
            print(f"WARNING: Pattern not found in {file_path}")
            print(f"  Looking for: {pattern}")
            return False

        # Replace pattern
        new_content, count = re.subn(pattern, replacement, content)

        if count == 0:
            print(f"WARNING: No replacements made in {file_path}")
            return False

        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✓ {description}")
        print(f"  File: {file_path}")
        print(f"  Changes: {count} replacement(s) made")
        return True

    except Exception as e:
        print(f"ERROR modifying {file_path}: {e}")
        return False


def enable_visual_debug():
    """Enable all visual debug settings."""
    print("=" * 70)
    print("ENABLING VISUAL DEBUG MODE")
    print("=" * 70)
    print()

    success_count = 0
    total_changes = 3

    # 1. Enable VIS_DEBUG_MODE in config.py
    print("[1/3] Enabling VIS_DEBUG_MODE in config.py...")
    if modify_file_setting(
        'config.py',
        r'VIS_DEBUG_MODE\s*=\s*False',
        'VIS_DEBUG_MODE = True',
        'Set VIS_DEBUG_MODE = True'
    ):
        success_count += 1
    print()

    # 2. Enable partition metrics in main_model_GF.py
    print("[2/3] Enabling partition metrics tracking in app/main_model_GF.py...")
    gf_changes = 0

    # Enable track_partition_metrics
    if modify_file_setting(
        'app/main_model_GF.py',
        r'track_partition_metrics\s*=\s*False\s*#',
        'track_partition_metrics = True  #',
        'Set track_partition_metrics = True in main_model_GF.py'
    ):
        gf_changes += 1

    # Enable enable_metrics_maps
    if modify_file_setting(
        'app/main_model_GF.py',
        r'enable_metrics_maps\s*=\s*False\s*#',
        'enable_metrics_maps = True      #',
        'Set enable_metrics_maps = True in main_model_GF.py'
    ):
        gf_changes += 1

    if gf_changes == 2:
        success_count += 1
    print()

    # 3. Enable partition metrics in main_model_XGB.py
    print("[3/3] Enabling partition metrics tracking in app/main_model_XGB.py...")
    xgb_changes = 0

    # Enable track_partition_metrics
    if modify_file_setting(
        'app/main_model_XGB.py',
        r'track_partition_metrics\s*=\s*False\s*#',
        'track_partition_metrics = True  #',
        'Set track_partition_metrics = True in main_model_XGB.py'
    ):
        xgb_changes += 1

    # Enable enable_metrics_maps
    if modify_file_setting(
        'app/main_model_XGB.py',
        r'enable_metrics_maps\s*=\s*False\s*#',
        'enable_metrics_maps = True      #',
        'Set enable_metrics_maps = True in main_model_XGB.py'
    ):
        xgb_changes += 1

    if xgb_changes == 2:
        success_count += 1
    print()

    # Summary
    print("=" * 70)
    if success_count == total_changes:
        print("✓ SUCCESS: All visual debug settings enabled!")
        print()
        print("Visual outputs will include:")
        print("  - Partition maps (result_Geo*/vis/partition_*.png)")
        print("  - Performance grids (result_Geo*/vis/performance_*.png)")
        print("  - Metrics tracking CSV (result_Geo*/partition_metrics/*.csv)")
        print("  - Improvement maps (result_Geo*/partition_metrics/*_improvement.png)")
        print()
        print("You can now run:")
        print("  - run_georf_batches_2021_2024_visual.bat")
        print("  - run_xgboost_batches_2021_2024_visual.bat")
    else:
        print(f"⚠ WARNING: Only {success_count}/{total_changes} changes completed successfully")
        print("Please check the error messages above and manually verify the settings.")
    print("=" * 70)


def disable_visual_debug():
    """Disable all visual debug settings (restore to production mode)."""
    print("=" * 70)
    print("DISABLING VISUAL DEBUG MODE (Restoring Production Settings)")
    print("=" * 70)
    print()

    success_count = 0
    total_changes = 3

    # 1. Disable VIS_DEBUG_MODE in config.py
    print("[1/3] Disabling VIS_DEBUG_MODE in config.py...")
    if modify_file_setting(
        'config.py',
        r'VIS_DEBUG_MODE\s*=\s*True',
        'VIS_DEBUG_MODE = False',
        'Set VIS_DEBUG_MODE = False'
    ):
        success_count += 1
    print()

    # 2. Disable partition metrics in main_model_GF.py
    print("[2/3] Disabling partition metrics tracking in app/main_model_GF.py...")
    gf_changes = 0

    # Disable track_partition_metrics
    if modify_file_setting(
        'app/main_model_GF.py',
        r'track_partition_metrics\s*=\s*True\s*#',
        'track_partition_metrics = False #',
        'Set track_partition_metrics = False in main_model_GF.py'
    ):
        gf_changes += 1

    # Disable enable_metrics_maps
    if modify_file_setting(
        'app/main_model_GF.py',
        r'enable_metrics_maps\s*=\s*True\s*#',
        'enable_metrics_maps = False     #',
        'Set enable_metrics_maps = False in main_model_GF.py'
    ):
        gf_changes += 1

    if gf_changes == 2:
        success_count += 1
    print()

    # 3. Disable partition metrics in main_model_XGB.py
    print("[3/3] Disabling partition metrics tracking in app/main_model_XGB.py...")
    xgb_changes = 0

    # Disable track_partition_metrics
    if modify_file_setting(
        'app/main_model_XGB.py',
        r'track_partition_metrics\s*=\s*True\s*#',
        'track_partition_metrics = False #',
        'Set track_partition_metrics = False in main_model_XGB.py'
    ):
        xgb_changes += 1

    # Disable enable_metrics_maps
    if modify_file_setting(
        'app/main_model_XGB.py',
        r'enable_metrics_maps\s*=\s*True\s*#',
        'enable_metrics_maps = False     #',
        'Set enable_metrics_maps = False in main_model_XGB.py'
    ):
        xgb_changes += 1

    if xgb_changes == 2:
        success_count += 1
    print()

    # Summary
    print("=" * 70)
    if success_count == total_changes:
        print("✓ SUCCESS: All visual debug settings disabled!")
        print()
        print("Production mode restored:")
        print("  - No visualization overhead")
        print("  - Faster execution")
        print("  - Minimal debug output")
        print()
        print("You can now run:")
        print("  - run_georf_batches.bat (full 36-batch production run)")
        print("  - run_xgboost_batches.bat (full 36-batch production run)")
    else:
        print(f"⚠ WARNING: Only {success_count}/{total_changes} changes completed successfully")
        print("Please check the error messages above and manually verify the settings.")
    print("=" * 70)


def main():
    """Main entry point."""
    # Check if we're in the correct directory
    if not os.path.exists('config.py'):
        print("ERROR: config.py not found in current directory")
        print("Please run this script from the Food_Crisis_Cluster directory")
        sys.exit(1)

    if not os.path.exists('app/main_model_GF.py'):
        print("ERROR: app/main_model_GF.py not found")
        print("Please run this script from the Food_Crisis_Cluster directory")
        sys.exit(1)

    # Check command line argument
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['disable', 'off', 'false']:
        disable_visual_debug()
    else:
        enable_visual_debug()


if __name__ == '__main__':
    main()
