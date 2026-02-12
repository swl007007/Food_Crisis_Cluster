from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create linked tables from merged correspondence tables."
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Experiment directory (GeoRFExperiment, GeoXGBExperiment, GeoDTExperiment).",
    )
    return parser.parse_args()


def load_merged_correspondence_tables(
    experiment_dir: Path, pickle_file: str = "merged_correspondence_tables.pkl"
) -> list[dict]:
    """Load merged correspondence tables from pickle file."""
    pickle_path = experiment_dir / pickle_file

    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    print(f"Loading merged correspondence tables from {pickle_path}...")
    with open(pickle_path, "rb") as f:
        merged_tables = pickle.load(f)

    print(f"Loaded {len(merged_tables)} merged correspondence tables")
    return merged_tables


def create_main_index_table(merged_tables: list[dict]) -> pd.DataFrame:
    """
    Create main index table with metadata and metrics.

    Columns: name, variant, year, month, forecasting_scope, f1(1), f1_base(1)
    """
    print("\nCreating main index table...")

    main_data = []
    for item in merged_tables:
        df = item["dataframe"]

        # Extract f1 metrics (should be the same for all rows)
        f1_1 = df["f1(1)"].iloc[0]
        f1_base_1 = df["f1_base(1)"].iloc[0]

        main_data.append(
            {
                "name": item["name"],
                "variant": item["variant"],
                "year": item["year"],
                "month": item["month"],
                "forecasting_scope": item["forecasting_scope"],
                "f1(1)": f1_1,
                "f1_base(1)": f1_base_1,
            }
        )

    main_df = pd.DataFrame(main_data)
    print(f"Main table created with {len(main_df)} rows")
    print(f"Columns: {main_df.columns.tolist()}")

    return main_df


def preprocess_partition_table(
    df: pd.DataFrame,
    name: str,
    admin_code_min: int = 0,
    admin_code_max: int = 5717,
) -> pd.DataFrame | None:
    """
    Preprocess partition table to ensure complete FEWSNET_admin_code range.

    Args:
        df: Original correspondence dataframe
        name: Name identifier for the table
        admin_code_min: Minimum admin code (default: 0)
        admin_code_max: Maximum admin code (default: 5717)

    Returns:
        Preprocessed dataframe with complete admin code range
    """
    # Extract only FEWSNET_admin_code and partition_id columns
    if "FEWSNET_admin_code" not in df.columns or "partition_id" not in df.columns:
        print(f"  [WARNING] {name}: Missing required columns")
        print(f"    Available columns: {df.columns.tolist()}")
        return None

    partition_df = df[["FEWSNET_admin_code", "partition_id"]].copy()

    # Convert FEWSNET_admin_code to integer if possible
    try:
        partition_df["FEWSNET_admin_code"] = partition_df[
            "FEWSNET_admin_code"
        ].astype(int)
    except Exception as e:
        print(f"  [WARNING] {name}: Error converting admin_code to int: {e}")
        return None

    # Create complete range of admin codes
    complete_admin_codes = pd.DataFrame(
        {"FEWSNET_admin_code": range(admin_code_min, admin_code_max + 1)}
    )

    # Merge with existing data (left join to keep all admin codes)
    merged_df = complete_admin_codes.merge(
        partition_df, on="FEWSNET_admin_code", how="left"
    )

    # Fill missing partition_id with "s-1"
    merged_df["partition_id"] = merged_df["partition_id"].fillna("s-1")

    # Count missing values
    n_missing = (merged_df["partition_id"] == "s-1").sum()
    n_total = len(merged_df)
    n_present = n_total - n_missing

    print(
        f"  {name}: {n_present}/{n_total} admin codes have partition assignments, {n_missing} filled with 's-1'"
    )

    return merged_df


def create_partition_tables(
    merged_tables: list[dict],
    experiment_dir: Path,
    output_dir: str = "linked_tables/partitions",
) -> list[str]:
    """
    Create individual partition tables for each correspondence table.

    Args:
        merged_tables: List of merged correspondence table dictionaries
        experiment_dir: Experiment directory path
        output_dir: Directory to save partition tables (relative to experiment_dir)

    Returns:
        List of successfully created partition table filenames
    """
    print("\nCreating partition tables...")

    output_path = experiment_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    partition_files = []

    for item in merged_tables:
        name = item["name"]
        df = item["dataframe"]

        # Preprocess the partition table
        partition_df = preprocess_partition_table(df, name)

        if partition_df is not None:
            # Create filename based on name
            filename = f"{name}_partition.csv"
            filepath = output_path / filename

            # Save to CSV
            partition_df.to_csv(filepath, index=False)
            partition_files.append(filename)

    print(f"\nSuccessfully created {len(partition_files)} partition tables")
    print(f"Saved to: {output_path}")

    return partition_files


def save_main_table(
    main_df: pd.DataFrame,
    experiment_dir: Path,
    output_dir: str = "linked_tables",
    filename: str = "main_index.csv",
) -> Path:
    """Save main index table to CSV."""
    output_path = experiment_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename
    main_df.to_csv(filepath, index=False)

    print(f"\nMain index table saved to: {filepath}")
    return filepath


def create_link_mapping(
    main_df: pd.DataFrame,
    partition_files: list[str],
    experiment_dir: Path,
    output_dir: str = "linked_tables",
    filename: str = "table_links.csv",
) -> pd.DataFrame:
    """
    Create a mapping file that links main table names to partition table filenames.
    """
    output_path = experiment_dir / output_dir

    # Create mapping dataframe
    mapping_data = []
    for _, row in main_df.iterrows():
        name = row["name"]
        partition_filename = f"{name}_partition.csv"

        if partition_filename in partition_files:
            mapping_data.append(
                {
                    "name": name,
                    "partition_table_file": partition_filename,
                    "partition_table_path": f"partitions/{partition_filename}",
                }
            )

    mapping_df = pd.DataFrame(mapping_data)

    # Save mapping file
    filepath = output_path / filename
    mapping_df.to_csv(filepath, index=False)

    print(f"\nTable link mapping saved to: {filepath}")
    return mapping_df


def generate_summary_report(
    main_df: pd.DataFrame,
    partition_files: list[str],
    experiment_dir: Path,
    output_dir: str = "linked_tables",
) -> str:
    """Generate a summary report of the created linked tables."""
    output_path = experiment_dir / output_dir

    report = []
    report.append("=" * 80)
    report.append("LINKED TABLES CREATION SUMMARY")
    report.append("=" * 80)
    report.append("")

    # Main table summary
    report.append("1. MAIN INDEX TABLE")
    report.append("-" * 80)
    report.append(f"   File: main_index.csv")
    report.append(f"   Rows: {len(main_df)}")
    report.append(f"   Columns: {', '.join(main_df.columns.tolist())}")
    report.append("")
    report.append("   Summary by variant:")
    variant_counts = main_df["variant"].value_counts()
    for variant, count in variant_counts.items():
        report.append(f"     - {variant}: {count}")
    report.append("")
    report.append("   Summary by year:")
    year_counts = main_df["year"].value_counts().sort_index()
    for year, count in year_counts.items():
        report.append(f"     - {year}: {count}")
    report.append("")
    report.append("   Summary by forecasting scope:")
    fs_counts = main_df["forecasting_scope"].value_counts().sort_index()
    for fs, count in fs_counts.items():
        report.append(f"     - {fs}: {count}")
    report.append("")

    # Partition tables summary
    report.append("2. PARTITION TABLES")
    report.append("-" * 80)
    report.append(f"   Directory: partitions/")
    report.append(f"   Number of files: {len(partition_files)}")
    report.append(f"   Naming pattern: {{name}}_partition.csv")
    report.append(f"   Columns: FEWSNET_admin_code, partition_id")
    report.append(f"   Admin code range: 0 - 5717 (5718 total codes)")
    report.append(f"   Missing partition indicator: 's-1'")
    report.append("")

    # F1 score statistics
    report.append("3. F1 SCORE STATISTICS")
    report.append("-" * 80)
    report.append(f"   Mean f1(1): {main_df['f1(1)'].mean():.4f}")
    report.append(f"   Mean f1_base(1): {main_df['f1_base(1)'].mean():.4f}")
    report.append(
        f"   Mean improvement: {(main_df['f1(1)'] - main_df['f1_base(1)']).mean():.4f}"
    )
    report.append("")
    report.append("   By variant:")
    for variant in main_df["variant"].unique():
        subset = main_df[main_df["variant"] == variant]
        report.append(f"     {variant}:")
        report.append(f"       f1(1): {subset['f1(1)'].mean():.4f}")
        report.append(f"       f1_base(1): {subset['f1_base(1)'].mean():.4f}")
        report.append(
            f"       improvement: {(subset['f1(1)'] - subset['f1_base(1)']).mean():.4f}"
        )
    report.append("")

    # File structure
    report.append("4. FILE STRUCTURE")
    report.append("-" * 80)
    report.append("   linked_tables/")
    report.append("   |-- main_index.csv                    # Main index table")
    report.append("   |-- table_links.csv                   # Mapping between tables")
    report.append("   |-- summary_report.txt                # This report")
    report.append("   `-- partitions/")
    report.append("       |-- GeoRF_2021_02_fs1_partition.csv")
    report.append("       |-- GeoRF_2021_02_fs2_partition.csv")
    report.append("       `-- ... (70 partition tables)")
    report.append("")

    # Usage instructions
    report.append("5. USAGE INSTRUCTIONS")
    report.append("-" * 80)
    report.append("   To use the linked tables:")
    report.append("")
    report.append("   import pandas as pd")
    report.append("")
    report.append("   # Load main index table")
    report.append("   main_df = pd.read_csv('linked_tables/main_index.csv')")
    report.append("")
    report.append("   # Load specific partition table")
    report.append("   name = 'GeoRF_2021_02_fs1'")
    report.append(
        "   partition_df = pd.read_csv(f'linked_tables/partitions/{name}_partition.csv')"
    )
    report.append("")
    report.append("   # Or use the mapping file")
    report.append("   links = pd.read_csv('linked_tables/table_links.csv')")
    report.append(
        "   partition_file = links[links['name'] == name]['partition_table_path'].iloc[0]"
    )
    report.append("   partition_df = pd.read_csv(f'linked_tables/{partition_file}')")
    report.append("")

    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)
    report_path = output_path / "summary_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\nSummary report saved to: {report_path}")

    # Also print to console
    print("\n" + report_text)

    return report_text


def main() -> int:
    """Main function to create linked tables."""
    try:
        args = parse_args()
        experiment_dir = Path(args.experiment_dir).expanduser().resolve()

        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

        print("=" * 80)
        print("CREATING LINKED TABLES FROM MERGED CORRESPONDENCE TABLES")
        print("=" * 80)
        print(f"Experiment directory: {experiment_dir}")

        # Step 1: Load merged correspondence tables
        merged_tables = load_merged_correspondence_tables(experiment_dir)

        # Step 2: Create main index table
        main_df = create_main_index_table(merged_tables)

        # Step 3: Create partition tables
        partition_files = create_partition_tables(merged_tables, experiment_dir)

        # Step 4: Save main table
        save_main_table(main_df, experiment_dir)

        # Step 5: Create link mapping
        create_link_mapping(main_df, partition_files, experiment_dir)

        # Step 6: Generate summary report
        generate_summary_report(main_df, partition_files, experiment_dir)

        print("\n" + "=" * 80)
        print("LINKED TABLES CREATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nAll files saved to: linked_tables/")
        print("  - main_index.csv: Main table with metadata and metrics")
        print("  - table_links.csv: Mapping file")
        print("  - summary_report.txt: Detailed summary report")
        print("  - partitions/: Directory containing all partition tables")

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
