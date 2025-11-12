import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_stacked_area_by_quarter(
    year,
    quarter,
    num_samples_0,
    num_samples_1,
    *,
    normalize=False,
    title="Stacked area by quarter"
):
    """
    Plot a stacked area chart for two series over (year, quarter).

    Args:
        year: iterable of years (e.g., [2023, 2023, 2023, 2024, ...])
        quarter: iterable of quarter numbers (1..4)
        num_samples_0: iterable of values for series 0
        num_samples_1: iterable of values for series 1
        normalize: if True, stacks show shares (0..1) instead of counts
        title: chart title
    """
    # --- Basic validation ---
    n = {len(year), len(quarter), len(num_samples_0), len(num_samples_1)}
    if len(n) != 1:
        raise ValueError("All inputs must have the same length.")

    df = pd.DataFrame({
        "year": pd.to_numeric(year, errors="coerce").astype("Int64"),
        "quarter": pd.to_numeric(quarter, errors="coerce").astype("Int64"),
        "s0": pd.to_numeric(num_samples_0, errors="coerce"),
        "s1": pd.to_numeric(num_samples_1, errors="coerce"),
    })
    if df[["year", "quarter", "s0", "s1"]].isna().any().any():
        raise ValueError("Inputs contain NaNs or non-numeric values.")
    if not df["quarter"].between(1, 4).all():
        raise ValueError("Quarter values must be in 1..4.")

    # --- Build a PeriodIndex for quarters, aggregate & sort ---
    periods = pd.PeriodIndex(year=df["year"].astype(int),
                             quarter=df["quarter"].astype(int),
                             freq="Q")
    df = (
        df.assign(period=periods)
          .groupby("period", as_index=True)[["s0", "s1"]]
          .sum()
          .sort_index()
    )

    # --- Optional normalization to shares ---
    if normalize:
        totals = df.sum(axis=1).replace(0, pd.NA)
        df = (df.T / totals).T.fillna(0)

    # --- Plot ---
    x = df.index.to_timestamp("Q")  # quarter-end timestamps
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.stackplot(x, df["s0"], df["s1"], labels=["num_samples(0)", "num_samples(1)"])
    ax.set_title(title)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Share" if normalize else "Count")
    ax.legend(loc="upper left")
    ax.margins(x=0)

    # Clean, explicit quarter labels like 2024-Q3
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p.year}-Q{p.quarter}" for p in df.index], rotation=45, ha="right")

    plt.tight_layout()
    #save figure
    fig.savefig(r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\stacked_area_by_quarter.png')

def main():
    PATH = r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\deliverables\GeoRF_Deliverables\descriptive.csv'
    #read descriptive stats
    descriptive_stats = pd.read_csv(PATH)
    plot_stacked_area_by_quarter(
        year = descriptive_stats['year'],
        quarter = descriptive_stats['quarter'],
        num_samples_0 = descriptive_stats['num_samples(0)'],
        num_samples_1 = descriptive_stats['num_samples(1)'],
        title="Stacked plot for Crisis counts in each quarter"
        )

if __name__ == "__main__":
    main()