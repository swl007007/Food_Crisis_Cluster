# Covariate-only probe: which rolling/z-score variant reproduces stored Tair_zscore/Rainf_zscore.
# Reads ONLY admin/date keys + Tair/Rainf raw + stored zscores. No IPC columns are read.
import sys, numpy as np, pandas as pd
SRC = "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data"
files = {"pinned": f"{SRC}/FEWSNET_forecast_unadjusted_bm.csv",
         "combined": f"{SRC}/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025_combined.csv",
         "normv1": f"{SRC}/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025_combined.normalized-v1.csv"}
cols = ["FEWSNET_admin_code", "date", "Tair_f_tavg_mean", "Rainf_f_tavg_mean", "Tair_zscore", "Rainf_zscore"]
which = sys.argv[1:] or list(files)
for name in which:
    df = pd.read_csv(files[name], usecols=cols, parse_dates=["date"])
    df = df.reset_index().rename(columns={"index": "srow"})
    df = df.sort_values(["FEWSNET_admin_code", "date", "srow"], kind="stable").reset_index(drop=True)
    g = df.groupby("FEWSNET_admin_code", sort=False)
    first_k = g.cumcount() < 11  # first 11 rows of each admin = boundary-affected rows for a global window
    print(f"## {name}: rows={len(df)} admins={df.FEWSNET_admin_code.nunique()} dates={df.date.min().date()}..{df.date.max().date()}")
    for v, z in [("Tair_f_tavg_mean", "Tair_zscore"), ("Rainf_f_tavg_mean", "Rainf_zscore")]:
        x = df[v]; stored = df[z]
        m12_global = x.rolling(12, min_periods=1).mean()
        m12_admin = g[v].transform(lambda s: s.rolling(12, min_periods=1).mean())
        cands = {}
        for tag, m12 in [("globalroll", m12_global), ("adminroll", m12_admin)]:
            gm = m12.groupby(df.FEWSNET_admin_code).transform("mean")
            gs = m12.groupby(df.FEWSNET_admin_code).transform("std")  # ddof=1
            cands[f"{tag}_then_admin_fullsample_z"] = (m12 - gm) / gs
        # rolling-window z-score variants (per admin)
        rm = g[v].transform(lambda s: s.rolling(12, min_periods=1).mean())
        rs = g[v].transform(lambda s: s.rolling(12, min_periods=1).std())
        cands["admin_rolling12_z_incl_current"] = (x - rm) / rs
        rmp = g[v].transform(lambda s: s.shift(1).rolling(12, min_periods=1).mean())
        rsp = g[v].transform(lambda s: s.shift(1).rolling(12, min_periods=1).std())
        cands["admin_rolling12_z_prior_only"] = (x - rmp) / rsp
        grm = x.rolling(12, min_periods=1).mean(); grs = x.rolling(12, min_periods=1).std()
        cands["global_rolling12_z_incl_current"] = (x - grm) / grs
        ok = stored.notna()
        for k, c in cands.items():
            d = (c - stored).abs()
            both = ok & c.notna()
            match = (d[both] < 1e-6).mean()
            mb = (d[both & first_k] < 1e-6).mean()
            mi = (d[both & ~first_k] < 1e-6).mean()
            print(f"{z:13s} {k:38s} n={both.sum():8d} match={match:.4f} maxabs={d[both].max():.3g} "
                  f"match_first11={mb:.4f} match_interior={mi:.4f}")
        print(f"{z} stored nonnull={ok.sum()} null={(~ok).sum()}")
    sys.stdout.flush()
