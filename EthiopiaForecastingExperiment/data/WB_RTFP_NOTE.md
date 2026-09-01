# WB RTFP data note

- Source snapshot: `WLD_RTFP_mkt_2026-04-20.csv`
- Source SHA-256: `cd81efe9ec6c3c1ab7aa14d1a1dbd6092606c015608cec354beccfd960192730`
- On 2026-09-01, the user reported manually checking the source and consulting
  Dr Bo, who confirmed that this snapshot remained the latest available data.
- The file has observation months but no separate publication-time or historical
  vintage fields. Results therefore use the confirmed latest snapshot and should
  retain this provenance note when interpreted.
- Ethiopia admins are linked to the nearest Ethiopia entity market; the provided
  `Market Average` row is excluded. No distance cutoff is imposed, and the
  nearest-market distance is retained for audit and later sensitivity analysis.
- The base index is `(o_food_price_index + c_food_price_index) / 2`. `lag1` is
  the previous market month; `MA4` is the complete mean of months `t-1` through
  `t-4`. These features are built per market before forecast-horizon alignment.
