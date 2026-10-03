"""Launch prep: build the CS release ledger (availability.LEDGER_COLUMNS) from a USER-DECIDED rule.

There is deliberately no default rule: the historical IPC release-date convention is a scientific
alignment decision (coordinator/user). Rows = every (ISO country, reference month) with any
non-null historical CS label in the pinned panel (presence only; the pinned panel ends 2024-12, so
no 2025 value is read). Country codes equal the observations' ``country`` (panel ISO).

python build_release_ledger.py --panel PANEL.csv --rule reference_month_end --source "..." --out LEDGER.csv
python build_release_ledger.py --panel PANEL.csv --rule following_month_end --source "..." --dry-run
"""
import argparse
import sys
import pandas as pd

RULES = {
    "reference_month_end": 0,    # released by the last day of the reference month M
    "following_month_end": 1,    # released by the last day of month M+1
}

p = argparse.ArgumentParser()
p.add_argument("--panel", required=True)
p.add_argument("--rule", required=True, choices=sorted(RULES))
p.add_argument("--source", required=True, help="citation of the decided convention (recorded per row)")
p.add_argument("--out")
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()
if not a.dry_run and not a.out:
    sys.exit("--out is required unless --dry-run")
panel = pd.read_csv(a.panel, usecols=["FEWSNET_admin_code", "date", "ISO", "fews_ipc"])
cells = panel.loc[panel["fews_ipc"].notna(), ["ISO", "date"]].drop_duplicates()
cells["ref"] = pd.PeriodIndex(cells["date"].astype(str).str[:7], freq="M")
assert cells["ref"].max() <= pd.Period("2024-12", "M"), "post-2024 label month in the pinned panel"
offset = RULES[a.rule]
ledger = pd.DataFrame({
    "cycle_id": "CS-" + cells["ref"].astype(str), "product": "CS", "country": cells["ISO"].astype(str),
    "reference_month": cells["ref"].astype(str),
    "release_date": (cells["ref"] + offset).dt.end_time.dt.strftime("%Y-%m-%d"),
    "evidence": "reconstructed", "source": f"{a.rule}: {a.source}"}).sort_values(["reference_month", "country"])
print(f"rule={a.rule} rows={len(ledger)} countries={ledger['country'].nunique()} cycles={ledger['reference_month'].nunique()} "
      f"first={ledger['reference_month'].min()} last={ledger['reference_month'].max()}")
if not a.dry_run:
    ledger.to_csv(a.out, index=False)
