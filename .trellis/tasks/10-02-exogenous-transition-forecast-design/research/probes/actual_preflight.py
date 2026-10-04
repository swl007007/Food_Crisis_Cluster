"""Read-only scen-actual preflight (no fit, no truth/expert read, no RUN write).

Checks the actual-CS availability table against the code contract and the prepared inputs:
schema/uniqueness/evidence/source; cohort-country coverage (Availability.countries) via the real
``actual_gate_intensity`` refusal path for each actual origin; consistency with the prepared release
ledger (no CS cycle after 2024-10; each country's last released cycle); the source month and age of
the latest lawful CS label per country at each origin; extension manifest v2 loading through
``load_extension``/Scaffold up to the static month O; the frozen recipe; fresh actual output; code
identity. Usage: python actual_preflight.py RUN_DIR TABLE_CSV MANIFEST_JSON
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[5] / "FEWSNETGeoXGBExperiment"))
import pandas as pd  # noqa: E402

from scripts import run_experiment as rx  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402
from src.utils.run_identity import file_sha256  # noqa: E402

run, table_path, manifest = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
mi = lambda s: int(s[:4]) * 12 + int(s[5:7]) - 1   # noqa: E731
lab = lambda x: f"{x // 12}-{x % 12 + 1:02d}"   # noqa: E731
out, problems = {"table": str(table_path), "table_sha256": file_sha256(table_path),
                 "manifest": str(manifest), "manifest_sha256": file_sha256(manifest)}, []
table = pd.read_csv(table_path, dtype={"origin_month": str})
out["rows"] = len(table)
if list(table.columns) != list(rx.ACTUAL_COLUMNS): problems.append(f"columns {list(table.columns)}")
if table.duplicated(["country", "product", "origin_month"]).any(): problems.append("duplicate country/product/origin")
if (table["evidence"] != "reconstructed").any() or table["source"].isna().any(): problems.append("evidence/source")
origins = sorted({rx.s3.ml(rx.s3.mi(t) - h) for t, h in rx.ACTUAL_CASES})
out["actual_cases"] = [list(c) for c in rx.ACTUAL_CASES]
out["origins_needed"] = origins
out["table_counts"] = {o: table[table.origin_month == o].groupby("missed_cycles").country.apply(sorted).to_dict()
                       for o in origins}
# real contract: cohort countries and the code's own refusal path
frozen = rx._accept_frozen(run)
out["frozen_recipe"] = {h: {k: v for k, v in e.items() if k in ("released", "strategy", "map_id")}
                        for h, e in frozen["recipe"].items()}
ctx = rx.scenario_context(run, 4, development_truth=False, extension=manifest)
out["cohort_countries"] = list(ctx.countries)
out["gate_intensity"] = {}
for o in origins:
    try:
        out["gate_intensity"][o] = rx.actual_gate_intensity(table, o, ctx.countries)
    except SystemExit as e:
        out["gate_intensity"][o] = f"REFUSED: {e}"
        problems.append(f"{o}: {e}")
# ledger consistency and latest lawful CS label age per country
led = pd.read_csv(run / "prepared/manifests/release_ledger.csv", dtype=str)
led = led[led["product"] == "CS"].assign(month=lambda d: d.reference_month.map(mi), release=lambda d: d.release_date.map(mi))
out["ledger_last_cycle"] = led.groupby("country").month.max().map(lab).to_dict()
out["ledger_cycles_after_2024_10"] = int((led.month > mi("2024-10")).sum())
obs = pd.read_csv(run / "prepared/ledgers/observations.csv", usecols=["area", "month", "country"])
obs = obs.merge(led[["country", "month", "release"]], on=["country", "month"], how="left")
ages = {}
for o in origins:
    vis = obs[(obs.release <= mi(o)) & (obs.month <= mi(o))]
    last = vis.groupby("country").month.max()
    ages[o] = {c: {"latest_cs": lab(int(m)), "age_months": mi(o) - int(m)} for c, m in last.items()}
out["latest_lawful_cs_by_origin"] = ages
# extension through the static month of the latest origin
sc = ctx.scaffold
out["scaffold"] = [int(sc.areas.size), lab(sc.first_month), lab(sc.first_month + sc.n_months - 1)]
if lab(sc.first_month + sc.n_months - 1) < max(origins): problems.append("scaffold ends before the latest origin")
out["scenario_actual_exists"] = (run / "scenario_actual").exists()
if out["scenario_actual_exists"]: problems.append("scenario_actual already exists (not fresh)")
out["code_identity"] = rid.code_identity()
sel = json.loads((run / "scenario_development/selection.json").read_text(encoding="utf-8"))
if out["code_identity"] != sel["code"]: problems.append("code identity differs from the selection")
out["problems"] = problems
HERE.with_name("actual_preflight_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k not in ("latest_lawful_cs_by_origin",)}, indent=1, ensure_ascii=False))
