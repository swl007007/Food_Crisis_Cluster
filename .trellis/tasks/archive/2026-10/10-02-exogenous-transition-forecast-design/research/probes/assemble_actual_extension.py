"""Assemble the explicit two-source covariate extension for the actual-2025 cases (no fit).

Splice: the 2024-12 overlap block from the PINNED panel plus the 2025-01..06 rows from the COMBINED
panel. Columns are exactly FEWSNET_admin_code, date and the 69 schema covariate sources. Value
fields are copied as the original source strings (stdlib csv: no numeric rewrite or rounding); only
the date key is written as YYYY-MM (combined stamps are YYYY-MM-DD; load_extension reads str[:7]).
No patching, imputation or dropped rows. Raw files are hashed before and after. Refuses to
overwrite. Usage: python assemble_actual_extension.py OUT_DIR
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PKG = HERE.parents[5] / "FEWSNETGeoXGBExperiment"
SRC = HERE.parents[8] / "1.Source Data"
PINNED = SRC / "FEWSNET_forecast_unadjusted_bm.csv"
COMBINED = SRC / "assembled_FEWSNET" / "FEWSNET_forecast_unadjusted_bm_2025_combined.csv"
PINNED_SHA = "611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651"
OVERLAP, LATER = ("2024-12",), ("2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


out_dir = Path(sys.argv[1])
if out_dir.exists():
    raise SystemExit(f"{out_dir} exists; refusing to overwrite")
schema = json.loads((PKG / "feature-schema.json").read_text(encoding="utf-8"))
sources = list(schema["static_sources"] + schema["dynamic_sources_at_origin"])
assert len(sources) == 69
before = {"pinned": sha(PINNED), "combined": sha(COMBINED)}
if before["pinned"] != PINNED_SHA:
    raise SystemExit("pinned panel bytes differ from the prepared sources.json pin")

columns = ["FEWSNET_admin_code", "date"] + sources
rows, counts = [], {}
for path, months, label in ((PINNED, OVERLAP, "pinned"), (COMBINED, LATER, "combined")):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in columns if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{label} lacks {missing}")
        for d in reader:
            ym = d["date"][:7]
            if ym in months:
                rows.append([d["FEWSNET_admin_code"], ym] + [d[s] for s in sources])
                counts[(label, ym)] = counts.get((label, ym), 0) + 1
keys = [(r[0], r[1]) for r in rows]
if len(set(keys)) != len(keys):
    raise SystemExit("duplicate (area, month) keys in the assembled rows")
out_dir.mkdir(parents=True)
target = out_dir / "covariate_extension_2024-12_2025-06.csv"
with open(target, "x", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(columns)
    w.writerows(rows)
after = {"pinned": sha(PINNED), "combined": sha(COMBINED)}
if after != before:
    raise SystemExit("raw source bytes changed during assembly")
report = {"output": str(target), "output_sha256": sha(target), "rows": len(rows), "unique_keys": len(set(keys)),
          "per_component_month": {f"{k[0]}:{k[1]}": v for k, v in sorted(counts.items())},
          "raw_sha256_before": before, "raw_sha256_after": after,
          "raw_paths": {"pinned": str(PINNED), "combined": str(COMBINED)}, "columns": columns}
(out_dir / "assembly_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "columns"}, indent=1))
