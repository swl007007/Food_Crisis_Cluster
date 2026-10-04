"""Verify the spliced actual-2025 covariate extension, write its manifest, and exercise the real
loader path (load_extension -> Scaffold -> covariate_features under the frozen alignment). No fit,
no outcome/IPC/expert column. Usage: python verify_actual_extension.py OUT_DIR RUN_DIR
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PKG = HERE.parents[5] / "FEWSNETGeoXGBExperiment"
sys.path.insert(0, str(PKG))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts import run_experiment as rx  # noqa: E402
from src.feature import fourclass_features as ff  # noqa: E402

out_dir, run = Path(sys.argv[1]), Path(sys.argv[2])
EXT = out_dir / "covariate_extension_2024-12_2025-06.csv"
report = json.loads((out_dir / "assembly_report.json").read_text(encoding="utf-8"))
PINNED, COMBINED = Path(report["raw_paths"]["pinned"]), Path(report["raw_paths"]["combined"])
MONTHS = ["2024-12", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


problems, res = [], {}
schema = json.loads((PKG / "feature-schema.json").read_text(encoding="utf-8"))
sources = list(schema["static_sources"] + schema["dynamic_sources_at_origin"])
columns = ["FEWSNET_admin_code", "date"] + sources
alignment_path = run / "prepared" / "manifests" / "alignment.json"
alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
kind = {s: alignment[s]["kind"] for s in sources}
res["raw_sha256"] = {"pinned": sha(PINNED), "combined": sha(COMBINED)}
if res["raw_sha256"] != report["raw_sha256_before"]:
    problems.append("raw source hashes differ from the assembly record")
res["output_sha256"] = sha(EXT)
if res["output_sha256"] != report["output_sha256"]:
    problems.append("extension bytes differ from the assembly record")

# 1-2. inventory and string-exact block equality against each named source
with open(EXT, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    emitted = {(r[0], r[1]): r[2:] for r in reader}
if header != columns:
    problems.append("extension columns differ from keys + 69 schema sources")
res["rows"] = len(emitted)
src_rows = {}
for path, months in ((PINNED, {"2024-12"}), (COMBINED, set(MONTHS[1:]))):
    with open(path, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            ym = d["date"][:7]
            if ym in months:
                src_rows[(d["FEWSNET_admin_code"], ym)] = [d[s] for s in sources]
res["source_keys"] = len(src_rows)
res["key_set_equal"] = set(src_rows) == set(emitted)
res["string_mismatch_rows"] = sum(src_rows.get(k) != v for k, v in emitted.items())
if not res["key_set_equal"] or res["string_mismatch_rows"]:
    problems.append("emitted blocks differ from their named sources")
ids = {}
for (a, ym) in emitted:
    ids.setdefault(ym, set()).add(a)
res["areas_per_month"] = {m: len(ids.get(m, ())) for m in MONTHS}
if any(n != 5718 for n in res["areas_per_month"].values()) or len({frozenset(v) for v in ids.values()}) != 1:
    problems.append("not all 5,718 areas in every month")

# 3. static boundary identity (pinned 2024-12 block vs combined 2025 rows, inside the file)
ext = pd.read_csv(EXT)
ext["ym"] = ext["date"].astype(str).str[:7]
base = ext[ext["ym"] == "2024-12"].set_index("FEWSNET_admin_code")
static_diff = {}
for m in MONTHS[1:]:
    cur = ext[ext["ym"] == m].set_index("FEWSNET_admin_code").reindex(base.index)
    for s in schema["static_sources"]:
        n = int((~np.isclose(base[s].to_numpy(float), cur[s].to_numpy(float), equal_nan=True, rtol=0, atol=1e-9)).sum())
        if n:
            static_diff[f"{m}:{s}"] = n
res["static_boundary_differences"] = static_diff
if static_diff:
    problems.append("static sources change across the splice boundary")
res["missing_by_kind_month"] = {f"{kind[s]}:{m}": 0 for s in sources if kind[s] != "excluded" for m in MONTHS}
for s in sources:
    if kind[s] == "excluded":
        continue
    for m in MONTHS:
        res["missing_by_kind_month"][f"{kind[s]}:{m}"] += int(ext.loc[ext["ym"] == m, s].isna().sum())
if problems:
    raise SystemExit(json.dumps({"problems": problems, **res}, indent=1, default=str))

# 4. manifest (written once)
overlap_evidence = HERE.with_name("extension_overlap_check_summary.json")
manifest = {
    "path": str(EXT), "sha256": res["output_sha256"], "first_month": "2024-12", "last_month": "2025-06",
    "overlap_months": ["2024-12"],
    "source": ("EXPLICIT TWO-SOURCE SPLICE (task 10-02, coordinator-approved). Rows 2024-12 are copied from the "
               "PINNED panel; rows 2025-01..2025-06 from the COMBINED 2025 panel; value fields are the original "
               "source strings, date normalised to YYYY-MM; columns = FEWSNET_admin_code, date and the 69 schema "
               "covariate sources; no patching, imputation or dropped rows. Because the overlap block comes from "
               "the pinned panel, the load_extension overlap check is TRUE BY CONSTRUCTION and is not evidence "
               "about the combined source. Identity evidence is external: on all 1,029,240 pinned keys "
               "(2010-01..2024-12) the 51 admitted sources of the combined panel equal the pinned panel exactly, "
               "and the 28 static sources are unchanged across the 2024-12/2025 boundary. The combined panel's "
               "historical Tair_zscore and Rainf_zscore DIFFER from the pinned panel and are NOT certified "
               "identical (excluded sources, never features). Limitations: ACLED sources are missing for 1,304 "
               "areas at 2025-01 and all 5,718 at 2025-05 (native NaN, no imputation); covariates are revised/"
               "latest values, not verified real-time vintages; the 2025 rows' producer run is not independently "
               "verified."),
    "components": [{"role": "overlap", "file": str(PINNED), "sha256": res["raw_sha256"]["pinned"],
                    "months": ["2024-12"]},
                   {"role": "extension", "file": str(COMBINED), "sha256": res["raw_sha256"]["combined"],
                    "months": MONTHS[1:]}],
    "alignment_sha256": sha(alignment_path),
    "evidence": {"assembly_script": str(HERE.with_name("assemble_actual_extension.py")),
                 "assembly_script_sha256": sha(HERE.with_name("assemble_actual_extension.py")),
                 "assembly_report_sha256": sha(out_dir / "assembly_report.json"),
                 "full_overlap_check": str(overlap_evidence), "full_overlap_check_sha256": sha(overlap_evidence),
                 "full_overlap_probe_sha256": sha(HERE.with_name("extension_overlap_check.py")),
                 "verify_script": str(HERE), "verify_script_sha256": sha(HERE),
                 "static_boundary_differences": 0},
}
mpath = out_dir / "extension_manifest.json"
with open(mpath, "x", encoding="utf-8") as f:
    json.dump(manifest, f, indent=1)
res["manifest_sha256"] = sha(mpath)

# 5. real loader path under the frozen alignment (no fit, no outcome columns)
src = json.loads((run / "prepared" / "manifests" / "sources.json").read_text(encoding="utf-8"))["sources"]["panel"]
pinned = rx._certified_panel(Path(src["path"]), src["sha256"], schema)
panel = rx.load_extension(mpath, schema, pinned)
scaffold = ff.Scaffold(panel, rx._covariate_columns(schema))
base_scaffold = ff.Scaffold(pinned, rx._covariate_columns(schema))
res["scaffold"] = {"areas": int(scaffold.areas.size), "first_month": ff.month_label([scaffold.first_month])[0],
                   "last_month": ff.month_label([scaffold.first_month + scaffold.n_months - 1])[0]}
names = ff.aligned_feature_names(schema, alignment)
res["aligned_features_total"] = len(names)
areas = scaffold.areas
checks = {}
for o_lab in ("2025-02", "2025-06", "2024-10"):
    o = ff.month_index([f"{o_lab}-01"])[0]
    origins = np.full(areas.size, o, dtype=np.int64)
    cov = ff.covariate_features(scaffold, schema, areas, origins + 4, origins, alignment)
    c = {"covariate_columns": int(cov.shape[1]), "subset_of_aligned": set(cov.columns) <= set(names)}
    cov_names = set(cov.columns)
    rows = np.array([scaffold.area_pos[int(a)] for a in areas])
    bad = []
    for s in cov.columns:
        k = kind.get(s)
        if k == "static":
            want = scaffold.at(s, rows, origins)
        elif k == "monthly":
            want = scaffold.at(s, rows, origins - alignment[s]["lag"])
        elif k == "annual":
            ref = ff.annual_reference_year(origins, alignment[s])
            want = scaffold.at(s, rows, (ref * 12 + alignment[s]["value_month"] - 1).astype(np.int64))
            c[f"{s}_reference_month"] = ff.month_label([int(ref[0] * 12 + alignment[s]["value_month"] - 1)])[0]
        else:
            continue
        if not np.array_equal(cov[s].to_numpy(float), want, equal_nan=True):
            bad.append(s)
    c["value_mismatch_vs_expected_month"] = bad
    c["static_month"] = o_lab
    c["monthly_month"] = ff.month_label([o - 1])[0]
    c["nan_cells_monthly"] = int(cov[[s for s in cov.columns if kind.get(s) == "monthly"]].isna().to_numpy().sum())
    if o_lab == "2024-10":
        ref_cov = ff.covariate_features(base_scaffold, schema, areas, origins + 4, origins, alignment)
        c["identical_to_pinned_only"] = bool(ref_cov.equals(cov))
    checks[o_lab] = c
res["covariate_features"] = checks
res["non_covariate_aligned_features"] = len(set(names) - cov_names)
(out_dir / "verification_report.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "missing_by_kind_month"}, indent=1, default=str))
