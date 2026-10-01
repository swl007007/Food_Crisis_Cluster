"""Producer-created integration run on a real-data area subset (NON-AUTHORITATIVE).

python -B tests/integration_minirun.py --out C:\\...\\geoxgb_runs\\minirun-<id> [--countries MW,TD,ML,SS]

Builds a source root holding the pinned panel / FEWS NET / coordinates / shapefile rows
of a few whole countries (every month of every selected area, so the panel stays a
complete scaffold), runs the production preparation with the source pins re-pointed to
that subset (the only patched step), then every production phase through its CLI —
gscreen, Stage 1 (all 162 roots), maps, develop, select, oldmap, freeze, final — and the
report and the verifier. No score, label or record is fabricated anywhere: the chain is
exactly the authoritative one on fewer areas. Its output is evidence that the producers
and consumers fit together; it is never an input to, or a result of, the experiment.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import pandas as pd  # noqa: E402

from scripts import prepare_fourclass as prep  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_sources(source_root: Path, target: Path, countries) -> dict:
    import geopandas as gpd
    target.mkdir(parents=True)
    panel_rel, fews_rel, coord_rel, shp_rel = (prep.PINNED_SOURCES[k][0] for k in ("panel", "fewsnet", "coordinates", "shapefile"))
    panel = pd.read_csv(source_root / panel_rel, low_memory=False)
    keep = sorted(panel.loc[panel["ISO"].isin(countries), "FEWSNET_admin_code"].unique())
    panel[panel["FEWSNET_admin_code"].isin(keep)].to_csv(target / panel_rel, index=False)
    fews = pd.read_csv(source_root / fews_rel, low_memory=False)
    (target / fews_rel).parent.mkdir(parents=True, exist_ok=True)
    fews[fews["admin_code"].isin(keep) | fews["admin_code"].isna()].to_csv(target / fews_rel, index=False)
    coords = pd.read_csv(source_root / coord_rel)
    coords[coords["FEWSNET_admin_code"].isin(keep)].to_csv(target / coord_rel, index=False)
    shapes = gpd.read_file(source_root / shp_rel)
    (target / shp_rel).parent.mkdir(parents=True, exist_ok=True)
    shapes[shapes["admin_code"].isin(keep)].to_file(target / shp_rel, encoding="utf-8")
    pins = {k: (rel, sha256(target / rel)) for k, (rel, _) in prep.PINNED_SOURCES.items()}
    return {"areas": len(keep), "countries": list(countries), "pins": pins}


def run(cmd, log):
    started = time.time()
    with open(log, "w", encoding="utf-8") as handle:
        code = subprocess.run(cmd, cwd=PACKAGE, stdout=handle, stderr=subprocess.STDOUT).returncode
    print(json.dumps({"step": " ".join(cmd[2:4]), "returncode": code, "seconds": round(time.time() - started)}), flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--countries", default="MW,TD,ML,SS")
    parser.add_argument("--workers", default="6")
    args = parser.parse_args()
    out = args.out
    if out.exists():
        raise FileExistsError(f"{out} exists")
    out.mkdir(parents=True)
    fixture = build_sources(prep.DEFAULT_SOURCE_ROOT, out / "sources", args.countries.split(","))
    (out / "INTEGRATION_FIXTURE.json").write_text(json.dumps({
        "authoritative": False,
        "purpose": "producer-created integration evidence on a real-data area subset; never an experiment input",
        **{k: v for k, v in fixture.items() if k != "pins"},
        "patched_step": "prepare_fourclass PINNED_SOURCES re-pointed to the subset files (in-process only)",
        "subset_source_sha256": {k: v[1] for k, v in fixture["pins"].items()}}, indent=2), encoding="utf-8")
    run_dir = out / "run"
    argv = ["prepare_fourclass.py", "--run-dir", str(run_dir), "--source-root", str(out / "sources")]
    with patch.dict(prep.PINNED_SOURCES, fixture["pins"]), patch.object(sys, "argv", argv):
        prep.main()
    py, logs = sys.executable, out / "logs"
    logs.mkdir()
    steps = [["scripts/run_experiment.py", "--run-dir", str(run_dir), "gscreen", "--workers", args.workers],
             ["scripts/run_stage1.py", "--run-dir", str(run_dir), "--workers", args.workers]]
    steps += [["scripts/run_experiment.py", "--run-dir", str(run_dir), phase] +
              (["--workers", args.workers] if phase in ("maps", "develop", "oldmap", "final") else [])
              for phase in ("maps", "develop", "select", "oldmap", "freeze", "final")]
    steps += [["scripts/report_fourclass.py", "--run-dir", str(run_dir)],
              ["scripts/verify_fourclass.py", "--run-dir", str(run_dir)]]
    for i, step in enumerate(steps):
        if run([py, "-B"] + step, logs / f"{i:02d}_{Path(step[0]).stem}_{step[3] if len(step) > 3 else ''}.log") != 0:
            print(f"integration step failed: {step}", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
