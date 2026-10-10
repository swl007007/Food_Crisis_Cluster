"""D30 production plan, completeness report and draft manifests (LOCAL ONLY; ipcch-geo interpreter; no EE call).

Builds the full piece plan from the staged frozen geometry and the P1 metadata footprints, checks per-admin completeness
(sourced windows + no-source windows = whole frozen area), writes the full production manifest (release_id None) and the
deterministic bounded live-sample manifests (release_id None). Nothing here is released or run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth import extraction
from ipcch_alphaearth.artifacts import write_csv_gz, write_json
from ipcch_alphaearth.contract import load_contract

import ee_pilot_c as epc

PROJECTION_CHECK = [[45.0, 13.0, 4341882.611290343, 1644631.1629230133], [-92.2, 38.4, -8896035.039132658, 4548060.477689638],
                    [127.3, -34.8, 12282703.475939127, -4177655.920457332]]
BASE = {"tile_scale": 4, "deadline_ms": 120000, "sdk_retries": 0, "budget_retries": 1}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pick(frame: pd.DataFrame, used: set, mask, key: str, q: float) -> str:
    """Deterministic: the unused piece in ``mask`` nearest the ``q`` quantile of ``key`` (ties by piece id)."""
    sub = frame[mask & ~frame["piece_id"].isin(used)]
    target = sub[key].quantile(q, interpolation="nearest")
    sub = sub.assign(_d=(sub[key] - target).abs()).sort_values(["_d", "piece_id"], kind="mergesort")
    pid = sub["piece_id"].iloc[0]
    used.add(pid)
    return pid


def select_samples(real: pd.DataFrame) -> dict:
    used: set = set()
    c, k, v = real["cells"], real["candidates"], real["vertices"]
    one, few, many = k == 1, k.between(2, 3), k >= 6
    s1 = [pick(real, used, one, "cells", 0.10), pick(real, used, one, "cells", 0.50), pick(real, used, one, "cells", 1.0),
          pick(real, used, few, "cells", 0.50), pick(real, used, few, "cells", 1.0), pick(real, used, few, "vertices", 1.0),
          pick(real, used, k == k.max(), "cells", 1.0), pick(real, used, many, "cells", 0.50), pick(real, used, many, "cells", 1.0),
          pick(real, used, real["parts"] > 1, "vertices", 1.0), pick(real, used, real["cells"] > 0, "vertices", 0.999),
          pick(real, used, c > 0, "cells", 0.0)]
    # 16 distinct pieces for concurrency 4: cell-count quantiles 1/17..16/17 across all pieces
    s2 = [pick(real, used, c > 0, "cells", (i + 1) / 17) for i in range(16)]
    # 8 small overhead-dominated pieces (cells near the 5th percentile) for the batch-vs-single comparison
    s3 = [pick(real, used, c > 0, "cells", 0.05) for _ in range(8)]
    return {"S1_sequential": s1, "S2_concurrency4": s2, "S3_batching": s3}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", required=True, help="staged frozen geometry .shp")
    p.add_argument("--list-images", required=True, help="P1 metadata ee/list_images.json (footprints)")
    p.add_argument("--out", required=True)
    p.add_argument("--max-cells", type=int, default=16_000_000)
    p.add_argument("--project", default="manuel-remote-sensing")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = load_contract()
    from shapely import get_num_coordinates  # noqa: PLC0415
    from shapely.geometry import shape  # noqa: PLC0415

    fps = {i["id"].split("/")[-1]: shape(i["geometry"]) for i in json.loads(Path(a.list_images).read_text())}
    geometry = extraction.load_geometry(Path(a.geometry))
    plan, geoms = extraction.plan_c_requests(geometry, fps, contract, max_cells=a.max_cells)
    shp = Path(a.geometry)
    provenance = {"geometry_sha256": {s: sha(shp.with_suffix(s)) for s in (".shp", ".shx", ".dbf", ".prj", ".cpg")},
                  "list_images_sha256": sha(Path(a.list_images)), "footprints": len(fps), "max_cells": a.max_cells,
                  "builder": "scripts/build_c_production.py"}
    plan_sha = write_csv_gz(out / "production_plan.csv.gz", plan)

    # ---------------------------------------------------------------- enumeration and completeness
    plan = plan.assign(vertices=[0 if g is None else int(get_num_coordinates(g)) for g in geoms],
                       parts=[0 if g is None else (len(g.geoms) if hasattr(g, "geoms") else 1) for g in geoms])
    real = plan[plan["piece_id"].notna()].assign(cells=lambda f: f["window_area_m2"] / 100.0)
    real = real.assign(bbox=[(c[1] - c[0]) * (w[1] - w[0]) for c, w in zip(real["cols"], real["rows"])])
    per_admin = plan.groupby("admin_code").agg(polygon=("polygon_area_m2", "first"), sourced=("window_area_m2", "sum"),
                                               no_source=("no_source_area_m2", "first"), pieces=("piece_id", "count"))
    per_admin["closure_rel"] = (per_admin["sourced"] + per_admin["no_source"]) / per_admin["polygon"] - 1.0
    q = [0, 0.1, 0.5, 0.9, 0.99, 1.0]
    report = {
        "plan_sha256": plan_sha, "provenance": provenance, "recipe": extraction.recipe_id(contract),
        "admins": int(plan["admin_code"].nunique()), "geometry_admins": int(len(geometry)),
        "admins_with_pieces": int((per_admin["pieces"] > 0).sum()), "zero_source_admins": int((per_admin["pieces"] == 0).sum()),
        "pieces": int(len(real)), "piece_ids_unique": bool(real["piece_id"].is_unique),
        "cells_total": float(real["cells"].sum()), "band_pixels_total": float(129 * real["cells"].sum()),
        "cells_quantiles": dict(zip(map(str, q), real["cells"].quantile(q).round(1).tolist())),
        "candidates_counts": {str(int(k)): int(n) for k, n in real["candidates"].value_counts().sort_index().items()},
        "pieces_per_admin_quantiles": dict(zip(map(str, q), per_admin["pieces"].quantile(q).tolist())),
        "vertices_quantiles": dict(zip(map(str, q), real["vertices"].quantile(q).tolist())),
        "multipart_pieces": int((real["parts"] > 1).sum()),
        "no_source_area_km2": float(per_admin["no_source"].sum() / 1e6), "no_source_windows": int(plan.groupby("admin_code")["no_source_windows"].first().sum()),
        "area_closure_rel_max_abs": float(per_admin["closure_rel"].abs().max()),
        "area_closure_rel_median_abs": float(per_admin["closure_rel"].abs().median()),
        "bbox_cells_max": int(real["bbox"].max()), "bbox_cells_le_cap": bool(real["bbox"].max() <= a.max_cells),
        "geodesic_over_bbox_max": float((real["cells"] / real["bbox"]).max()),
        "subcell_pieces": int((real["cells"] < 1).sum()), "pieces_below_100_cells": int((real["cells"] < 100).sum()),
    }

    # ---------------------------------------------------------------- full production manifest (draft, unreleased)
    full_limits = {**BASE, "concurrency": 4, "batch_size": 1, "max_high_level_calls": int(np.ceil(len(real) * 1.05)) + 2,
                   "max_consecutive_failures": 5, "max_failure_fraction": 0.02, "min_attempts_for_fraction": 200,
                   "wall_budget_s": 4 * 3600, "outer_timeout_s": 4 * 3600 + 600}
    full = extraction.write_production_manifest(plan.drop(columns=["vertices", "parts"]), geoms, out, contract, limits=full_limits,
                                                grid_crs_wkt=epc.EE_CRS_WKT, projection_check=PROJECTION_CHECK, project=a.project,
                                                provenance={**provenance, "plan_sha256": plan_sha}, name="production")
    report["full_manifest"] = {"request_sha256": full["request_sha256"], "pieces_sha256": full["pieces_sha256"],
                               "n_pieces": full["n_pieces"], "max_pixels": full["limits"]["max_pixels"],
                               "pieces_file_bytes": (out / full["pieces_file"]).stat().st_size, "limits": full["limits"]}

    # ---------------------------------------------------------------- bounded live sample manifests (draft, unreleased)
    samples = select_samples(real)
    sample_limits = {
        "S1_sequential": {"concurrency": 1, "batch_size": 1, "max_high_level_calls": 2 + 2 * 12, "wall_budget_s": 900},
        "S2_concurrency4": {"concurrency": 4, "batch_size": 1, "max_high_level_calls": 2 + 2 * 16, "wall_budget_s": 600},
        "S3_batching": {"concurrency": 1, "batch_size": 4, "max_high_level_calls": 2 + 2 * 2, "wall_budget_s": 300},
        "S3_single": {"concurrency": 1, "batch_size": 1, "max_high_level_calls": 2 + 2 * 8, "wall_budget_s": 300},
    }
    samples["S3_single"] = samples["S3_batching"]  # same 8 pieces singly: overhead and value identity vs the batch
    report["samples"] = {}
    for name, ids in samples.items():
        lim = {**BASE, **sample_limits[name], "outer_timeout_s": sample_limits[name]["wall_budget_s"] + 300, "max_consecutive_failures": 2, "max_failure_fraction": 0.25, "min_attempts_for_fraction": 4}
        m = extraction.write_production_manifest(plan.drop(columns=["vertices", "parts"]), geoms, out / "samples", contract,
                                                 limits=lim, grid_crs_wkt=epc.EE_CRS_WKT, projection_check=PROJECTION_CHECK,
                                                 project=a.project, provenance={**provenance, "plan_sha256": plan_sha},
                                                 select=ids, name=name)
        rows = real.set_index("piece_id").loc[ids]
        report["samples"][name] = {
            "request_sha256": m["request_sha256"], "pieces_sha256": m["pieces_sha256"], "n_pieces": m["n_pieces"],
            "limits": m["limits"], "cells_total": float(rows["cells"].sum()),
            "pieces": [{"piece_id": pid, "cells": float(r.cells), "candidates": int(r.candidates), "vertices": int(r.vertices),
                        "parts": int(r.parts)} for pid, r in rows.iterrows()]}
    write_json(out / "production_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples",)}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
