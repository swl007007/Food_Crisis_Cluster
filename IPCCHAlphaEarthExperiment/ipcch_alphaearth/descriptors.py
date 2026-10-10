"""AlphaEarth descriptor table, coverage/route ledger and the 424-column schema (design sections 4-5, D16/D25).

``descriptor_table`` reaggregates descriptors offline from saved sufficient statistics (the shared
``aemoments`` merge), so moments -> descriptors is reproducible without EE. ``attach_descriptors`` (P1) refuses
to run without a COMPLETE extraction manifest; P0 never contacts EE.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth import aemoments
from ipcch_alphaearth.artifacts import require_stage
from ipcch_alphaearth.errors import ContractError, TechnicalError


def ae_names(contract: dict) -> list:
    f = contract["features"]
    n = int(f["alphaearth_bands"])
    return [f"{f['alphaearth_prefix_mean']}{j:02d}" for j in range(n)] + [f"{f['alphaearth_prefix_sd']}{j:02d}" for j in range(n)]


def descriptor_table(pieces: pd.DataFrame, plan: pd.DataFrame, polygons: pd.DataFrame, contract: dict,
                     expected_admins=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-admin descriptors (admin_code + 128 columns; NaN rows for low/no coverage) and the coverage ledger.

    ``pieces``: admin_code, piece_id, asset_id, status, valid_area, S_A00..S_A63, Q_A00..Q_A63 (completed export rows).
    ``plan``: admin_code, piece_id (every planned piece). ``polygons``: admin_code, polygon_area_m2.
    A planned admin without its full completed piece set is a technical failure, never low coverage.
    """
    from ipcch_alphaearth.extraction import check_piece_binding  # noqa: PLC0415

    ae = contract["alphaearth"]
    s_cols = [f"S_{b}" for b in aemoments.BANDS]
    q_cols = [f"Q_{b}" for b in aemoments.BANDS]
    if polygons["admin_code"].duplicated().any():
        raise TechnicalError("polygon table has duplicate admins")
    plan_admins, poly_admins = set(plan["admin_code"]), set(polygons["admin_code"])
    if plan_admins != poly_admins:
        raise TechnicalError(f"plan and polygon admin sets differ ({len(plan_admins - poly_admins)} plan-only, "
                             f"{len(poly_admins - plan_admins)} polygon-only): every declared geometry needs a plan entry")
    if expected_admins is not None and poly_admins != set(expected_admins):
        raise TechnicalError("descriptor table admins differ from the declared geometry admins")
    from ipcch_alphaearth.extraction import recipe_id  # noqa: PLC0415

    want = recipe_id(contract)
    if "recipe" not in plan.columns or set(plan["recipe"].dropna()) != {want}:
        raise TechnicalError(f"piece plan was not built under the frozen extraction recipe {want} (old-rule artifacts are refused)")
    check_piece_binding(pieces, plan)
    area = polygons.set_index("admin_code")["polygon_area_m2"]
    planned = plan.dropna(subset=["piece_id"]).groupby("admin_code")["piece_id"].apply(list)  # zero-piece admins plan []
    got = {a: g for a, g in pieces.groupby("admin_code")}
    rows, ledger = [], []
    for admin in sorted(area.index):
        plan_ids = planned.get(admin, [])
        frame = got.get(admin)
        recs = [] if frame is None else [
            {"piece_id": r.piece_id, "status": r.status, "valid_area": float(r.valid_area),
             "S": [float(getattr(r, c)) for c in s_cols], "Q": [float(getattr(r, c)) for c in q_cols]}
            for r in frame.itertuples(index=False)]
        try:
            merged = aemoments.merge_pieces(recs, list(plan_ids))
            d = aemoments.descriptor(merged, float(area.loc[admin]), ae["coverage_threshold"], float(ae["variance_cancellation_tolerance"]))
        except aemoments.ExtractionFailure as error:
            raise TechnicalError(f"admin {admin}: {error}") from error
        rows.append([admin, *d["mean"], *d["sd"]])
        ledger.append({"admin_code": int(admin), "pieces": merged["pieces"], "valid_area_m2": d["valid_area_m2"],
                       "polygon_area_m2": d["polygon_area_m2"], "coverage": d["coverage"], "usable": d["usable"],
                       "route_reason": d["route_reason"], "negative_variance_zeroed": d["negative_variance_zeroed"]})
    table = pd.DataFrame(rows, columns=["admin_code", *ae_names(contract)])
    return table, pd.DataFrame(ledger)


def row_descriptors(admin_codes: np.ndarray, table: pd.DataFrame, contract: dict) -> tuple[np.ndarray, np.ndarray]:
    """(n, 128) descriptors per row (NaN for non-routable admins) and the routable mask."""
    names = ae_names(contract)
    t = table.set_index("admin_code")
    missing = sorted(set(np.unique(admin_codes)) - set(t.index))
    if missing:
        raise TechnicalError(f"{len(missing)} cohort admins absent from the descriptor table (not legitimate low coverage)")
    D = t.loc[admin_codes, names].to_numpy(dtype=np.float64)
    finite = np.isfinite(D)
    all_finite, all_nan = finite.all(axis=1), (~finite).all(axis=1)
    if not (all_finite | all_nan).all():
        raise TechnicalError("descriptor rows must be all finite or all NaN together")
    return D, all_finite


def ingest_extraction(run_dir: Path, contract: dict, plan_path: Path, manifest_path: Path, outputs_dir: Path) -> dict:
    """Bind a COMPLETE D30 production extraction to a run.

    Every check runs BEFORE anything is written to ``extraction/``: the released full-plan manifest; the supplied plan
    bytes equal the manifest's pinned ``provenance.plan_sha256`` (frozen denominators and geometry provenance); every
    planned piece present exactly once with a matching binding; every piece numerically complete (no failed, null or
    nonfinite moments: technical missingness is never low coverage; a completed all-zero piece stays genuine no
    coverage). A refusal leaves ``extraction/`` absent and records its context under ``extraction_attempts/``."""
    from ipcch_alphaearth.artifacts import digest, sha256_file, write_csv_gz, write_json  # noqa: PLC0415
    from ipcch_alphaearth.extraction import load_production_manifest, read_c_outputs, recipe_id, validate_export  # noqa: PLC0415

    require_stage(run_dir, "prepared")
    out = Path(run_dir) / "extraction"
    if out.exists():
        raise ContractError("extraction stage already exists; it is immutable")
    try:
        manifest, pieces = load_production_manifest(manifest_path)
        if not manifest.get("release_id") or manifest.get("sample"):
            raise ContractError("only the released full-plan production manifest can be ingested (not a sample or an unreleased draft)")
        plan_sha = sha256_file(plan_path)
        if plan_sha != (manifest.get("provenance") or {}).get("plan_sha256"):
            raise ContractError("supplied plan bytes differ from the plan pinned in the production manifest")
        plan = pd.read_csv(plan_path, float_precision="round_trip", dtype={"piece_id": str, "asset_id": str, "recipe": str})
        if manifest.get("recipe") != recipe_id(contract) or set(plan["recipe"]) != {recipe_id(contract)}:
            raise ContractError("plan/manifest were not built under the frozen D30 recipe")
        planned = {r.piece_id: (int(r.admin_code), str(r.asset_id)) for r in plan.dropna(subset=["piece_id"]).itertuples(index=False)}
        listed = {r["request_id"]: (int(r["admin_code"]), ",".join(r["candidates"])) for r in pieces}
        if planned != listed:
            raise ContractError("manifest pieces differ from the plan (piece set, admin or ordered candidates)")
        table = read_c_outputs(outputs_dir, manifest, pieces)
        got, stats = validate_export([table], plan)
        bad = got[got["status"] != "complete"]
        if len(bad):
            raise TechnicalError(f"{len(bad)} pieces are not numerically complete ({bad['status'].value_counts().to_dict()}); "
                                 "rerun them under a release, never ingest them as coverage")
    except (ContractError, TechnicalError) as error:
        attempts = Path(run_dir) / "extraction_attempts"
        attempts.mkdir(exist_ok=True)
        context = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "error_type": type(error).__name__, "error": str(error),
                   "plan": str(plan_path), "manifest": str(manifest_path), "outputs": str(outputs_dir)}
        if "bad" in locals() and len(bad):
            context["failed_pieces"] = bad[["piece_id", "admin_code", "status"]].to_dict("records")
        write_json(attempts / f"refused-{len(list(attempts.glob('refused-*.json'))) + 1:04d}.json", context)
        raise
    out.mkdir()
    files = {"plan.csv.gz": _copy_new(plan_path, out / "plan.csv.gz"), "pieces.csv.gz": write_csv_gz(out / "pieces.csv.gz", got),
             "manifest.json": write_json(out / "manifest.json", manifest)}
    record = {"stage": "extraction", "recipe": recipe_id(contract), "manifest_sha256": manifest["request_sha256"],
              "release_id": manifest["release_id"], "plan_sha256": plan_sha, "provenance": manifest["provenance"], "files": files,
              **stats, "status_counts": got["status"].value_counts().to_dict(),
              "plan_digest": digest(plan[["admin_code", "piece_id", "asset_id"]].astype(str).values.tolist())}
    write_json(out / "COMPLETE.json", record)
    return record


def _copy_new(src: Path, dst: Path) -> str:
    """Byte copy (the stored plan keeps the manifest-pinned sha), never overwriting."""
    import shutil  # noqa: PLC0415

    from ipcch_alphaearth.artifacts import sha256_file  # noqa: PLC0415

    if dst.exists():
        raise ContractError(f"refusing to overwrite existing artifact {dst}")
    shutil.copyfile(src, dst)
    return sha256_file(dst)


def attach_descriptors(run_dir: Path, contract: dict) -> dict:
    """P1: descriptor table + coverage ledger from the COMPLETE D30 extraction, bound to the current configuration."""
    from ipcch_alphaearth.artifacts import read_json, sha256_file, write_csv_gz, write_json  # noqa: PLC0415
    from ipcch_alphaearth.contract import config_identity  # noqa: PLC0415

    require_stage(run_dir, "prepared")
    ext = Path(run_dir) / "extraction"
    if not (ext / "COMPLETE.json").exists():
        raise ContractError("no COMPLETE extraction stage: AlphaEarth descriptors are produced only after the P1 cloud release")
    out = Path(run_dir) / "descriptors"
    if (out / "COMPLETE.json").exists():
        raise ContractError("descriptors stage is COMPLETE; it is immutable")
    record = read_json(ext / "COMPLETE.json")
    for name, sha in record["files"].items():
        if sha256_file(ext / name) != sha:
            raise ContractError(f"extraction file {name} differs from its COMPLETE record")
    if record["files"]["plan.csv.gz"] != read_json(ext / "manifest.json")["provenance"]["plan_sha256"]:
        raise ContractError("stored plan differs from the plan pinned in the production manifest")
    plan = pd.read_csv(ext / "plan.csv.gz", dtype={"piece_id": str, "asset_id": str, "recipe": str})
    pieces = pd.read_csv(ext / "pieces.csv.gz", float_precision="round_trip", dtype={"piece_id": str, "asset_id": str, "recipe": str, "status": str})
    polygons = plan.groupby("admin_code", as_index=False)["polygon_area_m2"].first()
    keyed = set()
    for h in contract["calendar"]["horizons_months"]:
        keyed |= set(pd.read_csv(Path(run_dir) / "prepared" / f"keys_h{h:02d}.csv.gz", usecols=["admin_code"])["admin_code"].astype(int))
    if not keyed <= set(polygons["admin_code"]):
        raise ContractError(f"{len(keyed - set(polygons['admin_code']))} prepared admins have no planned geometry")
    table, ledger = descriptor_table(pieces, plan, polygons, contract, expected_admins=sorted(polygons["admin_code"]))
    out.mkdir(parents=True, exist_ok=True)
    files = {"descriptor_table.csv.gz": write_csv_gz(out / "descriptor_table.csv.gz", table),
             "coverage_ledger.csv.gz": write_csv_gz(out / "coverage_ledger.csv.gz", ledger)}
    done = {"stage": "descriptors", "recipe": record["recipe"], "config_sha256": config_identity(),
            "prepared_manifest_sha256": sha256_file(Path(run_dir) / "prepared" / "prepared-manifest.json"),
            "extraction_complete_sha256": sha256_file(ext / "COMPLETE.json"), "plan_sha256": record["plan_sha256"],
            "provenance": record["provenance"], "files": files,
            "admins": int(len(table)), "usable": int(ledger["usable"].sum()),
            "route_reasons": ledger["route_reason"].value_counts().to_dict()}
    write_json(out / "COMPLETE.json", done)
    return done
