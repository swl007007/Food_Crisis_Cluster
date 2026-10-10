"""Local-only preparation (P0): canonical population truth, per-H cohorts, compact296 rebuilt at row origins,
persistence, comparisons with P6 keys and the compact tables, source-support perturbation and the no-fit
calendar/provider enumeration. No model is fitted and EE is never contacted.

Outputs ``<run>/prepared/``; every artifact SHA256 is bound in ``prepared-manifest.json``. AlphaEarth
descriptors are attached later by ``attach_descriptors`` (P1), which builds the 424-column schema.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth import compact as cp
from ipcch_alphaearth import schedule, targets
from ipcch_alphaearth.artifacts import (list_sha256, read_json, record_incomplete, sha256_file, write_csv_gz, write_json,
                                        write_npy)
from ipcch_alphaearth.contract import config_identity
from ipcch_alphaearth.errors import ContractError
from ipcch_alphaearth.months import month_label, month_ord, parse_month
from ipcch_alphaearth.sources import code_inventory, stage_inputs, staged_path

STAGE = "prepared"
SUPPORT_CUTOFFS = ("2022-06", "2024-03")
AVAILABILITY = {
    "id": "D29-observation-month-end",
    "origin": "O = T - H, end of month O",
    "fit_labels": "target month U <= fit origin (inclusive)",
    "row_inputs": "observation/report month <= the row's own origin; reported-phase history month <= min(O, T-1)",
    "seasons": "gs_end_date_exclusive <= first day of month O+1",
    "persistence": "latest QC-valid population-derived observation of the same admin with month <= O; no age cutoff",
    "limitation": "observation-month proxy; release vintages, static snapshots and supplier climatology samples unverified",
}


def log(t0: float, message: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {message}", flush=True)


def country_table(path: Path) -> pd.DataFrame:
    """country_key (P6 rule: trimmed country_en, else country) and ISO3 (blank -> None) per admin."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    for column in ("area_id", "iso3", "country", "country_en"):
        if column not in raw.columns:
            raise ContractError(f"country lookup lacks {column}")
    out = pd.DataFrame({"admin_code": raw["area_id"].astype(np.int64)})
    if out["admin_code"].duplicated().any():
        raise ContractError("country lookup has duplicate area_id")
    en, fallback = raw["country_en"].str.strip(), raw["country"].str.strip()
    out["country_key"] = np.where(en.ne(""), en, fallback)
    if (out["country_key"] == "").any():
        raise ContractError("country lookup has an admin without a country name")
    iso = raw["iso3"].str.strip()
    out["iso3"] = np.where(iso.ne(""), iso, None)
    return out.sort_values("admin_code", kind="mergesort").reset_index(drop=True)


def persistence(valid: pd.DataFrame, admin: np.ndarray, origin: np.ndarray) -> pd.DataFrame:
    """Latest QC-valid observation with month <= O per row (phase/q3/source month/age); none -> unavailable."""
    v = valid.sort_values(["admin_code", "month_ord"], kind="mergesort")
    scale = 1 << 20
    key = v["admin_code"].to_numpy(np.int64) * scale + v["month_ord"].to_numpy(np.int64)
    query = admin.astype(np.int64) * scale + origin.astype(np.int64)
    slot = np.searchsorted(key, query, side="right") - 1
    safe = np.clip(slot, 0, None)
    hit = (slot >= 0) & (v["admin_code"].to_numpy(np.int64)[safe] == admin)
    src = np.where(hit, v["month_ord"].to_numpy(np.int64)[safe], -1)
    if (src[hit] > origin[hit]).any():
        raise ContractError("persistence source after the row origin")
    return pd.DataFrame({
        "persistence_available": hit.astype(np.int64),
        "persistence_phase": np.where(hit, v["phase_truth"].to_numpy(np.int64)[safe], 0).astype(np.int64),
        "persistence_q3": np.where(hit, v["q3"].to_numpy(np.float64)[safe], np.nan),
        "persistence_source_month": month_label(src),
        "persistence_age_months": np.where(hit, origin - src, -1).astype(np.int64),
    })


def load_grids(src: dict, members: dict, t0: float) -> tuple:
    interim_path, monthly_path, seasonal_path = src["interim"], src["climate_monthly"], src["climate_seasonal"]
    header = pd.read_csv(interim_path, nrows=0).columns
    if cp.commodity_members(header) != members:
        raise ContractError("bbg member columns in the interim header differ from the pinned compact members")
    member_cols = [c for cols in members.values() for c in cols]
    panel = pd.read_csv(interim_path, usecols=["admin_code", "year", "month", "overall_phase", *cp.INTERIM_SOURCES[:7], *member_cols])
    panel = panel.rename(columns={"admin_code": "area_id"})
    if panel.duplicated(list(cp.KEYS)).any():
        raise ContractError("interim panel has duplicate keys")
    observations = cp.valid_phase_observations(panel.loc[pd.to_numeric(panel["overall_phase"], errors="coerce").isin(cp.VALID_PHASES)])
    interim = cp.interim_grid(panel, members)
    interim_long = panel[["area_id", "year", "month", *cp.INTERIM_SOURCES[:7]]].copy()
    del panel
    log(t0, f"interim grid {len(interim.area_ids)} areas, {month_label(interim.first_ord)}..{month_label(interim.first_ord + interim.n_months - 1)}; "
            f"reported-phase observations {len(observations)}")
    monthly = pd.read_csv(monthly_path, usecols=["admin_code", "year", "month", *cp.CLIMATE_SOURCES]).rename(columns={"admin_code": "area_id"})
    climate = cp.Grid.from_long(monthly, cp.CLIMATE_SOURCES)
    log(t0, f"climate grid {len(climate.area_ids)} areas, {month_label(climate.first_ord)}..{month_label(climate.first_ord + climate.n_months - 1)}")
    for grid in (interim, climate):
        for v, arr in grid.values.items():
            if np.isinf(arr).any():
                raise ContractError(f"{v}: infinite source values")
    seasons = pd.read_csv(seasonal_path, usecols=["admin_code", "season_year", "season", "gs_start_date", "gs_end_date_exclusive",
                                                  *cp.SEASON_METRICS])
    return interim, climate, seasons, observations, interim_long, monthly


def perturbed(grid: cp.Grid, cutoff: int, rng) -> cp.Grid:
    months = np.arange(grid.n_months) + grid.first_ord
    cols = months > cutoff
    values = {}
    for v, arr in grid.values.items():
        arr = arr.copy()
        arr[:, cols] = arr[:, cols] + rng.normal(100.0, 10.0, arr[:, cols].shape)
        values[v] = arr
    return cp.Grid(grid.area_ids, grid.first_ord, grid.n_months, values)


def same_cells(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    return (x == y) | (np.isnan(x) & np.isnan(y))


def run_prepare(run_dir: Path, contract: dict, inputs_cfg: dict, local_only: bool = True) -> dict:
    try:
        return _run_prepare(Path(run_dir), contract, inputs_cfg)
    except Exception as error:
        record_incomplete(run_dir, STAGE, {"stage": "prepare", "local_only": local_only}, error)
        raise


def _run_prepare(run_dir: Path, contract: dict, inputs_cfg: dict) -> dict:
    t0 = time.time()
    out = run_dir / STAGE
    if out.exists():
        raise ContractError(f"{out} exists; prepared outputs are immutable (create a new run)")
    out.mkdir()
    artifacts: dict[str, str] = {}

    def save(name: str, obj) -> None:
        path = out / name
        if isinstance(obj, pd.DataFrame):
            artifacts[name] = write_csv_gz(path, obj)
        elif isinstance(obj, np.ndarray):
            artifacts[name] = write_npy(path, obj)
        else:
            artifacts[name] = write_json(path, obj)

    manifest = stage_inputs(run_dir, inputs_cfg)
    src = {name: staged_path(run_dir, manifest, name) for name in manifest["inputs"]}
    log(t0, f"staged {len(src)} inputs")
    compact_manifest = read_json(src["compact_manifest"])
    frozen_order = compact_manifest["horizons"]["0"]["arms"]["compact_baseline"]["features"]
    if list_sha256(frozen_order) != contract["features"]["compact296_list_sha256"]:
        raise ContractError("compact manifest feature order differs from the frozen compact296 digest")
    groups = cp.feature_groups(frozen_order)
    members = compact_manifest["commodity_members"]

    # ------------------------------------------------------------------ canonical truth
    ledger = targets.build_target_ledger(src["raw_panel"])
    valid = targets.valid_targets(ledger)
    qc = contract["target_qc"]
    if len(valid) != qc["expected_valid_rows"] or valid["admin_code"].nunique() != qc["expected_valid_admins"]:
        raise ContractError(f"valid ledger {len(valid)} rows / {valid['admin_code'].nunique()} admins differs from "
                            f"{qc['expected_valid_rows']} / {qc['expected_valid_admins']} (source drift: stop and explain)")
    save("target_ledger.csv.gz", ledger)
    save("target_ledger_valid.csv.gz", valid)
    summary = targets.ledger_summary(ledger)
    log(t0, f"target ledger {summary['rows']} rows, valid {summary['valid']} / {summary['areas_with_valid']} admins")
    p6_valid = pd.read_csv(src["p6_target_ledger_valid"], dtype={"q2_str": str, "q3_str": str, "q4_str": str, "q5_str": str})
    cmp_cols = ["admin_code", "month_ord", "phase_truth", "q2_str", "q3_str", "q4_str", "q5_str"]
    mine = valid[cmp_cols].reset_index(drop=True)
    theirs = p6_valid[cmp_cols].sort_values(["admin_code", "month_ord"], kind="mergesort").reset_index(drop=True)
    p6_equal = bool(mine.astype(str).equals(theirs.astype(str)))
    if not p6_equal:
        raise ContractError("rebuilt canonical valid ledger differs from the saved P6 ledger (keys/phase/q strings)")
    del ledger

    countries = country_table(src["country_lookup"])
    missing_country = sorted(set(valid["admin_code"]) - set(countries["admin_code"]))
    if missing_country:
        raise ContractError(f"{len(missing_country)} valid admins lack a country key")
    country_by = countries.set_index("admin_code")

    # ------------------------------------------------------------------ sources and static
    interim, climate, seasons, observations, interim_long, monthly_long = load_grids(src, members, t0)
    coords = pd.read_csv(src["raw_panel"], usecols=["admin_code", "year", "month", "lat", "lon"]).rename(columns={"admin_code": "area_id"})
    tables = {}
    for h in (0, 3, 6, 12):
        tables[h] = pd.read_csv(src[f"compact_baseline_h{h}"], float_precision="round_trip", low_memory=False)
        if list(tables[h].columns[-296:]) != list(frozen_order):
            raise ContractError(f"compact baseline h{h} column order differs from the frozen order")
    static, static_audit = cp.static_per_admin([tables[h] for h in (0, 3, 6, 12)], groups["static"])
    missing_static = sorted(set(valid["admin_code"]) - set(static.index))
    if missing_static:
        raise ContractError(f"{len(missing_static)} canonical admins have no static row in the pinned compact tables")
    save("static29_by_admin.csv.gz", static.reset_index().rename(columns={"area_id": "admin_code"}))
    log(t0, f"static29 for {static_audit['admins']} admins; all-missing popdensity {static_audit['all_missing_admins_by_column'].get('popdensity')}")
    idp_obs = cp.idp_observations(pd.read_csv(src["idp_admin0_monthly"]))

    # ------------------------------------------------------------------ rows (identical canonical keys for every H)
    rows = pd.DataFrame({"area_id": valid["admin_code"].to_numpy(np.int64), "year": valid["year"].to_numpy(np.int64),
                         "month": valid["month"].to_numpy(np.int64)})
    target_ord = month_ord(rows["year"], rows["month"])
    horizons = list(contract["calendar"]["horizons_months"])
    monthly = cp.monthly_block(interim, climate, rows, horizons)
    log(t0, f"monthly blocks built for H={horizons}")
    ident = cp.identifier_block(rows, coords)
    del coords

    # source-support perturbation: months after the cutoff must not change rows with origin <= cutoff
    rng = np.random.default_rng(20261010)
    support_rows = []
    for cutoff_text in SUPPORT_CUTOFFS:
        cutoff = parse_month(cutoff_text)
        late = cp.monthly_block(perturbed(interim, cutoff, rng), perturbed(climate, cutoff, rng), rows, horizons)
        pseasons = seasons.copy()
        first_day_after = pd.Timestamp(year=(cutoff + 1) // 12, month=(cutoff + 1) % 12 + 1, day=1)
        late_season = (pd.to_datetime(pseasons["gs_end_date_exclusive"]) > first_day_after).to_numpy()
        for m in cp.SEASON_METRICS:
            pseasons.loc[late_season, m] = pseasons.loc[late_season, m] + 100.0
        for h in horizons:
            origin = target_ord - h
            early, later = origin <= cutoff, origin > cutoff
            base_s, _ = cp.completed_seasons(seasons, rows["area_id"].to_numpy(), origin)
            pert_s, _ = cp.completed_seasons(pseasons, rows["area_id"].to_numpy(), origin)
            same = np.concatenate([same_cells(monthly[h], late[h]), same_cells(base_s, pert_s)], axis=1)
            names = list(monthly[h].columns) + list(base_s.columns)
            changed_early = (~same & early[:, None]).sum(axis=0)
            changed_late = (~same & later[:, None]).sum(axis=0)
            for name, ce, cl in zip(names, changed_early, changed_late):
                support_rows.append({"horizon": h, "cutoff": cutoff_text, "feature": name, "rows_origin_le_cutoff": int(early.sum()),
                                     "changed_rows_origin_le_cutoff": int(ce), "power_changed_rows_origin_gt_cutoff": int(cl)})
        del late
    support = pd.DataFrame(support_rows)
    save("source_support.csv.gz", support)
    if (support["changed_rows_origin_le_cutoff"] > 0).any():
        bad = support[support["changed_rows_origin_le_cutoff"] > 0]
        raise ContractError(f"post-origin perturbation changed {len(bad)} feature/cutoff cells (temporal ancestry failure)")
    log(t0, f"source-support perturbation passed ({len(support)} cells)")

    # ------------------------------------------------------------------ per-H assembly
    compact_keyset = {h: set(map(tuple, tables[h][["area_id", "year", "month"]].to_numpy().tolist())) for h in (0, 3, 6, 12)}
    keys_by_h, horizon_audit = {}, {}
    for h in horizons:
        origin = target_ord - h
        season_block, season_ledger = cp.completed_seasons(seasons, rows["area_id"].to_numpy(), origin)
        cp.assert_season_ledger(season_block, season_ledger, origin)
        history, history_ledger = cp.build_safe_history(observations, rows, h)
        history_check = cp.reference_history_check(observations, history_ledger, history)
        iso = country_by.loc[rows["area_id"].to_numpy(), "iso3"].to_numpy(dtype=object)
        idp, idp_ledger = cp.build_idp_features(idp_obs, rows, iso, h)
        idp_check = cp.reference_idp_check(idp_obs, idp_ledger, idp)
        static_rows = static.reindex(rows["area_id"].to_numpy()).reset_index(drop=True)
        frame = pd.concat([static_rows[groups["static"]], monthly[h], season_block, ident, history, idp], axis=1)
        if list(frame.columns) != list(frozen_order):
            raise ContractError(f"H{h}: assembled column order differs from the frozen compact296 order")
        X = np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
        if np.isinf(X).any():
            raise ContractError(f"H{h}: infinite feature value")
        keys = pd.DataFrame({
            "admin_code": rows["area_id"].to_numpy(np.int64),
            "country_key": country_by.loc[rows["area_id"].to_numpy(), "country_key"].to_numpy(dtype=object),
            "horizon": h,
            "target_ord": target_ord, "target_month": month_label(target_ord),
            "origin_ord": origin, "origin_month": month_label(origin),
            "phase_truth": valid["phase_truth"].to_numpy(np.int64), "crisis_truth": valid["crisis_truth"].to_numpy(np.int64),
            **{q: valid[q].to_numpy(np.float64) for q in targets.TARGETS},
            **{f"{q}_str": valid[f"{q}_str"].to_numpy(dtype=object) for q in targets.TARGETS},
        })
        keys = pd.concat([keys, persistence(valid, keys["admin_code"].to_numpy(np.int64), origin)], axis=1)
        in_compact = np.array([k in compact_keyset[3] for k in map(tuple, rows.to_numpy().tolist())])
        keys["in_old_compact_cohort"] = in_compact.astype(np.int64)
        keys_by_h[h] = keys
        save(f"X296_h{h:02d}.npy", X)
        save(f"keys_h{h:02d}.csv.gz", keys)
        save(f"history_ledger_h{h:02d}.csv.gz", history_ledger)
        save(f"idp_ledger_h{h:02d}.csv.gz", idp_ledger)
        season_ledger.insert(0, "admin_code", rows["area_id"].to_numpy())
        season_ledger.insert(1, "target_ord", target_ord)
        save(f"season_ledger_h{h:02d}.csv.gz", season_ledger)
        miss = pd.DataFrame({"feature": list(frozen_order), "nan_fraction_all": np.isnan(X).mean(axis=0)})
        horizon_audit[str(h)] = {"rows": int(len(keys)), "history_reference_check": history_check, "idp_reference_check": idp_check,
                                 "nan_fraction": float(np.isnan(X).mean()),
                                 "all_nan_columns": [c for c, f in zip(frozen_order, miss["nan_fraction_all"]) if f == 1.0],
                                 "persistence_available": int(keys["persistence_available"].sum()),
                                 "keys_absent_from_old_compact": int((keys["in_old_compact_cohort"] == 0).sum())}
        save(f"feature_missingness_h{h:02d}.csv.gz", miss)
        log(t0, f"H{h}: {X.shape} assembled; history {history_check}; idp {idp_check}")

    # ------------------------------------------------------------------ comparisons
    comparisons = compare_with_compact(keys_by_h, out, tables, frozen_order, groups)
    save("compact_overlap_comparison.csv.gz", comparisons["table"])
    value_check = value_at_origin_check(keys_by_h, out, frozen_order, interim_long, monthly_long)
    save("value_at_origin_check.csv.gz", value_check)
    if (value_check["mismatch"] > 0).any():
        raise ContractError("value_at_origin differs from the direct source lookup at O = T - H")
    p6_keys = compare_p6_keys(keys_by_h, src)
    missing_keys = keys_by_h[3].loc[keys_by_h[3]["in_old_compact_cohort"] == 0, ["admin_code", "target_month"]]
    save("keys_absent_from_old_compact.csv.gz", missing_keys)

    # ------------------------------------------------------------------ calendar, providers, no-fit enumeration
    enum = schedule.enumerate_requests(contract, keys_by_h)
    folds = schedule.fold_calendar(contract, keys_by_h)
    save("schedule_blocks.json", enum["blocks"])
    save("schedule_requests.csv.gz", enum["requests"])
    save("schedule_corrections.csv.gz", enum["corrections"])
    save("fold_calendar.csv.gz", folds)
    for h in horizons:
        prov = pd.concat([keys_by_h[h][["admin_code", "target_month", "origin_month"]], enum["providers"][h]], axis=1)
        save(f"providers_h{h:02d}.csv.gz", prov)
    cohort = cohort_checks(contract, keys_by_h, folds)
    log(t0, f"enumeration {enum['summary']}; cohorts {cohort['main_keys']} / supplement {cohort['supplement_keys']}")

    schema = pd.DataFrame({"position": np.arange(296), "name": list(frozen_order),
                           "group": [g for g, cols in groups.items() for _ in cols]})
    save("feature_order_296.csv.gz", schema)
    result = {
        "stage": "P0-prepare-local-only",
        "contract_version": contract["contract_version"],
        "config_sha256": config_identity(),
        "code_inventory": code_inventory(),
        "availability": AVAILABILITY,
        "source_manifest_sha256": sha256_file(run_dir / "sources" / "source-manifest.json"),
        "ledger": summary,
        "p6_valid_ledger_equal": p6_equal,
        "p6_key_comparison": p6_keys,
        "static29": static_audit,
        "compact296_list_sha256": list_sha256(frozen_order),
        "horizons": horizon_audit,
        "compact_overlap_summary": comparisons["summary"],
        "value_at_origin_check": {"features": int(len(value_check)), "rows_checked": int(value_check["rows_checked"].sum())},
        "schedule_summary": enum["summary"],
        "cohorts": cohort,
        "source_support": {"cutoffs": list(SUPPORT_CUTOFFS), "cells": int(len(support)),
                           "power_changed_cells": int((support["power_changed_rows_origin_gt_cutoff"] > 0).sum())},
        "artifacts_sha256": artifacts,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    write_json(out / "prepared-manifest.json", result)
    write_json(out / "COMPLETE.json", {"stage": STAGE, "prepared_manifest_sha256": sha256_file(out / "prepared-manifest.json")})
    log(t0, "prepare COMPLETE")
    return result


def compare_with_compact(keys_by_h: dict, out: Path, tables: dict, frozen_order, groups: dict) -> dict:
    """Overlap comparison with the inherited compact tables (diagnostic; mismatches stop before fitting).

    H3/H6/H12 must equal the compact table exactly on shared keys. H1 is compared with compact H0 only to show
    it is a distinct origin rebuild: static/identifier columns equal, origin-dependent columns differ.
    """
    rows, summary = [], {}
    for h, ref_h in ((1, 0), (3, 3), (6, 6), (12, 12)):
        keys = keys_by_h[h]
        X = np.load(out / f"X296_h{h:02d}.npy")
        ref = tables[ref_h]
        idx = pd.MultiIndex.from_arrays([ref["area_id"], ref["year"], ref["month"]])
        mine_idx = pd.MultiIndex.from_arrays([keys["admin_code"], keys["target_ord"] // 12, keys["target_ord"] % 12 + 1])
        pos = idx.get_indexer(mine_idx)
        shared = pos >= 0
        R = ref[list(frozen_order)].to_numpy(dtype=np.float64)[pos[shared]]
        M = X[shared]
        same = (M == R) | (np.isnan(M) & np.isnan(R))
        group_of = {c: g for g, cols in groups.items() for c in cols}
        for j, name in enumerate(frozen_order):
            both = ~np.isnan(M[:, j]) & ~np.isnan(R[:, j])
            rows.append({"horizon": h, "reference": f"compact_baseline_h{ref_h}", "feature": name, "group": group_of[name],
                         "shared_rows": int(shared.sum()), "unequal_rows": int((~same[:, j]).sum()),
                         "max_abs_diff": float(np.max(np.abs(M[both, j] - R[both, j]))) if both.any() else 0.0,
                         "mine_nan_ref_value": int((np.isnan(M[:, j]) & ~np.isnan(R[:, j])).sum()),
                         "mine_value_ref_nan": int((~np.isnan(M[:, j]) & np.isnan(R[:, j])).sum())})
        unequal = int((~same).any(axis=0).sum())
        summary[str(h)] = {"reference": f"compact_baseline_h{ref_h}", "shared_rows": int(shared.sum()),
                           "rows_not_in_reference": int((~shared).sum()), "features_with_any_unequal_cell": unequal}
        if h != 1 and unequal:
            bad = [r["feature"] for r in rows if r["horizon"] == h and r["unequal_rows"]]
            raise ContractError(f"H{h}: rebuilt compact296 differs from the inherited table on shared keys for {len(bad)} features: {bad[:8]}")
        if h == 1:
            stable = groups["static"] + groups["identifier"]
            js = [list(frozen_order).index(c) for c in stable]
            if (~same[:, js]).any():
                raise ContractError("H1 static/identifier columns differ from the shared-key compact rows")
            summary["1"]["origin_dependent_features_differing_from_H0"] = unequal
    return {"table": pd.DataFrame(rows), "summary": summary}


def value_at_origin_check(keys_by_h: dict, out: Path, frozen_order, interim_long: pd.DataFrame,
                         monthly_long: pd.DataFrame) -> pd.DataFrame:
    """Independent direct lookup: every ordinary ``<source>__value_at_origin`` equals the source cell at (admin, O).

    Uses the long source tables and a keyed merge, not the Grid/gather path that built the matrix, so a wrong
    horizon (e.g. H0 values standing in for H1) or a shifted origin is detected row by row.
    """
    order = list(frozen_order)
    rows = []
    for h, keys in keys_by_h.items():
        X = np.load(out / f"X296_h{h:02d}.npy", mmap_mode="r")
        origin = keys["origin_ord"].to_numpy(np.int64)
        probe = pd.DataFrame({"area_id": keys["admin_code"].to_numpy(np.int64), "year": origin // 12, "month": origin % 12 + 1})
        for long, sources in ((interim_long, cp.INTERIM_SOURCES[:7]), (monthly_long, cp.CLIMATE_SOURCES)):
            merged = probe.merge(long[["area_id", "year", "month", *sources]], on=["area_id", "year", "month"], how="left",
                                 validate="many_to_one")
            for s in sources:
                direct = pd.to_numeric(merged[s], errors="coerce").to_numpy(dtype=float)
                built = np.asarray(X[:, order.index(f"{s}__value_at_origin")], dtype=float)
                same = (direct == built) | (np.isnan(direct) & np.isnan(built))
                rows.append({"horizon": h, "feature": f"{s}__value_at_origin", "rows_checked": int(len(direct)),
                             "finite_direct": int(np.isfinite(direct).sum()), "mismatch": int((~same).sum())})
    return pd.DataFrame(rows)


def compare_p6_keys(keys_by_h: dict, src: dict) -> dict:
    out = {}
    for h, keys in keys_by_h.items():
        p6 = pd.read_csv(src[f"p6_keys_h{h:02d}"], usecols=["admin_code", "target_month", "origin_month", "country_key", "phase_truth"])
        a = keys[["admin_code", "target_month", "origin_month", "country_key", "phase_truth"]].astype(str).reset_index(drop=True)
        b = p6.sort_values(["admin_code", "target_month"], kind="mergesort").astype(str).reset_index(drop=True)
        equal = bool(a.equals(b))
        if not equal:
            raise ContractError(f"H{h}: canonical keys/origins/countries/phases differ from the saved P6 key table")
        out[str(h)] = {"rows": int(len(a)), "equal_to_p6_keys": equal}
    return out


def cohort_checks(contract: dict, keys_by_h: dict, folds: pd.DataFrame) -> dict:
    cal = contract["calendar"]
    main_first_origin = parse_month(cal["main_first_origin"])
    main_last = parse_month(cal["main_last_target"])
    supp = [parse_month(m) for m in cal["supplement_target_months"]]
    main_keys, supp_keys = {}, {}
    for h, keys in keys_by_h.items():
        t, o = keys["target_ord"].to_numpy(), keys["origin_ord"].to_numpy()
        main_keys[str(h)] = int(((o >= main_first_origin) & (t <= main_last)).sum())
        supp_keys[str(h)] = int(np.isin(t, supp).sum())
    if main_keys != {k: v for k, v in cal["main_expected_keys"].items()}:
        raise ContractError(f"main cohorts {main_keys} differ from {cal['main_expected_keys']}")
    if set(supp_keys.values()) != {cal["supplement_expected_keys_per_h"]}:
        raise ContractError(f"supplement cohorts {supp_keys} differ from {cal['supplement_expected_keys_per_h']} per H")
    later = {str(h): int((k["target_ord"].to_numpy() > supp[-1]).sum()) for h, k in keys_by_h.items()}
    main = folds[folds["period"] == "main"]
    return {"main_keys": main_keys, "supplement_keys": supp_keys, "valid_keys_after_supplement": later,
            "main_folds": int(len(main)), "main_nonempty_folds": int((main["eval_keys"] > 0).sum()),
            "main_empty_fold_months": sorted(set(main.loc[main["eval_keys"] == 0, "target_month"])),
            "supplement_folds": int((folds["period"] == "supplement").sum())}
