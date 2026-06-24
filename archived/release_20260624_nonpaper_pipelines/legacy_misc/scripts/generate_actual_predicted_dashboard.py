#!/usr/bin/env python3
"""Generate a static actual-vs-predicted crisis dashboard."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon


DEFAULT_INPUT_DIR = Path("main_ablation_results/march2026_main_backup_month_ind_cont3")
DEFAULT_SHAPEFILE = Path(
    r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp"
)
DEFAULT_OUTPUT_SUBDIR = "actual_predicted_dashboard"
OUTPUT_HTML_NAME = "actual_predicted_dashboard.html"
OUTPUT_MANIFEST_NAME = "manifest.json"

INCLUDED_MODELS = {
    "GF": "GeoRF",
    "DT": "GeoDT",
}
EXCLUDED_MODEL_TOKENS = {"XGB"}
SCOPES = ("fs1", "fs2", "fs3")
REQUIRED_COLUMNS = ("FEWSNET_admin_code", "month_start", "y_true", "y_pred_partitioned")
OPTIONAL_COLUMNS = ("partition_id",)
ADMIN_CODE_ALIASES = ("FEWSNET_admin_code", "admin_code", "adm_code", "FNID")
RESULT_DIR_RE = re.compile(r"^result_partition_k40_compare_([A-Za-z0-9]+)_fs([0-9]+)$")

CLASS_COLORS = {
    0: "#2ca02c",
    1: "#d62728",
    "no_data": "#e0e0e0",
    "border": "#ffffff",
}
SVG_WIDTH = 1000
SVG_HEIGHT = 650
SIMPLIFY_TOLERANCE = 0.02


class DashboardError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an exploratory static dashboard comparing actual and predicted crisis maps."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--simplify-tolerance", type=float, default=SIMPLIFY_TOLERANCE)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    text = str(path)
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", text):
        drive = text[0].lower()
        rest = text[2:].replace("\\", "/").lstrip("/")
        return Path("/mnt") / drive / rest
    return path


def as_display_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def normalize_admin_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_month(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise DashboardError(f"Found {bad_count} unparseable month_start values")
    return parsed.dt.strftime("%Y-%m-%d")


def discover_prediction_sources(input_dir: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    if not input_dir.exists():
        raise DashboardError(f"Input directory not found: {input_dir}")

    included: list[dict[str, Any]] = []
    excluded: list[str] = []
    warnings: list[str] = []

    for result_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        match = RESULT_DIR_RE.match(result_dir.name)
        if not match:
            continue
        token, scope_number = match.groups()
        scope = f"fs{scope_number}"
        pred_file = result_dir / "predictions_monthly.csv"
        if token in EXCLUDED_MODEL_TOKENS:
            excluded.append(str(pred_file if pred_file.exists() else result_dir))
            continue
        if token not in INCLUDED_MODELS or scope not in SCOPES:
            warnings.append(f"Ignored unsupported result folder: {result_dir}")
            continue
        if not pred_file.exists():
            warnings.append(f"Missing predictions_monthly.csv for {result_dir.name}")
            continue
        included.append(
            {
                "model": INCLUDED_MODELS[token],
                "model_token": token,
                "scope": scope,
                "path": pred_file,
            }
        )

    if not included:
        raise DashboardError(f"No GeoRF or GeoDT prediction files found under {input_dir}")
    return included, excluded, warnings


def validate_binary_values(df: pd.DataFrame, column: str, source: Path) -> None:
    values = set(pd.Series(df[column]).dropna().astype(int).unique().tolist())
    if not values.issubset({0, 1}):
        raise DashboardError(f"Unexpected values in {column} for {source}: {sorted(values)}")


def load_prediction_datasets(
    sources: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, dict[str, int]]]], list[dict[str, Any]], list[str]]:
    datasets: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    included_sources: list[dict[str, Any]] = []
    warnings: list[str] = []

    for source in sources:
        path = source["path"]
        df = pd.read_csv(path, dtype={"FEWSNET_admin_code": "string"})
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            warnings.append(f"Omitted {path}: missing required columns {missing}")
            continue

        df = df[list(REQUIRED_COLUMNS) + [col for col in OPTIONAL_COLUMNS if col in df.columns]].copy()
        df["FEWSNET_admin_code"] = normalize_admin_code(df["FEWSNET_admin_code"])
        df["month_start"] = normalize_month(df["month_start"])
        for col in ("y_true", "y_pred_partitioned"):
            if df[col].isna().any():
                raise DashboardError(f"Missing values in {col} for {path}")
            validate_binary_values(df, col, path)
            df[col] = df[col].astype(int)

        duplicate_mask = df.duplicated(["FEWSNET_admin_code", "month_start"], keep="first")
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count:
            warnings.append(f"{path} had {duplicate_count} duplicate polygon-month rows; kept first")
            df = df[~duplicate_mask].copy()

        key_prefix = f"{source['model']}|{source['scope']}"
        available_dates = sorted(df["month_start"].unique().tolist())
        for month, month_df in df.groupby("month_start", sort=True):
            key = f"{key_prefix}|{month}"
            datasets[key] = {
                row["FEWSNET_admin_code"]: {
                    "actual": int(row["y_true"]),
                    "predicted": int(row["y_pred_partitioned"]),
                }
                for _, row in month_df.iterrows()
            }

        included_sources.append(
            {
                "model": source["model"],
                "model_token": source["model_token"],
                "scope": source["scope"],
                "path": str(path),
                "row_count": int(len(df)),
                "available_dates": available_dates,
                "required_columns_present": True,
            }
        )

    if not datasets:
        raise DashboardError("No valid prediction datasets remained after validation")
    return datasets, included_sources, warnings


def load_global_shapefile(shapefile_path: Path, simplify_tolerance: float) -> tuple[list[dict[str, str]], set[str], list[str]]:
    resolved = resolve_path(shapefile_path)
    if not resolved.exists():
        raise DashboardError(f"Shapefile not found: {resolved}")

    gdf = gpd.read_file(resolved)
    found_col = next((col for col in ADMIN_CODE_ALIASES if col in gdf.columns), None)
    if found_col is None:
        raise DashboardError(f"No admin code column found in shapefile. Tried {ADMIN_CODE_ALIASES}")
    if found_col != "FEWSNET_admin_code":
        gdf = gdf.rename(columns={found_col: "FEWSNET_admin_code"})

    warnings: list[str] = []
    invalid_count = int((~gdf.geometry.is_valid).sum())
    if invalid_count:
        warnings.append(f"Fixed {invalid_count} invalid shapefile geometries with buffer(0)")
        gdf["geometry"] = gdf.geometry.buffer(0)

    if gdf.crs is None:
        warnings.append("Shapefile CRS missing; assuming EPSG:4326 for dashboard rendering")
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    gdf["FEWSNET_admin_code"] = normalize_admin_code(gdf["FEWSNET_admin_code"])
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if simplify_tolerance > 0:
        gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)

    bounds = gdf.total_bounds
    features: list[dict[str, str]] = []
    codes: set[str] = set()
    for _, row in gdf.iterrows():
        code = str(row["FEWSNET_admin_code"])
        path = geometry_to_svg_path(row.geometry, bounds)
        if not path:
            continue
        features.append({"code": code, "path": path})
        codes.add(code)

    if not features:
        raise DashboardError("No shapefile features could be converted for dashboard rendering")
    return features, codes, warnings


def transform_xy(x: float, y: float, bounds: Any) -> tuple[float, float]:
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1e-9)
    height = max(maxy - miny, 1e-9)
    scale = min((SVG_WIDTH - 40) / width, (SVG_HEIGHT - 40) / height)
    drawn_w = width * scale
    drawn_h = height * scale
    offset_x = (SVG_WIDTH - drawn_w) / 2
    offset_y = (SVG_HEIGHT - drawn_h) / 2
    sx = offset_x + (x - minx) * scale
    sy = offset_y + (maxy - y) * scale
    return sx, sy


def ring_to_path(coords: Any, bounds: Any) -> str:
    parts: list[str] = []
    for idx, (x, y) in enumerate(coords):
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        sx, sy = transform_xy(x, y, bounds)
        command = "M" if idx == 0 else "L"
        parts.append(f"{command}{sx:.2f},{sy:.2f}")
    if parts:
        parts.append("Z")
    return " ".join(parts)


def polygon_to_path(poly: Polygon, bounds: Any) -> str:
    if poly.is_empty:
        return ""
    parts = [ring_to_path(poly.exterior.coords, bounds)]
    parts.extend(ring_to_path(ring.coords, bounds) for ring in poly.interiors)
    return " ".join(part for part in parts if part)


def geometry_to_svg_path(geometry: Any, bounds: Any) -> str:
    if isinstance(geometry, Polygon):
        return polygon_to_path(geometry, bounds)
    if isinstance(geometry, MultiPolygon):
        return " ".join(polygon_to_path(poly, bounds) for poly in geometry.geoms)
    return ""


def build_availability(datasets: dict[str, dict[str, dict[str, int]]]) -> dict[str, Any]:
    models: set[str] = set()
    scopes_by_model: dict[str, set[str]] = {}
    dates_by_model_scope: dict[str, set[str]] = {}

    for key in datasets:
        model, scope, month = key.split("|")
        models.add(model)
        scopes_by_model.setdefault(model, set()).add(scope)
        dates_by_model_scope.setdefault(f"{model}|{scope}", set()).add(month)

    return {
        "models": sorted(models),
        "scopesByModel": {model: sorted(scopes) for model, scopes in sorted(scopes_by_model.items())},
        "datesByModelScope": {key: sorted(dates) for key, dates in sorted(dates_by_model_scope.items())},
    }


def build_smoke_test(datasets: dict[str, dict[str, dict[str, int]]], geometry_codes: set[str]) -> dict[str, Any]:
    preferred_key = "GeoRF|fs2|2024-06-01"
    keys = [preferred_key] if preferred_key in datasets else []
    keys.extend(key for key in sorted(datasets) if key not in keys)

    for key in keys:
        model, scope, month = key.split("|")
        records = datasets[key]
        join_rows = len(set(records) & geometry_codes)
        if join_rows > 0:
            return {"model": model, "scope": scope, "date": month, "join_rows": join_rows, "passed": True}
    first_key = sorted(datasets)[0]
    model, scope, month = first_key.split("|")
    return {"model": model, "scope": scope, "date": month, "join_rows": 0, "passed": False}


def validate_output_dir(output_dir: Path, input_dir: Path) -> Path:
    resolved = output_dir.resolve()
    forbidden_names = {"deliverables", "prediction_pipeline"}
    if any(part in forbidden_names for part in resolved.parts):
        raise DashboardError(f"Refusing to write exploratory dashboard under protected directory: {resolved}")
    if resolved.suffix.lower() in {".csv", ".xlsx", ".shp"}:
        raise DashboardError(f"Output directory cannot be a source/data file path: {resolved}")
    source_root = input_dir.resolve()
    if not str(resolved).startswith(str(source_root)):
        print(f"WARNING: output directory is outside input result directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def make_manifest(
    input_dir: Path,
    shapefile_path: Path,
    included_sources: list[dict[str, Any]],
    excluded_sources: list[str],
    availability: dict[str, Any],
    output_html: Path,
    smoke_test: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    available_scopes = sorted({scope for scopes in availability["scopesByModel"].values() for scope in scopes})
    available_dates = sorted({date for dates in availability["datesByModelScope"].values() for date in dates})
    return {
        "workflow_mode": "exploratory diagnostics/tooling",
        "production_status": "exploratory",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_directory": str(input_dir),
        "included_sources": included_sources,
        "excluded_sources": excluded_sources,
        "shapefile_path": str(shapefile_path),
        "join_key": "FEWSNET_admin_code",
        "available_models": availability["models"],
        "available_scopes": available_scopes,
        "available_dates": available_dates,
        "label_contract": {
            "actual_field": "y_true",
            "predicted_field": "y_pred_partitioned",
            "non_crisis_value": 0,
            "crisis_value": 1,
        },
        "threshold_contract": "none; existing binary labels only",
        "output_html": str(output_html),
        "optional_assets": [],
        "smoke_test": smoke_test,
        "omissions_or_warnings": warnings,
    }


def json_for_script(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def render_html(payload: dict[str, Any]) -> str:
    data_json = json_for_script(payload)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Actual vs Predicted Crisis Dashboard</title>
<style>
:root {{ --crisis: {CLASS_COLORS[1]}; --noncrisis: {CLASS_COLORS[0]}; --nodata: {CLASS_COLORS['no_data']}; --border: #ffffff; }}
body {{ margin: 0; font-family: Arial, sans-serif; color: #222; background: #f6f7f9; }}
header {{ padding: 16px 20px; background: #1f2937; color: white; }}
header h1 {{ margin: 0 0 6px; font-size: 22px; }}
header p {{ margin: 0; font-size: 13px; color: #d1d5db; }}
.controls {{ display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 12px; padding: 14px 20px; background: white; border-bottom: 1px solid #d7dce2; }}
.control label {{ display: block; font-size: 12px; font-weight: bold; margin-bottom: 4px; }}
select, input[type=range] {{ width: 100%; }}
.status {{ min-height: 20px; padding: 0 20px 12px; background: white; color: #7f1d1d; font-weight: bold; }}
.panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px 20px 8px; }}
.panel {{ background: white; border: 1px solid #d7dce2; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.panel h2 {{ margin: 0; padding: 12px 14px 4px; font-size: 18px; }}
.panel .subtitle {{ padding: 0 14px 8px; font-size: 12px; color: #4b5563; }}
svg {{ display: block; width: 100%; height: auto; background: #eef2f7; }}
.base path, .overlay path {{ stroke: var(--border); stroke-width: .35; vector-effect: non-scaling-stroke; }}
.summary {{ padding: 10px 14px 14px; font-size: 13px; }}
.legend {{ display: flex; gap: 18px; padding: 8px 20px 18px; font-size: 13px; align-items: center; }}
.swatch {{ display: inline-block; width: 14px; height: 14px; border: 1px solid #999; vertical-align: -2px; margin-right: 5px; }}
footer {{ padding: 0 20px 16px; font-size: 12px; color: #555; }}
@media (max-width: 900px) {{ .controls, .panels {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>Actual vs Partitioned Predicted Food Crisis</h1>
  <p>Exploratory diagnostics using existing GeoRF/GeoDT month-ind outputs and the global FEWSNET shapefile.</p>
</header>
<section class=\"controls\">
  <div class=\"control\"><label for=\"modelSelect\">Model</label><select id=\"modelSelect\"></select></div>
  <div class=\"control\"><label for=\"scopeSelect\">Forecasting scope</label><select id=\"scopeSelect\"></select></div>
  <div class=\"control\"><label for=\"dateSelect\">Test month</label><select id=\"dateSelect\"></select></div>
  <div class=\"control\"><label for=\"leftAlpha\">Left prediction alpha: <span id=\"leftAlphaValue\">0.00</span></label><input id=\"leftAlpha\" type=\"range\" min=\"0\" max=\"1\" step=\"0.05\" value=\"0\"></div>
  <div class=\"control\"><label for=\"rightAlpha\">Right prediction alpha: <span id=\"rightAlphaValue\">1.00</span></label><input id=\"rightAlpha\" type=\"range\" min=\"0\" max=\"1\" step=\"0.05\" value=\"1\"></div>
</section>
<div id=\"status\" class=\"status\"></div>
<section class=\"panels\">
  <article class=\"panel\">
    <h2>Actual crisis distribution</h2>
    <div class=\"subtitle\">Base layer: actual labels (y_true). Prediction overlay defaults to alpha 0.</div>
    <svg id=\"actualMap\" viewBox=\"0 0 {SVG_WIDTH} {SVG_HEIGHT}\" role=\"img\" aria-label=\"Actual crisis map\"></svg>
    <div id=\"actualSummary\" class=\"summary\"></div>
  </article>
  <article class=\"panel\">
    <h2>Predicted crisis distribution</h2>
    <div class=\"subtitle\">Base layer: actual labels (y_true). Prediction overlay defaults to alpha 1.</div>
    <svg id=\"predictedMap\" viewBox=\"0 0 {SVG_WIDTH} {SVG_HEIGHT}\" role=\"img\" aria-label=\"Predicted crisis map\"></svg>
    <div id=\"predictedSummary\" class=\"summary\"></div>
  </article>
</section>
<div class=\"legend\">
  <span><span class=\"swatch\" style=\"background: var(--noncrisis)\"></span>Non-crisis (0)</span>
  <span><span class=\"swatch\" style=\"background: var(--crisis)\"></span>Crisis (1)</span>
  <span><span class=\"swatch\" style=\"background: var(--nodata)\"></span>No data</span>
</div>
<footer id=\"provenance\"></footer>
<script>
const DASHBOARD_DATA = {data_json};
const COLORS = {{0: DASHBOARD_DATA.colors[0], 1: DASHBOARD_DATA.colors[1], no_data: DASHBOARD_DATA.colors.no_data}};
const modelSelect = document.getElementById('modelSelect');
const scopeSelect = document.getElementById('scopeSelect');
const dateSelect = document.getElementById('dateSelect');
const statusEl = document.getElementById('status');
const leftAlpha = document.getElementById('leftAlpha');
const rightAlpha = document.getElementById('rightAlpha');
const leftAlphaValue = document.getElementById('leftAlphaValue');
const rightAlphaValue = document.getElementById('rightAlphaValue');

function option(value, text) {{
  const opt = document.createElement('option');
  opt.value = value;
  opt.textContent = text;
  return opt;
}}

function fillSelect(select, values) {{
  select.textContent = '';
  values.forEach(value => select.appendChild(option(value, value)));
}}

function setupSelectors() {{
  fillSelect(modelSelect, DASHBOARD_DATA.availability.models);
  updateScopes();
}}

function updateScopes() {{
  const model = modelSelect.value;
  fillSelect(scopeSelect, DASHBOARD_DATA.availability.scopesByModel[model] || []);
  updateDates();
}}

function updateDates() {{
  const key = `${{modelSelect.value}}|${{scopeSelect.value}}`;
  fillSelect(dateSelect, DASHBOARD_DATA.availability.datesByModelScope[key] || []);
  renderAll();
}}

function classColor(value) {{
  return value === 0 || value === 1 ? COLORS[value] : COLORS.no_data;
}}

function buildLayer(records, field, opacity) {{
  const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  group.setAttribute('class', field === 'actual' ? 'base' : 'overlay');
  group.setAttribute('opacity', opacity);
  for (const feature of DASHBOARD_DATA.features) {{
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const rec = records ? records[feature.code] : null;
    const value = rec ? rec[field] : null;
    path.setAttribute('d', feature.path);
    path.setAttribute('fill', classColor(value));
    path.setAttribute('fill-rule', 'evenodd');
    path.dataset.code = feature.code;
    group.appendChild(path);
  }}
  return group;
}}

function renderMap(svg, records, predOpacity) {{
  svg.textContent = '';
  svg.appendChild(buildLayer(records, 'actual', 1));
  svg.appendChild(buildLayer(records, 'predicted', predOpacity));
}}

function summarize(records) {{
  if (!records) return 'No data available for this selection.';
  const values = Object.values(records);
  const matched = values.length;
  const actualCrisis = values.filter(row => row.actual === 1).length;
  const predictedCrisis = values.filter(row => row.predicted === 1).length;
  const noData = DASHBOARD_DATA.features.length - matched;
  return `Matched polygons: ${{matched.toLocaleString()}} | Actual crisis: ${{actualCrisis.toLocaleString()}} | Predicted crisis: ${{predictedCrisis.toLocaleString()}} | No data polygons: ${{noData.toLocaleString()}}`;
}}

function renderAll() {{
  leftAlphaValue.textContent = Number(leftAlpha.value).toFixed(2);
  rightAlphaValue.textContent = Number(rightAlpha.value).toFixed(2);
  const key = `${{modelSelect.value}}|${{scopeSelect.value}}|${{dateSelect.value}}`;
  const records = DASHBOARD_DATA.datasets[key];
  if (!records) {{
    statusEl.textContent = 'No data available for the selected model, scope, and date.';
  }} else {{
    statusEl.textContent = '';
  }}
  renderMap(document.getElementById('actualMap'), records, Number(leftAlpha.value));
  renderMap(document.getElementById('predictedMap'), records, Number(rightAlpha.value));
  const summary = summarize(records);
  document.getElementById('actualSummary').textContent = summary;
  document.getElementById('predictedSummary').textContent = summary;
}}

modelSelect.addEventListener('change', updateScopes);
scopeSelect.addEventListener('change', updateDates);
dateSelect.addEventListener('change', renderAll);
leftAlpha.addEventListener('input', renderAll);
rightAlpha.addEventListener('input', renderAll);

document.getElementById('provenance').textContent = `Generated from ${{DASHBOARD_DATA.manifest.source_directory}} | Shapefile: ${{DASHBOARD_DATA.manifest.shapefile_path}} | Threshold: ${{DASHBOARD_DATA.manifest.threshold_contract}}`;
setupSelectors();
</script>
</body>
</html>
"""


def write_dashboard(output_dir: Path, payload: dict[str, Any], manifest: dict[str, Any]) -> tuple[Path, Path]:
    html_path = output_dir / OUTPUT_HTML_NAME
    manifest_path = output_dir / OUTPUT_MANIFEST_NAME
    html_path.write_text(render_html(payload), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return html_path, manifest_path


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    shapefile_path = resolve_path(args.shapefile)
    output_dir_arg = args.output_dir if args.output_dir is not None else input_dir / DEFAULT_OUTPUT_SUBDIR
    output_dir = validate_output_dir(resolve_path(output_dir_arg), input_dir)

    print("Generating actual-vs-predicted crisis dashboard")
    print(f"Input directory: {input_dir}")
    print(f"Shapefile: {shapefile_path}")
    print(f"Output directory: {output_dir}")

    sources, excluded_sources, warnings = discover_prediction_sources(input_dir)
    datasets, included_sources, dataset_warnings = load_prediction_datasets(sources)
    warnings.extend(dataset_warnings)
    features, geometry_codes, geometry_warnings = load_global_shapefile(shapefile_path, args.simplify_tolerance)
    warnings.extend(geometry_warnings)

    availability = build_availability(datasets)
    smoke_test = build_smoke_test(datasets, geometry_codes)
    if not smoke_test["passed"]:
        raise DashboardError("Smoke-test shapefile join failed for all available selections")

    html_path = output_dir / OUTPUT_HTML_NAME
    manifest = make_manifest(
        input_dir=input_dir,
        shapefile_path=shapefile_path,
        included_sources=included_sources,
        excluded_sources=excluded_sources,
        availability=availability,
        output_html=html_path,
        smoke_test=smoke_test,
        warnings=warnings,
    )
    payload = {
        "features": features,
        "datasets": datasets,
        "availability": availability,
        "colors": CLASS_COLORS,
        "manifest": manifest,
    }
    html_path, manifest_path = write_dashboard(output_dir, payload, manifest)

    print(f"Included sources: {len(included_sources)}")
    print(f"Excluded XGB sources: {len(excluded_sources)}")
    print(f"Feature paths: {len(features)}")
    print(f"Smoke test: {smoke_test}")
    print(f"HTML: {html_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DashboardError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
