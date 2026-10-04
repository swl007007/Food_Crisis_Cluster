"""Task 10-03 driver: confirmatory pooled-vs-persistence onset test (pre-registered; see prd/design/implement).

Usage (pinned Windows Python, from FEWSNETGeoXGBExperiment/):
    python -B <this> --run-dir RUN STEP      STEP in pin|alignment|extension|inputs|predict|truth|evaluate|check

Every step writes ``RUN/steps/<step>.json`` last and refuses to run twice or out of order. Product code is
imported, never modified. No 2026-02 CS value is read before ``truth``, which requires the prediction
freeze record. The ``check`` step imports no package code.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
TASK = HERE.parents[1]
REPO = HERE.parents[4]
PKG = REPO / "FEWSNETGeoXGBExperiment"
ARCH = REPO / ".trellis/tasks/archive/2026-10/10-02-exogenous-transition-forecast-design"
G = Path(r"C:\Users\swl00\geoxgb_runs")
RUN10 = G / "scen-b43ef6a-v1"
TRUTH_OCT = G / "scen-b43ef6a-v1.truth-release-v2"
SRC = Path(r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data")
PINNED = SRC / "FEWSNET_forecast_unadjusted_bm.csv"
COMBINED = SRC / "assembled_FEWSNET" / "FEWSNET_forecast_unadjusted_bm_2025_combined.csv"
RAW26 = SRC / "Outcome" / "FEWSNET_IPC" / "2025_2026_FEWSNET.csv"
FEWS = SRC / "Outcome" / "FEWSNET_IPC" / "FEWSNET.csv"
SHP = SRC / "Outcome" / "FEWSNET_IPC" / "FEWS NET Admin Boundaries" / "FEWS_Admin_LZ_v3.shp"
CODE_ID = "fd25e2f7b30d9be34273e95b2b69b387c2fa0892e153244c5bdfc7dba0f05b94"
ORDER = ["pin", "alignment", "extension", "inputs", "predict", "truth", "evaluate", "check"]
H, ORIGIN, TARGET = 4, "2025-10", "2026-02"
SCEN = {"S3": {"gate_k": 3, "oct_visible": False, "role": "primary"},
        "S0": {"gate_k": 0, "oct_visible": True, "role": "secondary"}}
EXT_MONTHS = ["2024-12"] + [f"2025-{m:02d}" for m in range(1, 11)]
sys.path.insert(0, str(PKG))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mi(s):
    return int(s[:4]) * 12 + int(s[5:7]) - 1


def ml(x):
    return f"{int(x) // 12}-{int(x) % 12 + 1:02d}"


def done(run, step, record):
    (run / "steps").mkdir(parents=True, exist_ok=True)
    path = run / "steps" / f"{step}.json"
    with open(path, "x", encoding="utf-8") as f:
        json.dump(record, f, indent=1, default=str)
    print(json.dumps({"step": step, **{k: v for k, v in record.items() if not isinstance(v, (dict, list)) or len(str(v)) < 400}},
                     indent=1, default=str))


def require(run, step):
    i = ORDER.index(step)
    if (run / "steps" / f"{step}.json").exists():
        raise SystemExit(f"step {step} already done; refusing to rerun")
    for prev in ORDER[:i]:
        if not (run / "steps" / f"{prev}.json").exists():
            raise SystemExit(f"step {step} requires {prev} first")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------- pin
def step_pin(run):
    if run.exists():
        raise SystemExit(f"{run} exists; refusing to overwrite")
    from src.utils import run_identity as rid
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "FEWSNETGeoXGBExperiment"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    code = rid.code_identity()
    if code["sha256"] != CODE_ID or dirty:
        raise SystemExit(f"package not at the pinned identity or dirty: {code} {dirty[:200]}")
    run.mkdir(parents=True)
    files = {
        "observations_10_02": RUN10 / "prepared/ledgers/observations.csv",
        "release_ledger_10_02": RUN10 / "prepared/manifests/release_ledger.csv",
        "alignment_10_02": RUN10 / "prepared/manifests/alignment.json",
        "sources_10_02": RUN10 / "prepared/manifests/sources.json",
        "frozen_10_02": RUN10 / "scenario_final/frozen.json",
        "truth_oct2025_v2": TRUTH_OCT / "truth_oct2025.csv",
        "release_oct2025_v2": TRUTH_OCT / "release.json",
        "crosswalk_oct2025_v2": TRUTH_OCT / "crosswalk_oct2025.csv",
        "pinned_panel": PINNED, "combined_panel": COMBINED,
        "raw_2025_2026_cs_PROTECTED_hash_only": RAW26, "fewsnet_history": FEWS,
        "shapefile_dbf": SHP.with_suffix(".dbf"), "shapefile_shp": SHP,
        "feature_schema": PKG / "feature-schema.json",
        "prd": TASK / "prd.md", "design": TASK / "design.md", "implement": TASK / "implement.md", "driver": HERE}
    frozen = load(files["frozen_10_02"])
    mid = frozen["recipe"]["4"]["map_id"]
    cons = load(RUN10 / "scenario_maps" / mid / "consensus.json")
    files["map_consensus"] = RUN10 / "scenario_maps" / mid / "consensus.json"
    files["map_clusters"] = RUN10 / "scenario_maps" / mid / cons["cluster_map"]
    runtime = rid.runtime_identity()
    sel = load(RUN10 / "scenario_development/selection.json")
    if runtime != sel["runtime"]:
        raise SystemExit(f"runtime differs from the pinned 10-02 stack: {runtime}")
    man = {"git_head": head, "code_identity": code, "runtime": runtime, "recipe_h4": frozen["recipe"]["4"],
           "files": {k: {"path": str(v), "sha256": sha(v)} for k, v in files.items()},
           "blinding": "raw 2025-2026 CS file hashed as bytes only; no value read before the truth step"}
    with open(run / "inputs_manifest.json", "x", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    done(run, "pin", {"inputs_manifest_sha256": sha(run / "inputs_manifest.json"), "git_head": head,
                      "code_identity": code["sha256"]})


# -------------------------------------------------------------------------------------- alignment
def acled_sources(align):
    return [k for k, v in align.items() if v["kind"] == "monthly" and not k.endswith("_tavg_mean")]


def step_alignment(run):
    from src.feature import fourclass_features as ff
    schema = load(PKG / "feature-schema.json")
    a = load(RUN10 / "prepared/manifests/alignment.json")
    acled = acled_sources(a)
    assert len(acled) == 19, acled
    for s in acled:
        a[s] = {"kind": "excluded", "status": "reconstructed",
                "evidence": "10-03 D3: excluded by availability (all ACLED missing for every area at 2025-09/10); fixed before evaluation"}
    ff.check_alignment(schema, a, real=True)
    names = ff.aligned_feature_names(schema, a)
    path = run / "alignment_v3.json"
    with open(path, "x", encoding="utf-8") as f:
        json.dump(a, f, indent=1)
    done(run, "alignment", {"alignment_v3_sha256": sha(path), "acled_excluded": acled, "features": len(names),
                            "features_10_02": len(ff.aligned_feature_names(schema, load(RUN10 / "prepared/manifests/alignment.json")))})


# -------------------------------------------------------------------------------------- extension
def step_extension(run):
    import numpy as np
    import pandas as pd
    from scripts import run_experiment as rx
    from src.feature import fourclass_features as ff
    schema = load(PKG / "feature-schema.json")
    sources = list(schema["static_sources"] + schema["dynamic_sources_at_origin"])
    a3 = load(run / "alignment_v3.json")
    a10 = load(RUN10 / "prepared/manifests/alignment.json")
    admitted10 = [s for s in sources if a10[s]["kind"] != "excluded"]
    before = {"pinned": sha(PINNED), "combined": sha(COMBINED)}
    rows, seen, dedup = [], {}, []
    for path, months in ((PINNED, {"2024-12"}), (COMBINED, set(EXT_MONTHS[1:]))):
        with open(path, newline="", encoding="utf-8") as f:
            for d in csv.DictReader(f):
                ym = d["date"][:7]
                if ym not in months:
                    continue
                key = (d["FEWSNET_admin_code"], ym)
                vals = [d[s] for s in sources]
                if key in seen:
                    prev = seen[key]
                    diff = [s for s, x, y in zip(sources, prev, vals) if x != y]
                    if set(diff) & set(admitted10):
                        raise SystemExit(f"duplicate {key} differs on admitted sources {diff}")
                    dedup.append({"key": key, "differing_columns": diff, "kept": "first occurrence"})
                    continue
                seen[key] = vals
                rows.append([d["FEWSNET_admin_code"], ym] + vals)
    ext_dir = run / "extension_v3"
    ext_dir.mkdir()
    csv_path = ext_dir / "covariate_extension_2024-12_2025-10.csv"
    with open(csv_path, "x", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["FEWSNET_admin_code", "date"] + sources)
        w.writerows(rows)
    if {"pinned": sha(PINNED), "combined": sha(COMBINED)} != before:
        raise SystemExit("raw bytes changed during assembly")
    per_month = pd.Series([r[1] for r in rows]).value_counts().sort_index().to_dict()
    if any(n != 5718 for n in per_month.values()) or set(per_month) != set(EXT_MONTHS):
        raise SystemExit(f"extension months/areas {per_month}")
    manifest = {"path": str(csv_path), "sha256": sha(csv_path), "first_month": "2024-12", "last_month": "2025-10",
                "overlap_months": ["2024-12"],
                "source": ("EXPLICIT TWO-SOURCE SPLICE (task 10-03): 2024-12 copied from the PINNED panel; 2025-01..2025-10 "
                           "from the COMBINED panel; original value strings, date normalised to YYYY-MM; keys + 69 schema "
                           "sources. The load_extension overlap check is TRUE BY CONSTRUCTION. Admitted 10-02 sources agree "
                           "with the pinned panel within rtol 0/atol 1e-9 on all 2010-2024 keys (10-02 evidence); historical "
                           "excluded Tair/Rainf z-scores of the combined panel differ and are NOT certified. Duplicate admin "
                           "2996 at 2025-10 is identical on all admitted sources (differs only in excluded z-scores); first "
                           "occurrence kept. ACLED excluded from features by alignment v3."),
                "components": [{"file": str(PINNED), "sha256": before["pinned"], "months": ["2024-12"]},
                               {"file": str(COMBINED), "sha256": before["combined"], "months": EXT_MONTHS[1:]}],
                "deduplicated": dedup}
    mpath = ext_dir / "extension_manifest_v3.json"
    with open(mpath, "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    src = load(RUN10 / "prepared/manifests/sources.json")["sources"]["panel"]
    pinned = rx._certified_panel(Path(src["path"]), src["sha256"], schema)
    panel = rx.load_extension(mpath, schema, pinned)
    sc = ff.Scaffold(panel, rx._covariate_columns(schema))
    areas = sc.areas
    o = mi(ORIGIN)
    origins = np.full(areas.size, o, dtype=np.int64)
    cov = ff.covariate_features(sc, schema, areas, origins + H, origins, a3)
    rws = np.array([sc.area_pos[int(x)] for x in areas])
    bad = []
    for s in cov.columns:
        k = a3.get(s, {}).get("kind")
        if k == "static":
            want = sc.at(s, rws, origins)
        elif k == "monthly":
            want = sc.at(s, rws, origins - a3[s]["lag"])
        elif k == "annual":
            ref = ff.annual_reference_year(origins, a3[s])
            want = sc.at(s, rws, (ref * 12 + a3[s]["value_month"] - 1).astype(np.int64))
        else:
            continue
        if not np.array_equal(cov[s].to_numpy(float), want, equal_nan=True):
            bad.append(s)
    acled_in = [s for s in acled_sources(a10) if s in cov.columns]
    if bad or acled_in:
        raise SystemExit(f"loader verification failed: {bad} acled {acled_in}")
    done(run, "extension", {"csv_sha256": sha(csv_path), "manifest_sha256": sha(mpath), "rows": len(rows),
                            "per_month": per_month, "deduplicated": dedup,
                            "scaffold": [int(sc.areas.size), ml(sc.first_month), ml(sc.first_month + sc.n_months - 1)],
                            "covariate_columns": int(cov.shape[1]), "monthly_nan_cells": int(cov[[c for c in cov.columns if a3.get(c, {}).get("kind") == "monthly"]].isna().to_numpy().sum())})


# ----------------------------------------------------------------------------------------- inputs
def scenario_frames(run, name):
    import pandas as pd
    obs = pd.read_csv(RUN10 / "prepared/ledgers/observations.csv")
    led = pd.read_csv(RUN10 / "prepared/manifests/release_ledger.csv", dtype=str)
    if SCEN[name]["oct_visible"]:
        tr = pd.read_csv(TRUTH_OCT / "truth_oct2025.csv")
        cw = pd.read_csv(TRUTH_OCT / "crosswalk_oct2025.csv", dtype=str)
        cmap = cw[cw.match_status == "admitted"].assign(area=lambda d: d.area.astype(float).astype(int)).set_index("area").country_code
        add = pd.DataFrame({"area": tr.area, "raw_phase": tr.raw_phase.astype(float), "country": tr.area.map(cmap),
                            "month": mi(ORIGIN), "merged_class": tr.class_code + 1, "class_code": tr.class_code,
                            "month_label": ORIGIN})
        if add.country.isna().any():
            raise SystemExit("Oct-2025 truth rows without a country")
        obs = pd.concat([obs, add[obs.columns]], ignore_index=True)
        src = ("10-03 S0: genuine Oct-2025 CS from approved 10-02 truth release v2 (exact-name crosswalk), treated as "
               "released at its month-end (reference_month_end convention); reconstructed")
        led = pd.concat([led, pd.DataFrame({"cycle_id": "CS-2025-10", "product": "CS", "country": sorted(add.country.unique()),
                                            "reference_month": ORIGIN, "release_date": "2025-10-31",
                                            "evidence": "reconstructed", "source": src})], ignore_index=True)
    return obs, led


def build_ctx(run, name, obs, led):
    from scripts import run_experiment as rx
    from src.experiment import availability as av
    from src.feature import fourclass_features as ff
    schema = load(PKG / "feature-schema.json")
    src = load(RUN10 / "prepared/manifests/sources.json")["sources"]["panel"]
    pinned = rx._certified_panel(Path(src["path"]), src["sha256"], schema)
    panel = rx.load_extension(run / "extension_v3" / "extension_manifest_v3.json", schema, pinned)
    scaffold = ff.Scaffold(panel, rx._covariate_columns(schema))
    ledger = av.ReleaseLedger(led, real=True)
    return av.Availability(obs[["area", "month", "country", "class_code"]], ledger, scaffold, schema, H,
                           load(run / "alignment_v3.json"), truth=None)


def step_inputs(run):
    import numpy as np
    out = {}
    for name in SCEN:
        obs, led = scenario_frames(run, name)
        d = run / name
        d.mkdir()
        obs.to_csv(d / "observations.csv", index=False, lineterminator="\n")
        led.to_csv(d / "release_ledger.csv", index=False, lineterminator="\n")
        ctx = build_ctx(run, name, obs, led)
        areas = ctx.scaffold.areas
        view = ctx.prediction_view(areas, np.full(areas.size, mi(TARGET)), 0)
        src = view.persistence_source_month
        sd = view[view.country == "SD"].persistence_source_month.dropna()
        last = ml(int(src.max()))
        rec = {"observations_sha256": sha(d / "observations.csv"), "ledger_sha256": sha(d / "release_ledger.csv"),
               "availability_inputs_sha256": ctx.inputs_sha256, "features": len(ctx.features),
               "latest_persistence_source": last, "sd_latest_source": ml(int(sd.max())) if len(sd) else None,
               "persistence_rows": int(src.notna().sum()), "gate_k": SCEN[name]["gate_k"],
               "countries": len(ctx.countries), "hidden_at_origin_k3": [ml(m) for m in sorted(ctx.ledger.hidden(mi(ORIGIN), 3))]}
        want_last = "2025-10" if SCEN[name]["oct_visible"] else "2024-10"
        want_sd = "2025-10" if SCEN[name]["oct_visible"] else "2024-06"
        if last != want_last or rec["sd_latest_source"] != want_sd:
            raise SystemExit(f"{name}: persistence check failed {rec}")
        out[name] = rec
    done(run, "inputs", out)


# ---------------------------------------------------------------------------------------- predict
def transition_fit(state_train, y, state_pred):
    """Empirical class frequencies per (class, age bucket, country); < 30 rows -> (class, age bucket); then overall."""
    import numpy as np
    import pandas as pd
    tr = pd.DataFrame({"c": state_train[0], "a": state_train[1], "k": state_train[2], "y": y})
    full = tr.groupby(["c", "a", "k", "y"]).size().unstack(fill_value=0).reindex(columns=range(4), fill_value=0)
    part = tr.groupby(["c", "a", "y"]).size().unstack(fill_value=0).reindex(columns=range(4), fill_value=0)
    fn, pn = full.sum(axis=1), part.sum(axis=1)
    full, part = {i: r.to_numpy() for i, r in full.iterrows()}, {i: r.to_numpy() for i, r in part.iterrows()}
    overall = np.bincount(tr.y, minlength=4)
    pred, level = [], []
    for c, a, k in zip(*state_pred):
        if (c, a, k) in fn.index and fn[(c, a, k)] >= 30:
            v, lv = full[(c, a, k)], "country"
        elif (c, a) in pn.index and pn[(c, a)] >= 30:
            v, lv = part[(c, a)], "pooled_class_age"
        else:
            v, lv = overall, "overall"
        pred.append(int(np.argmax(v)))
        level.append(lv)
    return np.array(pred), level


def transition_state(X, names, countries, horizon):
    import numpy as np
    ph = X[:, names.index("hist_latest_observed_phase")]
    age = X[:, names.index("hist_latest_observed_age")] + horizon
    c = np.where(np.isnan(ph), -1, np.minimum(np.nan_to_num(ph), 4) - 1).astype(int)
    a = np.select([np.isnan(age), age <= 4, age <= 8, age <= 12], [-1, 0, 1, 2], 3).astype(int)
    return c, a, np.asarray(countries).astype(str)


def logistic_fit(Xtr, y, Xte):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    def prep(X):
        miss = np.isnan(X).astype(float)
        Z = np.where(np.isnan(X), med, X)
        return Z, miss
    Ztr, Mtr = prep(Xtr)
    mu, sd = Ztr.mean(axis=0), Ztr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    keep = Mtr.std(axis=0) > 0
    A = np.hstack([(Ztr - mu) / sd, Mtr[:, keep]])
    Zte, Mte = prep(Xte)
    B = np.hstack([(Zte - mu) / sd, Mte[:, keep]])
    m = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(A, y)
    return m.classes_[np.argmax(m.predict_proba(B), axis=1)].astype(int), int(m.n_iter_.max())


def self_check():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 4000
    c = rng.integers(0, 4, n); a = rng.integers(0, 4, n); k = np.where(np.arange(n) < n // 2, "X", "Y")
    y = c.copy()
    p, lv = transition_fit((c, a, k), y, (c[:50], a[:50], k[:50]))
    assert (p == y[:50]).all() and set(lv) == {"country"}, "transition baseline must recover a deterministic state->class map"
    p2, lv2 = transition_fit((c, a, k), y, (np.array([1]), np.array([2]), np.array(["Z"])))
    assert lv2 == ["pooled_class_age"] and p2[0] == 1, "unseen country must fall back to (class, age)"
    p3, lv3 = transition_fit((c[:40], a[:40], k[:40]), y[:40], (np.array([1]), np.array([2]), np.array(["X"])))
    assert lv3 == ["overall"], "sparse states must fall back to overall frequencies"
    X = np.c_[y + rng.normal(0, .1, n), rng.normal(size=n)]
    X[::7, 1] = np.nan
    q, _ = logistic_fit(X, y, X[:50])
    assert (q == y[:50]).mean() > .9, "logistic must fit a separable synthetic"
    return "passed"


def step_predict(run):
    import numpy as np
    import pandas as pd
    from scripts import run_experiment as rx
    from src.experiment import stage3 as s3
    out = {"self_check": self_check()}
    frozen = load(RUN10 / "scenario_final/frozen.json")
    entry = frozen["recipe"]["4"]
    record, cluster_of = rx._frozen_map(RUN10, entry)
    o, t = mi(ORIGIN), TARGET
    for name, cfg in SCEN.items():
        d = run / name
        obs = pd.read_csv(d / "observations.csv")
        led = pd.read_csv(d / "release_ledger.csv", dtype=str)
        ctx = build_ctx(run, name, obs, led)
        store = s3.GlobalStore(d / "globals")
        res = rx.scen_dev_fold(ctx, entry["strategy"], H, 0, t, store, record, cluster_of, gate_k=cfg["gate_k"])
        identity = {"task": "10-03", "scenario": name, "role": cfg["role"], "phase": "confirm_onset", "strategy": entry["strategy"],
                    "horizon": H, "target_month": t, "origin_month": ORIGIN, "gate_k": cfg["gate_k"], "map_id": entry["map_id"],
                    "availability_inputs_sha256": ctx.inputs_sha256, "truth": "not loaded (blind)"}
        rx.save_fold(d / "fold", res["system"], identity, False, extra={"pooled_predictions.csv.gz": res["pooled"]["predictions"]})
        panel = s3.ScenarioPanel(ctx, 0, entry["strategy"], gate_k=cfg["gate_k"])
        pool, rows = panel.fit_pool(o)
        names = list(panel.features)
        hist = [i for i, n in enumerate(names) if n.startswith("hist_")]
        assert len(hist) == 75, len(hist)
        src, test = panel.target_rows(mi(t))
        ctry_tr = [ctx.area_country.get(int(x), "") for x in pool.area[rows]]
        ctry_te = [ctx.area_country.get(int(x), "") for x in src.area[test]]
        ptr, lv = transition_fit(transition_state(pool.X[rows], names, ctry_tr, H), pool.y[rows],
                                 transition_state(src.X[test], names, ctry_te, H))
        plo, iters = logistic_fit(pool.X[rows][:, hist], pool.y[rows], src.X[test][:, hist])
        base = pd.DataFrame({"area": src.area[test], "target_month": t, "origin_month": ORIGIN, "country": ctry_te,
                             "y_pred_transition": ptr, "transition_level": lv, "y_pred_logistic": plo})
        bpath = d / "baseline_predictions.csv.gz"
        rx.write_csv_gz(bpath, base)
        out[name] = {"fit_rows": int(len(rows)), "prediction_rows": int(len(test)), "logistic_iterations": iters,
                     "transition_levels": pd.Series(lv).value_counts().to_dict(),
                     "fold_routes": load(d / "fold" / "fold.json")["routes"]}
    files = {}
    for name in SCEN:
        for p in sorted((run / name / "fold").iterdir()):
            files[f"{name}/fold/{p.name}"] = sha(p)
        files[f"{name}/baseline_predictions.csv.gz"] = sha(run / name / "baseline_predictions.csv.gz")
    from src.utils import run_identity as rid
    freeze = {"files": files, "code_identity": rid.code_identity(), "inputs_manifest_sha256": sha(run / "inputs_manifest.json"),
              "implement_md_sha256": sha(TASK / "implement.md"), "prd_sha256": sha(TASK / "prd.md"),
              "design_sha256": sha(TASK / "design.md"), "driver_sha256": sha(HERE),
              "note": "all predictions written before any 2026-02 CS value is read"}
    with open(run / "predictions_frozen.json", "x", encoding="utf-8") as f:
        json.dump(freeze, f, indent=1, default=str)
    out["predictions_frozen_sha256"] = sha(run / "predictions_frozen.json")
    done(run, "predict", out)


# ------------------------------------------------------------------------------------------ truth
def step_truth(run):
    import numpy as np
    import pandas as pd
    import pyogrio
    fz = load(run / "predictions_frozen.json")
    for rel, h in fz["files"].items():
        if sha(run / rel) != h:
            raise SystemExit(f"frozen prediction {rel} changed")
    raw_sha = sha(RAW26)
    pin = load(run / "inputs_manifest.json")["files"]["raw_2025_2026_cs_PROTECTED_hash_only"]["sha256"]
    if raw_sha != pin:
        raise SystemExit("raw 2025-2026 file differs from its pin")
    raw = pd.read_csv(RAW26, encoding="utf-8-sig", dtype=str)
    o = raw[(raw.scenario == "CS") & raw.reporting_date.str.startswith(TARGET)].copy()
    o = o[["id", "fnid", "country", "country_code", "geographic_unit_full_name", "value", "description",
           "is_allowing_for_assistance", "status"]].reset_index(drop=True)
    hist = pd.read_csv(FEWS, usecols=["admin_code", "admin_name"], dtype={"admin_code": "Int64"}).dropna()
    codes_of = hist.groupby("admin_name").admin_code.agg(lambda s: sorted(set(int(x) for x in s)))
    dbf = pyogrio.read_dataframe(SHP, read_geometry=False, columns=["admin_code", "admin_name"])
    dbf_name = dict(zip(dbf.admin_code.astype(int), dbf.admin_name))
    obs = pd.read_csv(RUN10 / "prepared/ledgers/observations.csv", usecols=["area", "country"])
    obs_country = obs.drop_duplicates("area").set_index("area").country.to_dict()
    panel_countries = set(obs.country)
    o["name_codes"] = o.geographic_unit_full_name.map(lambda n: codes_of.get(n, []))
    o["area"] = o.name_codes.map(lambda c: c[0] if len(c) == 1 else np.nan).astype("Int64")
    o["dbf_admin_name"] = o.area.map(lambda a: dbf_name.get(int(a)) if pd.notna(a) else None)
    o["obs_country"] = o.area.map(lambda a: obs_country.get(int(a)) if pd.notna(a) else None)
    o["phase"] = pd.to_numeric(o.value, errors="coerce")
    dup = set(o.area.dropna()[o.area.dropna().duplicated()].astype(int))

    def status(r):
        if r.country_code not in panel_countries: return "excluded_country_outside_panel"
        if len(r.name_codes) == 0: return "excluded_unmatched_name"
        if len(r.name_codes) > 1: return "excluded_ambiguous_name_multiple_codes:" + "/".join(map(str, r.name_codes))
        a = int(r.area)
        if a not in dbf_name: return "excluded_code_outside_universe"
        if a in dup: return "excluded_code_receives_multiple_rows"
        if r.obs_country is not None and r.obs_country != r.country_code: return "excluded_country_mismatch"
        if r.dbf_admin_name != r.geographic_unit_full_name: return "excluded_dbf_name_mismatch"
        if r.phase not in (1, 2, 3, 4, 5): return f"excluded_no_genuine_phase:{r.status}"
        return "admitted"
    o["match_status"] = o.apply(status, axis=1)
    o["class_code"] = np.where(o.match_status == "admitted", np.minimum(o.phase.fillna(0), 4) - 1, np.nan)
    o["name_codes"] = o.name_codes.map(lambda c: "/".join(map(str, c)))
    rel = run / "truth_release_feb2026"
    rel.mkdir()
    o.to_csv(rel / "crosswalk_feb2026.csv", index=False, lineterminator="\n")
    adm = o[o.match_status == "admitted"]
    truth = pd.DataFrame({"area": adm.area.astype(int), "target_month": TARGET, "class_code": adm.class_code.astype(int),
                          "raw_phase": adm.phase.astype(int), "is_allowing_for_assistance": adm.is_allowing_for_assistance,
                          "fnid": adm.fnid, "source_id": adm.id}).sort_values("area")
    assert not truth.duplicated(["area"]).any() and truth.class_code.isin([0, 1, 2, 3]).all()
    truth.to_csv(rel / "truth_feb2026.csv", index=False, lineterminator="\n")
    release = {"approved": True, "approved_by": "pre-registered automatic rule (10-03 D11; no manual gate)",
               "truth_file": "truth_feb2026.csv", "truth_sha256": sha(rel / "truth_feb2026.csv"),
               "crosswalk": "crosswalk_feb2026.csv", "crosswalk_sha256": sha(rel / "crosswalk_feb2026.csv"),
               "predictions_frozen_sha256": sha(run / "predictions_frozen.json"),
               "sources": {"raw_2025_2026_cs": {"path": str(RAW26), "sha256": raw_sha},
                           "fewsnet_history_names": {"path": str(FEWS), "sha256": sha(FEWS)},
                           "shapefile_dbf_attributes": {"path": str(SHP.with_suffix(".dbf")), "sha256": sha(SHP.with_suffix(".dbf"))},
                           "shapefile_shp_pin": {"path": str(SHP), "sha256": sha(SHP)}},
               "rule": "10-02 approved exact full-name + country + canonical DBF-name one-to-one; genuine phase 1-5; "
                       "class = min(phase,4)-1; six out-of-panel countries excluded; geometric continuity NOT certified",
               "status_counts": o.match_status.value_counts().to_dict()}
    with open(rel / "release.json", "x", encoding="utf-8") as f:
        json.dump(release, f, indent=1, ensure_ascii=False)
    done(run, "truth", {"release_sha256": sha(rel / "release.json"), "truth_sha256": release["truth_sha256"],
                        "crosswalk_sha256": release["crosswalk_sha256"], "admitted": int(len(truth)),
                        "status_counts": release["status_counts"],
                        "class_counts": truth.class_code.value_counts().sort_index().to_dict()})


# --------------------------------------------------------------------------------------- evaluate
def keyed(run, name):
    import pandas as pd
    from scripts import run_experiment as rx
    d = run / name
    sysp = rx.read_csv(d / "fold" / "predictions.csv.gz")
    pool = rx.read_csv(d / "fold" / "pooled_predictions.csv.gz")[["area", "y_pred_code"]].rename(columns={"y_pred_code": "y_pred_pooled"})
    base = rx.read_csv(d / "baseline_predictions.csv.gz")[["area", "y_pred_transition", "y_pred_logistic"]]
    k = sysp.rename(columns={"y_pred_code": "y_pred_system"}).merge(pool, on="area", validate="one_to_one").merge(base, on="area", validate="one_to_one")
    truth = pd.read_csv(run / "truth_release_feb2026/truth_feb2026.csv")[["area", "class_code"]].rename(columns={"class_code": "truth_code"})
    origin = pd.read_csv(TRUTH_OCT / "truth_oct2025.csv")[["area", "class_code"]].rename(columns={"class_code": "origin_truth"})
    k = k.drop(columns=["y_true_code"]).merge(truth, on="area", how="left").merge(origin, on="area", how="left")
    return k


def step_evaluate(run):
    import numpy as np
    import pandas as pd
    from scripts import report_fourclass as rep
    from scripts import run_experiment as rx
    panel_c = set(pd.read_csv(RUN10 / "prepared/ledgers/observations.csv", usecols=["country"]).country)
    comps = {"persistence": "persistence_class_code", "transition": "y_pred_transition", "logistic": "y_pred_logistic",
             "system": "y_pred_system"}

    def cat(b):
        if b["ci95_low"] is None:
            return "NA:" + (b.get("ci_reason") or "undefined")
        return "CONFIRMED" if b["ci95_low"] > 0 else "CONTRADICTED" if b["ci95_high"] < 0 else "INCONCLUSIVE"
    res = {}
    for name in SCEN:
        k = keyed(run, name)
        k["country"] = k["country"].astype(str)
        lab = k.truth_code.notna() & k.country.isin(panel_c)
        risk = lab & k.origin_truth.notna() & (k.origin_truth < 2)
        matched = risk & k.persistence_class_code.notna()
        rows = k[matched].copy()
        cells = {c: rep.crisis_paired_bootstrap(rows, "y_pred_pooled", col) for c, col in comps.items()}
        h1 = cat(cells["persistence"])
        h2 = None
        if h1 == "CONFIRMED":
            c2 = [cat(cells["transition"]), cat(cells["logistic"])]
            h2 = "CONFIRMED" if all(x == "CONFIRMED" for x in c2) else (
                "CONTRADICTED" if any(x == "CONTRADICTED" for x in c2) else "INCONCLUSIVE" if all(not x.startswith("NA") for x in c2) else "NA")
        s1 = k[lab & k.persistence_class_code.notna()].copy()
        study1 = {c: rep.crisis_paired_bootstrap(s1, "y_pred_pooled", col) for c, col in comps.items()}
        y = rows.truth_code.to_numpy(int)
        pr = {}
        for col in ["y_pred_pooled"] + list(comps.values()):
            p = rows[col].to_numpy(int)
            tp = int(((y >= 2) & (p >= 2)).sum()); fp = int(((y < 2) & (p >= 2)).sum()); fn = int(((y >= 2) & (p < 2)).sum())
            pr[col] = {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None,
                       "recall": tp / (tp + fn) if tp + fn else None}
        ctab = []
        for c, g in rows.groupby("country"):
            def f1(col):
                yy, pp = g.truth_code.to_numpy(int) >= 2, g[col].to_numpy(int) >= 2
                den = 2 * (yy & pp).sum() + (~yy & pp).sum() + (yy & ~pp).sum()
                return None if den == 0 else float(2 * (yy & pp).sum() / den)
            ctab.append({"country": c, "onset_keys": int(len(g)), "onsets": int((g.truth_code >= 2).sum()),
                         **{f"f1_{n}": f1(col) for n, col in [("pooled", "y_pred_pooled")] + list(comps.items())}})
        pd.DataFrame(ctab).to_csv(run / name / "country_onset.csv", index=False, lineterminator="\n")
        rx.write_csv_gz(run / name / "keyed_eval.csv.gz", k)
        res[name] = {"role": SCEN[name]["role"],
                     "counts": {"forecast_keys": int(len(k)), "with_feb_truth_panel": int(lab.sum()),
                                "onset_risk_set": int(risk.sum()), "excluded_no_persistence": int((risk & ~matched).sum()),
                                "matched_onset_keys": int(matched.sum()), "onsets": int((rows.truth_code >= 2).sum()),
                                "countries": int(rows.country.nunique()),
                                "excluded_no_origin_truth": int((lab & k.origin_truth.isna()).sum()),
                                "excluded_origin_crisis": int((lab & (k.origin_truth >= 2)).sum())},
                     "onset_pooled_minus": cells, "H1": h1, "H2": h2, "study1_pooled_minus": study1,
                     "precision_recall_onset": pr,
                     "note": ("PRIMARY decision scenario" if name == "S3" else
                              "secondary; onset-set persistence F1 = 0 is mechanical when Oct-2025 is visible")}
    out = run / "evaluation.json"
    with open(out, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    done(run, "evaluate", {"evaluation_sha256": sha(out), "S3_H1": res["S3"]["H1"], "S3_H2": res["S3"]["H2"],
                           "S3_counts": res["S3"]["counts"]})


# ------------------------------------------------------------------------------------------ check
def step_check(run):
    import pandas as pd
    fz = load(run / "predictions_frozen.json")
    mism = [rel for rel, h in fz["files"].items() if sha(run / rel) != h]
    ev = load(run / "evaluation.json")
    rec = {"frozen_hash_mismatches": mism}
    for name in SCEN:
        k = pd.read_csv(run / name / "keyed_eval.csv.gz")
        panel_c = set(pd.read_csv(RUN10 / "prepared/ledgers/observations.csv", usecols=["country"]).country)
        m = k.truth_code.notna() & k.country.astype(str).isin(panel_c) & k.origin_truth.notna() & (k.origin_truth < 2) & k.persistence_class_code.notna()
        r = k[m]
        y = r.truth_code.to_numpy(int) >= 2
        f1 = {}
        for col in ["y_pred_pooled", "persistence_class_code", "y_pred_transition", "y_pred_logistic", "y_pred_system"]:
            p = r[col].to_numpy(int) >= 2
            den = 2 * (y & p).sum() + (~y & p).sum() + (y & ~p).sum()
            f1[col] = None if den == 0 else float(2 * (y & p).sum() / den)
        mp = {"persistence": "persistence_class_code", "transition": "y_pred_transition", "logistic": "y_pred_logistic", "system": "y_pred_system"}
        diffs = []
        for c, col in mp.items():
            b = ev[name]["onset_pooled_minus"][c]
            if b["model_f1"] is not None and abs(b["model_f1"] - f1["y_pred_pooled"]) > 1e-12: diffs.append(f"{c}:model")
            if b["comparator_f1"] is not None and f1[col] is not None and abs(b["comparator_f1"] - f1[col]) > 1e-12: diffs.append(f"{c}:comparator")
        truth = pd.read_csv(run / "truth_release_feb2026/truth_feb2026.csv")
        joined = k.truth_code.notna().sum()
        rec[name] = {"matched_onset_keys": int(m.sum()), "f1_recount": f1, "recount_mismatches": diffs,
                     "truth_rows": int(len(truth)), "truth_joined_forecast_keys": int(joined),
                     "unmatched_truth_rows": int(len(set(truth.area) - set(k.area)))}
    rec["problems"] = mism + [f"{n}:{d}" for n in SCEN for d in rec[n]["recount_mismatches"]]
    done(run, "check", rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("step", choices=ORDER)
    a = ap.parse_args()
    if a.step != "pin":
        require(a.run_dir, a.step)
    globals()[f"step_{a.step}"](a.run_dir)


if __name__ == "__main__":
    main()
