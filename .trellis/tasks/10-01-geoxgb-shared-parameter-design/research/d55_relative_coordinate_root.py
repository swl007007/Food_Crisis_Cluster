#!/usr/bin/env python3
"""D55 / A29: relative-coordinate root contrast for the 21 saved D34 roots (one-off research, diagnostic only).

python .trellis/tasks/10-01-geoxgb-shared-parameter-design/research/d55_relative_coordinate_root.py \
    --d34-run D34_RUN [--out C:\\Users\\swl00\\geoxgb_runs\\geoxgb-d55-relative-coordinate-root-20261002] \
    [--d52-run C:\\Users\\swl00\\geoxgb_runs\\geoxgb-d52-binary-root-20261002] [--producer-rev 7b2bf6f]
python .../d55_relative_coordinate_root.py --selftest      # synthetic fits only, no real data

Encoding: k = rr.persistence_codes(hist_phase_o00) of each role's own matrix (schema index 87), known-mask kept
separately; z = y XOR k when known, z = y (k = 0 coordinate convention) when missing. One nx.fit_global on z per
root (original G invocation). Decoding: p_y[:, j] = q[:, j XOR k] row-wise, then np.argmax(p_y) on the original
axis (never argmax(q) XOR k). Pass 1 gates all 21 roots (D34 acceptance, rr.gate_root replay, G lock, FIT keys,
D52 hash-bound row alignment, D49 E3 consistency, D38 post-hoc reference == D54 post_root) before ANY fit.
Pass 2 re-gates, requires the same FIT-key sha, then fits. First failure stops (failure.json).
The relative UBJ is diagnostic-only: not an absolute-label root, never for local continuation or production use.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
RESEARCH = SCRIPT.parent
PACKAGE = REPO / "FEWSNETGeoXGBExperiment"
sys.path.insert(0, str(PACKAGE))
from scripts import stage1_recency_root as rr  # noqa: E402
from scripts import stage1_shallow_replay as sr  # noqa: E402
from scripts.stage1_rootconf_compare import accept_mode  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.feature.fourclass_features import load_schema, month_label  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.model import native_xgb as nx  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402

G_BY_H = {4: "G1", 8: "G4", 12: "G2"}
HORIZONS = (4, 8, 12)
LABELS = fourclass.CLASS_LABELS
PARTS = ("FIT", "C", "E3")
CF = ("tp", "fp", "fn", "tn")
ORIGIN_FEATURE = "hist_phase_o00"
ORIGIN_INDEX = 87
MAPPING_VERSION = "xor-v1"
Q_ORIGIN, Q_OTHER = 0.625, 0.125
DEFAULT_OUT = Path(r"C:\Users\swl00\geoxgb_runs\geoxgb-d55-relative-coordinate-root-20261002")
DEFAULT_D52 = Path(r"C:\Users\swl00\geoxgb_runs\geoxgb-d52-binary-root-20261002")
D49_SUMMARY = RESEARCH / "d49_summary.json"
D52_COMPLETION = RESEARCH / "d52_completion.json"
D54_PER_ROOT = RESEARCH / "d54_per_root.csv"
GateError = rr.GateError


# ---------------------------------------------------------------- encoding / decoding

def origin_index() -> int:
    idx = load_schema(sr.SCHEMA)["ordered_features"].index(ORIGIN_FEATURE)
    if idx != ORIGIN_INDEX:
        raise GateError(f"{ORIGIN_FEATURE} at schema index {idx}, not {ORIGIN_INDEX}")
    return idx


def reference(X) -> tuple[np.ndarray, np.ndarray]:
    """(k, known) from the role's own origin column; k = 0 where missing (coordinate convention only)."""
    codes = rr.persistence_codes(np.asarray(X)[:, ORIGIN_INDEX])
    known = np.isfinite(codes)
    return np.where(known, codes, 0.0).astype(np.int64), known


def encode(y, k, known) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    if y.size and (y.min() < 0 or y.max() > 3):
        raise GateError("labels outside 0..3")
    k_eff = np.where(known, k, 0).astype(np.int64)
    z = np.bitwise_xor(y, k_eff)
    if not np.array_equal(np.bitwise_xor(z, k_eff), y) or (z.size and (z.min() < 0 or z.max() > 3)):
        raise GateError("y -> z map is not a bijection on 0..3")
    if np.any(z[~np.asarray(known)] != y[~np.asarray(known)]):
        raise GateError("missing-origin rows are not identity")
    return z


def decode(q, k, known) -> np.ndarray:
    """p_y[i, j] = q[i, j XOR k_i] (k_i = 0 where missing); returns a new array."""
    q = np.asarray(q, dtype=np.float64)
    k_eff = np.where(known, k, 0).astype(np.int64)
    cols = np.bitwise_xor(np.arange(4)[None, :], k_eff[:, None])
    return np.take_along_axis(q, cols, axis=1)


def decide(p_y) -> np.ndarray:
    """Production decision: argmax on the decoded original axis (ties -> first column)."""
    return np.argmax(np.asarray(p_y), axis=1).astype(np.int64)


def labels_sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes()).hexdigest()


def posthoc_d38(p, k, known) -> np.ndarray:
    """D38 fixed post-hoc reference p ∝ p*q (q=.625 origin / .125 others); missing origin unchanged."""
    p = np.asarray(p, dtype=np.float64)
    out = p.copy()
    if known.any():
        q = np.full((int(known.sum()), 4), Q_OTHER)
        q[np.arange(q.shape[0]), np.asarray(k)[known]] = Q_ORIGIN
        w = p[known] * q
        out[known] = w / w.sum(axis=1, keepdims=True)
    return out


# ---------------------------------------------------------------- fit

def fit_relative_root(X_fit, z_fit, g_config: str):
    """The single D55 fit: original nx.fit_global invocation on the transformed labels (no weight/margin)."""
    return nx.fit_global(X_fit, z_fit, plan.G_CONFIGS[g_config])


def check_fit(booster, record: dict, g_config: str, n_rows: int) -> None:
    params, rounds = nx.booster_params(plan.G_CONFIGS[g_config])
    problems = []
    if record.get("params") != params or record.get("rounds_total") != rounds:
        problems.append("params / rounds differ from booster_params(G)")
    if booster.num_boosted_rounds() != rounds:
        problems.append("boosted rounds differ")
    if record.get("base_score") != "5E-1" or nx.base_score(booster) != "5E-1":
        problems.append(f"base_score {record.get('base_score')!r} != '5E-1'")
    if nx.num_class(booster) != 4:
        problems.append("num_class != 4")
    if "sample_weight" in record or "base_margin" in record or nx.is_margin_marked(booster):
        problems.append("weight or margin present")
    if record.get("rows") != n_rows:
        problems.append("rows differ")
    if problems:
        raise GateError("relative fit check failed: " + "; ".join(problems))


# ---------------------------------------------------------------- scoring

def confusion(truth, pred) -> dict:
    z, c = np.asarray(truth) >= 2, np.asarray(pred) >= 2
    return {k: int(v.sum()) for k, v in zip(CF, (z & c, ~z & c, z & ~c, ~z & ~c))}


def f1(cf: dict) -> Fraction:
    d = 2 * cf["tp"] + cf["fp"] + cf["fn"]
    return Fraction(2 * cf["tp"], d) if d else Fraction(0)


def macro_f1(truth, pred) -> float:
    t, p = np.asarray(truth), np.asarray(pred)
    out = []
    for c in range(4):
        tp, fp, fn = int(np.sum((t == c) & (p == c))), int(np.sum((t != c) & (p == c))), int(np.sum((t == c) & (p != c)))
        out.append(float(f1({"tp": tp, "fp": fp, "fn": fn})))
    return float(np.mean(out))


def normalise(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return p / p.sum(axis=1, keepdims=True)


def block(truth, pred, p=None) -> dict:
    cf = confusion(truth, pred)
    f = f1(cf)
    out = {"n": int(len(truth)), **cf, "crisis_f1_exact": str(f), "crisis_f1": float(f),
           "macro_f1_fourclass": macro_f1(truth, pred) if len(truth) else None}
    z = (np.asarray(truth) >= 2).astype(np.float64)
    if p is None:                                   # persistence: one-hot Brier only
        out["crisis_brier"] = float(np.mean(((np.asarray(pred) >= 2) - z) ** 2)) if len(z) else None
        return out
    pn = normalise(p)
    out["crisis_brier"] = float(np.mean((pn[:, 2] + pn[:, 3] - z) ** 2)) if len(z) else None
    if len(z):
        pt = pn[np.arange(len(pn)), np.asarray(truth, dtype=np.int64)]
        if np.any(~(pt > 0)):
            raise GateError(f"non-positive true-class probability in {int(np.sum(~(pt > 0)))} rows")
        out["logloss_fourclass"] = float(np.mean(-np.log(pt)))
    return out


def score_part(fr: pd.DataFrame, part: str) -> dict:
    t = fr["truth"].to_numpy()
    known = fr["known"].to_numpy(bool)
    P = {a: fr[[f"{c}_{lab}" for lab in LABELS]].to_numpy(np.float64)
         for a, c in (("relative", "p_y"), ("original", "p_original"))}
    D = {"relative": fr["y_relative"].to_numpy(), "original": fr["y_original"].to_numpy()}
    if part == "E3":
        P["d38_posthoc_reference"] = fr[[f"p_posthoc_{lab}" for lab in LABELS]].to_numpy(np.float64)
        D["d38_posthoc_reference"] = fr["y_posthoc"].to_numpy()
    out = {"n_all": int(len(t)), "missing_origin": int((~known).sum()),
           "all_keys": {a: block(t, D[a], P[a]) for a in P},
           "missing_origin_keys": {a: block(t[~known], D[a][~known], P[a][~known]) for a in P},
           "matched": {a: block(t[known], D[a][known], P[a][known]) for a in P}}
    out["matched"]["persistence"] = block(t[known], fr["k"].to_numpy()[known])
    return out


def screen(per_root: dict) -> dict:
    """12 strict exact-Fraction comparisons on matched E3: relative > original and > persistence, per H,
    pooled and mean-fold. Equality fails."""
    table, comps = {}, []
    for h in HORIZONS:
        roots = [r for r in per_root.values() if r["horizon"] == h]
        if len(roots) != 7:
            raise GateError(f"H{h}: {len(roots)} roots, not 7")
        m = {a: [r["scores"]["E3"]["matched"][a] for r in roots] for a in ("relative", "original", "persistence")}
        pooled = {a: f1({c: sum(b[c] for b in v) for c in CF}) for a, v in m.items()}
        mean = {a: sum((Fraction(b["crisis_f1_exact"]) for b in v), Fraction(0)) / len(v) for a, v in m.items()}
        entry = {"pooled_f1_exact": {a: str(v) for a, v in pooled.items()},
                 "mean_fold_f1_exact": {a: str(v) for a, v in mean.items()},
                 "pooled_f1_display": {a: float(v) for a, v in pooled.items()},
                 "mean_fold_f1_display": {a: float(v) for a, v in mean.items()}, "comparisons": {}}
        for agg, vals in (("pooled", pooled), ("mean_fold", mean)):
            for other in ("original", "persistence"):
                ok = vals["relative"] > vals[other]
                entry["comparisons"][f"{agg}_relative_gt_{other}"] = ok
                comps.append(ok)
        wins = {}
        for other in ("original", "persistence"):
            d = [Fraction(a["crisis_f1_exact"]) - Fraction(b["crisis_f1_exact"])
                 for a, b in zip(m["relative"], m[other])]
            wins[other] = {"wins": sum(x > 0 for x in d), "ties": sum(x == 0 for x in d),
                           "losses": sum(x < 0 for x in d)}
        entry["fold_wins_relative_vs"] = wins
        table[f"H{h}"] = entry
    if len(comps) != 12:
        raise GateError("screen does not have exactly 12 comparisons")
    return {"table": table, "n_comparisons": 12, "n_true": int(sum(comps)), "screen_pass": bool(all(comps))}


# ---------------------------------------------------------------- gate (pass 1, zero fits)

def read_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", dtype={"target_month": str})


def check_d52(d52: Path) -> dict:
    ext = d52 / "completion.json"
    if ext.read_bytes() != D52_COMPLETION.read_bytes():
        raise GateError("external D52 completion.json differs from research/d52_completion.json")
    rec = json.loads(D52_COMPLETION.read_text(encoding="utf-8"))["outputs"]
    rows = {k: v for k, v in rec.items() if "/rows_" in k}
    if len(rows) != 63:
        raise GateError(f"D52 record lists {len(rows)} rows files, not 63")
    bad = [k for k, v in rows.items() if rid.file_sha256(d52 / k) != v]
    if bad:
        raise GateError(f"D52 rows hashes differ: {bad[:3]}")
    return {"completion_sha256": rid.file_sha256(ext), "rows_files": 63}


def align_d52(d52: Path, name: str, part: str, y, g, m, k, known, p_orig) -> None:
    f = read_rows(d52 / name / f"rows_{part}.csv.gz")
    n = len(y)
    pc = f["persistence_code"].to_numpy(float) if len(f) == n else None
    problems = []
    if len(f) != n:
        problems.append(f"rows {len(f)} != {n}")
    else:
        if not (f["area"].to_numpy() == np.asarray(g)).all() or not (f["target_month"].astype(str).to_numpy()
                                                                     == np.asarray(m).astype(str)).all():
            problems.append("keys")
        if not (f["truth"].to_numpy() == np.asarray(y)).all():
            problems.append("truth")
        if not (np.array_equal(np.isfinite(pc), known) and np.array_equal(pc[known].astype(np.int64), k[known])):
            problems.append("persistence_code vs k / known-mask")
        if not np.array_equal(f[[f"p_original_{lab}" for lab in LABELS]].to_numpy(np.float64), p_orig):
            problems.append("original probabilities")
    if problems:
        raise GateError(f"{name} {part}: D52 alignment failed: {problems}")


def gate_one(run, stage, name, pair, d49, d54, d52):
    """Replay gate + G lock + FIT keys + D52 alignment + D49 + D38/D54 reference; returns everything for pass 2."""
    root = json.loads((stage / "roots" / name / "root.json").read_text(encoding="utf-8"))
    gate, data, replay, root_ubj = rr.gate_root(run, stage, name, pair, root)
    h = int(root["horizon"])
    if G_BY_H.get(h) != root["g_config"]:
        gate["passed"] = False
        gate["error"] = "G lock"
    if not gate["passed"]:
        return gate, None
    _, _, gfit, mfit = data["FIT"]
    fit_sha = nx.keys_sha(gfit, mfit)
    if fit_sha != root.get("fitting_keys_sha256"):
        raise GateError("FIT keys differ from the original FIT keys")
    original = nx.from_raw(root_ubj.read_bytes())
    probs, refs = {}, {}
    for part in PARTS:
        X, y, g, m = data[part]
        probs[part] = nx.proba(original, X) if part == "FIT" else replay[part]
        refs[part] = reference(X)
        align_d52(d52, name, part, y, g, month_label(m) if part == "FIT" else m, *refs[part], probs[part])
    _, y3, _, _ = data["E3"]
    k3, kn3 = refs["E3"]
    r49 = d49["per_root"][name]
    orig = confusion(y3[kn3], fourclass.argmax_codes(probs["E3"])[kn3])
    pers = confusion(y3[kn3], k3[kn3])
    if (int(kn3.sum()), int((~kn3).sum())) != (r49["n"], r49["excluded_missing_origin"]) \
            or any(orig[c] != r49["arms"]["original"]["argmax"][c] for c in CF) \
            or any(pers[c] != r49["persistence"][c] for c in CF):
        raise GateError(f"{name}: D49 E3 consistency failed")
    post = decide(posthoc_d38(probs["E3"], k3, kn3))
    ref = d54[(d54["root"] == name) & (d54["part"] == "E3") & (d54["keyset"] == "matched")
              & (d54["arm"] == "post_root")]
    mine = confusion(y3[kn3], post[kn3])
    if len(ref) != 1 or any(int(ref.iloc[0][c]) != mine[c] for c in CF):
        raise GateError(f"{name}: D38 post-hoc reference != D54 post_root on matched E3")
    gate["checks"].update({"g_config_locked": {"mismatches": 0, "n": 1}, "fit_keys_sha": {"mismatches": 0, "n": 1},
                           "d52_alignment": {"mismatches": 0, "n": 3}, "d49_consistency": {"mismatches": 0, "n": 1},
                           "d38_reference_d54": {"mismatches": 0, "n": 1}})
    return gate, {"root": root, "data": data, "probs": probs, "refs": refs, "fit_sha": fit_sha,
                  "root_ubj": root_ubj, "d38_e3": posthoc_d38(probs["E3"], k3, kn3)}


# ---------------------------------------------------------------- fit (pass 2)

def write_rows(path: Path, frame: pd.DataFrame) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, float_format="%.17g")


def support(y, z, known, months) -> dict:
    y, z, known = np.asarray(y), np.asarray(z), np.asarray(known, bool)
    return {grp: {"rows": int(mask.sum()), "y_counts": [int(np.sum(y[mask] == c)) for c in range(4)],
                  "z_counts": [int(np.sum(z[mask] == c)) for c in range(4)],
                  "label_dates": int(len(np.unique(np.asarray(months)[mask])))}
            for grp, mask in (("known", known), ("missing", ~known), ("all", np.ones(len(y), bool)))}


def fit_one(out: Path, name: str, g1: dict, producer_rev: str, base) -> tuple:
    root, data, probs, refs = g1["root"], g1["data"], g1["probs"], g1["refs"]
    h, g_config = int(root["horizon"]), root["g_config"]
    X_fit, y_fit, gfit, mfit = data["FIT"]
    y_copy, X_sha = np.array(y_fit, copy=True), hashlib.sha256(np.ascontiguousarray(X_fit).tobytes()).hexdigest()
    k_fit, kn_fit = refs["FIT"]
    z_fit = encode(y_fit, k_fit, kn_fit)
    booster, record = fit_relative_root(X_fit, z_fit, g_config)
    check_fit(booster, record, g_config, len(z_fit))
    if not np.array_equal(y_fit, y_copy) or hashlib.sha256(np.ascontiguousarray(X_fit).tobytes()).hexdigest() != X_sha:
        raise GateError("input matrix or labels were mutated")
    rdir = out / name
    rdir.mkdir()
    (rdir / "relative_root.ubj").write_bytes(nx.raw(booster))
    frozen = nx.from_raw((rdir / "relative_root.ubj").read_bytes())
    rows, reload = {}, {}
    for part in PARTS:
        X, y, g, m = data[part]
        k, known = refs[part]
        q_live, q = nx.proba(booster, X), nx.proba(frozen, X)
        p_live, p_y = decode(q_live, k, known), decode(q, k, known)
        bad = int(np.sum(q_live != q)) + int(np.sum(p_live != p_y))
        reload[part] = {"mismatches": bad, "n": int(q.size)}
        if bad:
            raise GateError(f"reloaded relative root does not reproduce {part} q / p_y")
        fr = pd.DataFrame({"area": g, "target_month": month_label(m) if part == "FIT" else m, "horizon": h,
                           "truth": y, "k": k, "known": known.astype(int),
                           "persistence_code": np.where(known, k, np.nan)})
        for j, lab in enumerate(LABELS):
            fr[f"q_{j}"] = q[:, j]
        for j, lab in enumerate(LABELS):
            fr[f"p_y_{lab}"] = p_y[:, j]
        for j, lab in enumerate(LABELS):
            fr[f"p_original_{lab}"] = probs[part][:, j]
        fr["y_relative"] = decide(p_y)
        fr["y_original"] = fourclass.argmax_codes(probs[part])
        if part == "E3":
            for j, lab in enumerate(LABELS):
                fr[f"p_posthoc_{lab}"] = g1["d38_e3"][:, j]
            fr["y_posthoc"] = decide(g1["d38_e3"])
        fr["call_relative"] = (fr["y_relative"] >= 2).astype(int)
        fr["call_original"] = (fr["y_original"] >= 2).astype(int)
        rows[part] = fr
        write_rows(rdir / f"rows_{part}.csv.gz", fr)
    fit_record = {"mapping_version": MAPPING_VERSION, "diagnostic_only": True,
                  "compatible_with_production_consumers": False,
                  "note": ("relative-coordinate (z = y XOR k) booster; NOT an absolute-label root; decode with "
                           "the same row's k before any use; never for local continuation"),
                  "origin_source": {"feature": ORIGIN_FEATURE, "schema_index": ORIGIN_INDEX},
                  "missing_origin_convention": "k = 0 identity (coordinate convention, not imputation)",
                  "decoded_external_class_labels": list(LABELS), "root": name, "horizon": h,
                  "target_month": root["target_month"], "g_config": g_config,
                  "fitting_keys_sha256": g1["fit_sha"], "y_sha256": labels_sha(y_fit), "z_sha256": labels_sha(z_fit),
                  "k_sha256": labels_sha(k_fit), "known_mask_sha256": labels_sha(kn_fit.astype(np.int64)),
                  "support": support(y_fit, z_fit, kn_fit, mfit), "reload_exact": reload, "fit_record": record,
                  "relative_root_sha256": rid.file_sha256(rdir / "relative_root.ubj"),
                  "original_root_sha256": root["root_booster_sha256"], "producer_rev": producer_rev}
    rid.write_json_atomic(rdir / "relative_root.json", fit_record)
    dc = rr.dev_baseline_check(rows["E3"], base, h) if base is not None else {"status": "skipped"}
    if dc["status"] == "checked" and (dc["joined"] != dc["e3_rows"] or dc["persistence_mismatches"]
                                      or dc["truth_mismatches"]):
        raise GateError(f"dev_baselines cross-check failed: {dc}")
    return rows, {"horizon": h, "target_month": root["target_month"], "dev_baselines_check": dc,
                  "support": fit_record["support"], "scores": {p: score_part(rows[p], p) for p in PARTS}}


def _stop(out, gates, phase, name, err, roots, extra) -> int:
    rid.write_json_atomic(out / "gate.json", {"passed": False, "phase": phase, "roots": gates})
    rid.write_json_atomic(out / "failure.json", {"status": "stopped", "phase": phase, "failed_root": name,
                                                 "error": err, **extra,
                                                 "not_attempted": [r for r in roots if r not in gates]})
    return 2


def run_all(run, stage, out, roots, cands, producer_rev, base, d49, d54, d52) -> tuple:
    pairs = {name: {cands[c]["e1"]: c for c in cands if cands[c]["root"] == name} for name in roots}
    gates, fit_sha, per_root = {}, {}, {}
    for name in roots:                                                  # pass 1: zero fits
        try:
            gate, g1 = gate_one(run, stage, name, pairs[name], d49, d54, d52)
            if g1 is not None:
                fit_sha[name] = g1["fit_sha"]
            del g1
        except Exception as exc:   # noqa: BLE001
            gate = {"root": name, "passed": False, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
        gates[name] = {"gate_pass": gate}
        print(f"{name}: gate {'passed' if gate['passed'] else 'FAILED'}", flush=True)
        if not gate["passed"]:
            return _stop(out, gates, "gate_pass", name, gate.get("error", "gate mismatches"), roots,
                         {"fitted_roots": []}), gates, per_root
    for name in roots:                                                  # pass 2: re-gate, then fit
        try:
            gate, g1 = gate_one(run, stage, name, pairs[name], d49, d54, d52)
            if not gate["passed"] or g1["fit_sha"] != fit_sha[name]:
                raise GateError("pass-2 re-gate failed or FIT-key sha differs from pass 1")
            _, per_root[name] = fit_one(out, name, g1, producer_rev, base)
            del g1
        except Exception as exc:   # noqa: BLE001
            gate = {"root": name, "passed": False, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
        gates[name]["fit_pass"] = gate
        print(f"{name}: fit {'passed' if gate['passed'] else 'FAILED'}", flush=True)
        if not gate["passed"]:
            return _stop(out, {r: v for r, v in gates.items()}, "fit_pass", name, gate["error"], roots,
                         {"completed_roots": sorted(per_root)}), gates, per_root
    rid.write_json_atomic(out / "gate.json", {"passed": True, "roots": gates})
    return 0, gates, per_root


# ---------------------------------------------------------------- identity

def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def script_identity() -> dict:
    rel = SCRIPT.relative_to(REPO).as_posix()
    blob = _git("hash-object", rel)
    head_blob = _git("rev-parse", f"HEAD:{rel}")
    dirty = _git("status", "--porcelain", "--", rel)
    if blob != head_blob or dirty:
        raise RuntimeError(f"D55 script {rel} differs from HEAD (blob {blob} vs {head_blob}; status {dirty!r})")
    return {"path": rel, "git_blob": blob, "sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}


# ---------------------------------------------------------------- selftest (synthetic only)

def fit_call_sites() -> list:
    tree, sites = ast.parse(SCRIPT.read_text(encoding="utf-8")), []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and (node.func.value.id, node.func.attr) in
                        {("xgb", "train"), ("nx", "fit_global"), ("nx", "continue_booster")}):
                    sites.append(fn.name)
    return sites


def selftest() -> int:
    assert fit_call_sites() == ["fit_relative_root"], fit_call_sites()
    assert origin_index() == ORIGIN_INDEX
    # (1) all 16 (y, k) pairs: bijection, decode round-trip, crisis bit identity
    for y in range(4):
        for k in range(4):
            z = encode(np.array([y]), np.array([k]), np.array([True]))[0]
            assert z == y ^ k and ((z >> 1) & 1) == (((y >> 1) & 1) ^ ((k >> 1) & 1))
            q = np.zeros((1, 4))
            q[0, z] = 1.0
            assert decide(decode(q, np.array([k]), np.array([True])))[0] == y
    # (2) missing identity, no mutation
    y0, k0, kn0 = np.array([0, 1, 2, 3]), np.array([0, 0, 0, 0]), np.zeros(4, bool)
    yc = y0.copy()
    assert np.array_equal(encode(y0, k0, kn0), y0) and np.array_equal(y0, yc)
    Xm = np.full((4, 162), np.nan)
    k_m, kn_m = reference(Xm)
    assert not kn_m.any() and np.array_equal(k_m, np.zeros(4)) and np.isnan(Xm[:, ORIGIN_INDEX]).all()
    qq = np.random.default_rng(1).dirichlet(np.ones(4), 4)
    assert np.array_equal(decode(qq, k_m, kn_m), qq)
    # (3) tie case: decode first -> class 0; prohibited shortcut -> class 1
    q = np.array([[.4, .4, .1, .1]])
    assert decide(decode(q, np.array([1]), np.array([True])))[0] == 0
    assert (int(np.argmax(q[0])) ^ 1) == 1
    # (4) tiny synthetic fit with absent transformed classes -> 4-column softprob, exact reload
    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(300, 162))
    Xs[:, ORIGIN_INDEX] = rng.choice([1.0, 2.0, np.nan], size=300)
    k_s, kn_s = reference(Xs)
    ys = np.where(kn_s, k_s, 0)                     # z = 0 everywhere: classes 1..3 absent in z
    zs = encode(ys, k_s, kn_s)
    assert set(np.unique(zs)) == {0}
    plan.G_CONFIGS["_SELFTEST"] = {**copy.deepcopy(plan.G_CONFIGS["G1"]), "rounds": 3}
    try:
        booster, record = fit_relative_root(Xs, zs, "_SELFTEST")
        check_fit(booster, record, "_SELFTEST", len(zs))
    finally:
        del plan.G_CONFIGS["_SELFTEST"]
    q_live = nx.proba(booster, Xs)
    assert q_live.shape == (300, 4)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "relative_root.ubj"
        path.write_bytes(nx.raw(booster))
        frozen = nx.from_raw(path.read_bytes())
    q_re = nx.proba(frozen, Xs)
    assert np.array_equal(q_re, q_live) and np.array_equal(decode(q_re, k_s, kn_s), decode(q_live, k_s, kn_s))
    # (5) later-root pass-1 failure -> zero fit calls
    calls = []
    real_gate, real_fit = globals()["gate_one"], globals()["fit_relative_root"]

    def fake_gate(run, stage, name, pair, *rest):
        if name == "r2":
            raise GateError("synthetic later-root failure")
        return {"root": name, "checks": {}, "passed": True}, {"fit_sha": "sha-" + name}

    globals()["gate_one"] = fake_gate
    globals()["fit_relative_root"] = lambda *a, **k: calls.append(a)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cands = {f"c{i}": {"root": r, "e1": "hard_f1"} for i, r in enumerate(("r", "r2", "r3"))}
            code, gates, per_root = run_all(None, None, out, ["r", "r2", "r3"], cands, "7b2bf6f", None,
                                            None, None, None)
            failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
            assert code == 2 and not calls and not per_root
            assert failure["failed_root"] == "r2" and failure["not_attempted"] == ["r3"]
    finally:
        globals()["gate_one"], globals()["fit_relative_root"] = real_gate, real_fit
    # (6) exact-Fraction screen strictness: equality fails
    def root(h, rel, orig, pers):
        mk = lambda tp, fp, fn: {"n": tp + fp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": 0,
                                 "crisis_f1_exact": str(f1({"tp": tp, "fp": fp, "fn": fn}))}
        return {"horizon": h, "scores": {"E3": {"matched": {"relative": mk(*rel), "original": mk(*orig),
                                                            "persistence": mk(*pers)}}}}
    better = {f"{h}_{i}": root(h, (6, 1, 1), (5, 2, 2), (4, 2, 2)) for h in HORIZONS for i in range(7)}
    assert screen(better)["screen_pass"] and screen(better)["n_true"] == 12
    tied = dict(better)
    for i in range(7):
        tied[f"4_{i}"] = root(4, (6, 1, 1), (6, 1, 1), (4, 2, 2))
    s = screen(tied)
    assert not s["screen_pass"] and s["n_true"] == 10
    assert not s["table"]["H4"]["comparisons"]["pooled_relative_gt_original"]
    # (7) D38 reference: missing unchanged, known renormalised
    pp = posthoc_d38(qq, np.array([0, 1, 0, 0]), np.array([False, True, False, False]))
    assert np.array_equal(pp[[0, 2, 3]], qq[[0, 2, 3]]) and abs(pp[1].sum() - 1) < 1e-12
    print("SELFTEST OK")
    return 0


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d34-run", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--d52-run", type=Path, default=DEFAULT_D52)
    parser.add_argument("--producer-rev", default="7b2bf6f")
    parser.add_argument("--selftest", action="store_true", help="synthetic checks only; no real data or fit")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.d34_run is None:
        parser.error("--d34-run is required unless --selftest")
    run, out, d52 = args.d34_run.resolve(), args.out.resolve(), args.d52_run.resolve()
    if out.is_relative_to(run):
        raise ValueError("--out must not be inside the read-only D34 run")
    rid.refuse_existing(out, "D55 relative-coordinate root")
    origin_index()
    script = script_identity()
    producer = rid.code_identity_at(args.producer_rev)
    if rid.code_identity() != rid.code_identity_at("HEAD"):
        raise RuntimeError("D55 must run from committed package code (working tree differs from HEAD)")
    roots, cands, ident = accept_mode(run, plan.E1PAIR, producer, args.producer_rev)
    stage = Path(ident["stage"])
    d49 = json.loads(D49_SUMMARY.read_text(encoding="utf-8"))
    d54 = pd.read_csv(D54_PER_ROOT)
    if len(set(roots)) != 21 or set(roots) != set(d49["per_root"]):
        raise GateError("accepted 21-root inventory differs from D49 per_root")
    d52_check = check_d52(d52)
    base_path = run / "prepared" / "ledgers" / "dev_baselines.csv"
    base = pd.read_csv(base_path)
    identity = {"stage": "d55_relative_coordinate_root", "d34_run": str(run), "producer_rev": args.producer_rev,
                "producer_code": producer, "repo_head": _git("rev-parse", "HEAD"), "script": script,
                "package_code": rid.code_identity(), "runtime": rid.runtime_identity(), "max_month": "2020-12",
                "g_by_horizon": G_BY_H, "g_configs": {g: plan.G_CONFIGS[g] for g in sorted(set(G_BY_H.values()))},
                "mapping_version": MAPPING_VERSION,
                "origin_source": {"feature": ORIGIN_FEATURE, "schema_index": ORIGIN_INDEX},
                "d38_q": {"origin": Q_ORIGIN, "other": Q_OTHER}, "d52": d52_check,
                "consistency_sources": {"d49": rid.file_sha256(D49_SUMMARY), "d54_per_root": rid.file_sha256(D54_PER_ROOT),
                                        "d52_completion": rid.file_sha256(D52_COMPLETION)},
                "dev_baselines": {"path": str(base_path), "sha256": rid.file_sha256(base_path)},
                "acceptance": {k: (str(v) if isinstance(v, Path) else v) for k, v in ident.items()}}
    out.mkdir(parents=True)
    rid.write_json_atomic(out / "identity.json", identity)          # before any fit
    code, gates, per_root = run_all(run, stage, out, roots, cands, args.producer_rev, base, d49, d54, d52)
    if code:
        print("STOPPED at the first failure: see gate.json / failure.json")
        return code
    summary = {"screen": screen(per_root), "per_root": per_root,
               "interpretation": ("Primary: matched-E3 crisis F1 (decoded argmax >= 2), 12 strict exact-Fraction "
                                  "comparisons; floats display only. Secondary: normalised crisis Brier, unclipped "
                                  "four-class log loss on normalised rows, macro-F1, fold wins. FIT in-sample, C "
                                  "interpolation; all-key and missing-origin E3 model arms separately. D38 post-hoc "
                                  "is an E3 reference only. Diagnostic; no adoption; no D56.")}
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "completion.json", {"status": "completed", "outputs": rid.output_hashes(out)})
    print(f"D55 relative-coordinate root completed: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
