"""D38 fixed add-one transition-prior diagnostic; no XGBoost fits or production imports."""
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\swl00\geoxgb_runs")
D34 = BASE / "geoxgb-d34-e1-brier-20261002"
D37 = BASE / "geoxgb-d37-recency-root-20261002"
OUT = BASE / "d38_prior_support.json"
assert not OUT.exists(), "Keep existing diagnostic evidence"
TARGETS = ("2018-06", "2018-10", "2019-02", "2019-06", "2019-10", "2020-02", "2020-06")


def month(s):
    return int(s[:4]) * 12 + int(s[5:]) - 1


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def score(t, y, p):
    z, a = np.asarray(t) >= 2, np.asarray(y) >= 2
    tp, fp, fn, tn = (int(v.sum()) for v in (z & a, ~z & a, z & ~a, ~z & ~a))
    f = Fraction(2 * tp, 2 * tp + fp + fn) if tp + fp + fn else Fraction(0)
    return {"n": len(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "f1": float(f), "f1_exact": str(f),
            "brier": float(np.mean((p[:, 2:].sum(axis=1) - z) ** 2)) if len(t) else None}


results, pooled = [], []
input_hashes = {}
for h, g in [(4, "G1"), (8, "G4"), (12, "G2")]:
    sp = D34 / "prepared" / f"snapshot_h{h}.parquet"
    input_hashes[str(sp)] = sha(sp)
    snap = pd.read_parquet(sp, columns=["area", "target_month", "class_code", "hist_phase_o00"],
                           filters=[("target_month", "<=", month("2020-12"))]).set_index(["area", "target_month"])
    assert not snap.index.duplicated().any()
    for target in TARGETS:
        name = f"h{h}_{target}_{g}_r80_s42_e1pair"
        rd = D34 / "stage1_e1pair" / "roots" / name
        mem = pd.read_csv(rd / "fold_membership.csv.gz", float_precision="round_trip")
        fit = mem[mem.role == "fitting"]
        root = json.loads((rd / "root.json").read_text())
        fm = np.array([month(s) for s in fit.target_month], dtype=np.int64)
        keys = np.column_stack([fit.area.to_numpy(dtype=np.int64), fm])
        assert hashlib.sha256(keys.tobytes()).hexdigest() == root["fitting_keys_sha256"]
        assert np.all((fm >= month(target) - h - 59) & (fm < month(target) - h))
        data = snap.loc[list(map(tuple, keys))]
        t = data.class_code.to_numpy(dtype=int)
        assert np.array_equal(t, fit.class_code)
        phase = data.hist_phase_o00.to_numpy(dtype=float)
        known = np.isfinite(phase)
        assert np.isin(phase[known], [1, 2, 3, 4]).all()
        origin = phase[known].astype(int) - 1
        counts = np.zeros((4, 4), dtype=np.int64)
        np.add.at(counts, (origin, t[known]), 1)
        q = (counts + 1) / (counts.sum(axis=1, keepdims=True) + 4)
        unconditional = np.bincount(t, minlength=4)
        fallback = (unconditional + 1) / (len(t) + 4)
        support = {}
        for c in range(4):
            mask = known & (phase == c + 1)
            support[str(c)] = {"rows": int(mask.sum()), "areas": int(fit.area.to_numpy()[mask].size and np.unique(fit.area.to_numpy()[mask]).size),
                               "label_months": int(np.unique(fm[mask]).size)}
        e = pd.read_csv(D37 / name / "rows_E3.csv.gz", float_precision="round_trip")
        desired = mem[mem.role == "heldout_target"]
        assert list(zip(e.area, e.target_month)) == list(zip(desired.area, desired.target_month))
        ek = list(zip(e.area, [month(s) for s in e.target_month]))
        ed = snap.loc[ek]
        assert np.array_equal(ed.class_code, e.truth)
        per = ed.hist_phase_o00.to_numpy(float) - 1
        assert np.allclose(per, e.persistence_code, equal_nan=True)
        known_e = np.isfinite(per)
        probs = np.tile(fallback, (len(e), 1))
        probs[known_e] = q[per[known_e].astype(int)]
        assert np.allclose(probs.sum(axis=1), 1)
        e["y_prior"] = probs.argmax(axis=1)
        for c in range(4):
            e[f"prior_{c}"] = probs[:, c]
        e["root"] = name
        original_p = e[[f"p_original_{c}" for c in ("1", "2", "3", "4或5")]].to_numpy()
        metrics = {"all": {"original": score(e.truth, e.y_original, original_p), "prior": score(e.truth, e.y_prior, probs)},
                   "matched": {"original": score(e.truth[known_e], e.y_original[known_e], original_p[known_e]),
                               "prior": score(e.truth[known_e], e.y_prior[known_e], probs[known_e]),
                               "persistence": score(e.truth[known_e], per[known_e], np.eye(4)[per[known_e].astype(int)])}}
        results.append({"root": name, "horizon": h, "target": target, "fitting_rows": len(fit),
                        "fitting_keys_sha256": root["fitting_keys_sha256"], "fit_known_origin_rows": int(known.sum()),
                        "fit_missing_origin_share": float((~known).mean()), "origin_support": support,
                        "fit_target_year_counts": dict(sorted(Counter(map(int, fm // 12)).items())),
                        "known_origin_target_year_counts": dict(sorted(Counter(map(int, fm[known] // 12)).items())),
                        "known_origin_year_counts": dict(sorted(Counter(map(int, (fm[known] - h) // 12)).items())),
                        "counts_origin_target": counts.tolist(), "q": q.tolist(), "fallback": fallback.tolist(),
                        "prior_argmax_by_origin": q.argmax(axis=1).tolist(), "e3_missing_origin_rows": int((~known_e).sum()),
                        "scores": metrics, "membership_sha256": sha(rd / "fold_membership.csv.gz")})
        pooled.append(e)

frame = pd.concat(pooled, ignore_index=True)
overall = {}
for part, f in [("all", frame), ("matched", frame[frame.persistence_code.notna()])]:
    overall[part] = {"original": score(f.truth, f.y_original, f[[f"p_original_{c}" for c in ("1", "2", "3", "4或5")]].to_numpy()),
                     "prior": score(f.truth, f.y_prior, f[[f"prior_{c}" for c in range(4)]].to_numpy())}
    if part == "matched":
        overall[part]["persistence"] = score(f.truth, f.persistence_code, np.eye(4)[f.persistence_code.astype(int)])
out = {"rule": "Fixed add-one fitting-only transition counts; missing-origin fallback add-one full-fitting prior; no XGBoost fits, no production imports, no alternative alpha or tuning. E3 exposed development only.",
       "script_sha256": sha(Path(__file__)), "snapshot_hashes": input_hashes, "per_root": results, "overall": overall}
OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(OUT)
print(json.dumps(overall, indent=2))
print("known origin fitting shares", min(1-r["fit_missing_origin_share"] for r in results), max(1-r["fit_missing_origin_share"] for r in results))
print("prior argmax rows", Counter(tuple(r["prior_argmax_by_origin"]) for r in results))
