#!/usr/bin/env python3
"""Phase 5: apply the frozen tau to 2021-2024 ONCE and adjudicate.

Runs strictly after Phase 4's freeze. The R14 consistency diagnostic looks at
test data and therefore runs last; it never feeds back into tau.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from persistencecorrection.override import apply_override, flip_report
from persistencecorrection.selection import crisis_f1, select_threshold, sha256_file

ROOT = Path(__file__).resolve().parent
PHASE3, PHASE4 = ROOT / "outputs/phase3_20260918", ROOT / "outputs/phase4_20260919"
OUT = ROOT / "outputs/phase5_20260919"; OUT.mkdir(parents=True, exist_ok=False)
REPO = ROOT.parent
EXPERT = REPO / "Step3ExpertCorrectionExperiment/outputs/full_fs1_fs2_up_only_20260918/predictions_monthly_correction.csv"
RNG = np.random.default_rng(5)

frozen = json.loads((PHASE4 / "frozen_thresholds.json").read_text())
for s in (1, 2):  # the frozen artifact must still match its recorded inputs
    assert frozen["selection_inputs_sha256"][f"fs{s}"] == sha256_file(
        PHASE3 / f"calibrated_selection_2020_fs{s}.csv"), f"fs{s} selection input changed"

exp_all = pd.read_csv(EXPERT, usecols=["scope", "admin_code", "month_start", "expert"])
report = {"frozen_thresholds": {k: v["tau"] for k, v in frozen["frozen_thresholds"].items()}}

for scope in (1, 2):
    tau = frozen["frozen_thresholds"][f"fs{scope}"]["tau"]
    d = pd.read_csv(PHASE3 / f"calibrated_test_2021_2024_fs{scope}.csv")
    e = exp_all[exp_all.scope == scope].rename(columns={"admin_code": "FEWSNET_admin_code"})
    d = d.merge(e[["FEWSNET_admin_code", "month_start", "expert"]],
                on=["FEWSNET_admin_code", "month_start"], how="left", validate="one_to_one")
    assert d.expert.notna().all(), "expert join incomplete"

    y, pe = d.y_true.values.astype(int), d.persistence.values.astype(int)
    d["y_override"] = apply_override(pe, d.p_cal.values, tau)

    models = {
        "persistence": pe,
        "two_layer_override": d.y_override.values,
        "georf_partitioned_uncalibrated": d.y_pred_partitioned.values.astype(int),
        "georf_calibrated_standalone": (d.p_cal.values > 0.5).astype(int),
        "expert": d.expert.values.astype(int),
    }
    base = crisis_f1(y, pe)
    sc = {k: {"f1": crisis_f1(y, v), "delta_vs_persistence": crisis_f1(y, v) - base}
          for k, v in models.items()}

    # fold-level bootstrap and leave-one-fold-out on the override's delta
    d["fold"] = d.month_start
    folds = sorted(d.fold.unique())
    def delta(frame):
        return crisis_f1(frame.y_true, frame.y_override) - crisis_f1(frame.y_true, frame.persistence)
    boots = []
    for _ in range(2000):
        pick = RNG.choice(folds, size=len(folds), replace=True)
        boots.append(delta(pd.concat([d[d.fold == f] for f in pick], ignore_index=True)))
    boots = np.array(boots)
    lofo = {f: float(delta(d[d.fold != f])) for f in folds}

    fr = flip_report(pe, d.y_override.values, y)
    # pre-registered diagnostics, no candidate status
    down = pe.copy(); down[(pe == 1) & (d.p_cal.values < (1 - tau))] = 0
    raw_sel = select_threshold(d.persistence, d.y_prob_partitioned, d.y_true)
    r14 = select_threshold(d[d.fold.isin(folds[:6])].persistence,
                           d[d.fold.isin(folds[:6])].p_cal,
                           d[d.fold.isin(folds[:6])].y_true)

    report[f"fs{scope}"] = {
        "tau": tau, "n": int(len(d)), "scores": sc,
        "override_delta": sc["two_layer_override"]["delta_vs_persistence"],
        "bootstrap": {"mean": float(boots.mean()), "sd": float(boots.std(ddof=1)),
                      "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                      "p_gt_0": float((boots > 0).mean()), "draws": 2000},
        "lofo": lofo, "lofo_min": float(min(lofo.values())), "lofo_max": float(max(lofo.values())),
        "lofo_sign_flips": [f for f, v in lofo.items()
                            if np.sign(v) != np.sign(sc["two_layer_override"]["delta_vs_persistence"])],
        "flips": fr,
        "diagnostics_no_candidate_status": {
            "down_flip_variant_f1": crisis_f1(y, down),
            "raw_probability_variant": {"best_tau_on_test": raw_sel["tau"],
                                        "best_f1_on_test": raw_sel["selected_f1"]},
            "r14_consistency_threshold_first6_folds": r14["tau"],
        },
    }
    d.to_csv(OUT / f"predictions_2layer_fs{scope}.csv", index=False)

(OUT / "adjudication.json").write_text(json.dumps(report, indent=2, sort_keys=True))
for s in (1, 2):
    r = report[f"fs{s}"]
    print(f"\n===== fs{s}  tau={r['tau']:.4f}  n={r['n']}")
    for k, v in r["scores"].items():
        print(f"   {k:34s} F1={v['f1']:.4f}  delta={v['delta_vs_persistence']:+.4f}")
    b = r["bootstrap"]
    print(f"   delta={r['override_delta']:+.4f} | CI95=[{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] "
          f"sd={b['sd']:.4f} P(>0)={b['p_gt_0']:.3f}")
    print(f"   LOFO range=[{r['lofo_min']:+.4f},{r['lofo_max']:+.4f}] sign-flipping folds={r['lofo_sign_flips']}")
    print(f"   flips={r['flips']}")
    print(f"   diagnostics={r['diagnostics_no_candidate_status']}")
