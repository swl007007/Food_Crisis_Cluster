"""Stage 1 generalisation diagnostics for completed roots (D26 review evidence; never fits).

python scripts/stage1_diagnostics.py --run-dir RUN [--roots name1,name2,...]

For every completed root (completion record and every recorded output hash checked;
producer code/runtime must equal the current ones) and each of its candidates:

* E2 total gain: final partition vs root on ALL of the candidate's validation rows
  (the E2 population), recomputed from validation_predictions.csv.gz;
* E3 gain: final partition vs the candidate's root on the target month, recomputed from
  target_predictions.csv / root_target_predictions.csv;

both for the primary crisis-positive F1 (four-class argmax collapsed to IPC >= 3) and
the secondary fixed-four macro F1; plus the summed recorded decision gains, terminal
counts and pooled four-class / binary confusion matrices. Writes RUN/stage1_diagnostics/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from scripts.run_stage1 import CANDIDATE_FILES, ROOT_FILES  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402
from src.utils.run_identity import write_json_atomic  # noqa: E402


def read(path, str_cols=()):
    return pd.read_csv(path, float_precision="round_trip", low_memory=False, converters={c: str for c in str_cols})


def accepted_root(stage1: Path, name: str) -> dict:
    record = json.loads((stage1 / "roots" / name / "completion.json").read_text(encoding="utf-8"))
    if record.get("code") != rid.code_identity() or record.get("runtime") != rid.runtime_identity():
        raise RuntimeError(f"{name}: produced by other code/runtime")
    required = [f"roots/{name}/{f}" for f in ROOT_FILES]
    if record["status"] == "completed":
        required += [f"candidates/{c}/{f}" for c in record["candidates"] for f in CANDIDATE_FILES]
    problems = rid.check_inventory(stage1, record["outputs"], required)
    if problems:
        raise RuntimeError(f"{name}: {problems[:3]}")
    return record


def confusion4(t, p):
    return np.bincount(np.asarray(t, int) * 4 + np.asarray(p, int), minlength=16).reshape(4, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--roots", default="")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    stage1 = run / "stage1"
    out = run / "stage1_diagnostics"
    if out.exists():
        raise FileExistsError(f"{out} exists")
    names = [n for n in args.roots.split(",") if n] or sorted(
        p.name for p in (stage1 / "roots").iterdir() if (p / "completion.json").is_file())
    rows, conf = [], {}
    for name in names:
        record = accepted_root(stage1, name)
        if record["status"] != "completed":
            rows.append({"root": name, "status": record["status"]})
            continue
        root = json.loads((stage1 / "roots" / name / "root.json").read_text(encoding="utf-8"))
        pooled = read(stage1 / "roots" / name / "root_target_predictions.csv")
        for cand in record["candidates"]:
            d = stage1 / "candidates" / cand
            c = json.loads((d / "candidate.json").read_text(encoding="utf-8"))
            tp = read(d / "target_predictions.csv", ("branch_id",))
            vp = read(d / "validation_predictions.csv.gz", ("branch_id",))
            t_true, t_part, t_pool = tp.y_true_code, tp.y_pred_partitioned_code, pooled.y_pred_pooled_code
            v_true, v_final, v_root = vp.y_true, vp.y_final, vp.y_root
            e3c = fourclass.crisis_f1(t_true, t_part) - fourclass.crisis_f1(t_true, t_pool)
            e3f = fourclass.macro_f1(t_true, t_part) - fourclass.macro_f1(t_true, t_pool)
            e2c = fourclass.crisis_f1(v_true, v_final) - fourclass.crisis_f1(v_true, v_root)
            e2f = fourclass.macro_f1(v_true, v_final) - fourclass.macro_f1(v_true, v_root)
            if not (np.isclose(e3c, c["scores"]["score"] - c["scores"]["score_base"], rtol=0, atol=1e-12)):
                raise RuntimeError(f"{cand}: recorded E3 score does not recompute")
            decisions = c["partition"]["decisions"]
            rows.append({
                "root": name, "candidate": cand, "status": "completed", "horizon": root["horizon"],
                "target_month": root["target_month"], "ratio": root["ratio"], "split_seed": root["split_seed"],
                "local_config": c["local_config"], "threshold_family": c["threshold_family"],
                "n_terminal": c["partition"]["n_terminal"],
                "accepted_splits": sum(x["outcome"] == "accepted" for x in decisions),
                "decisions": len(decisions), "child_fits": c["fits"]["child_fits"],
                "e2_recorded_gain_sum": float(sum(float(__import__("fractions").Fraction(x["gain"]))
                                                  for x in decisions if x["outcome"] == "accepted")),
                "val_rows": len(vp), "target_rows": len(tp),
                "e2_crisis_gain": e2c, "e3_crisis_gain": e3c, "e2_fourclass_gain": e2f, "e3_fourclass_gain": e3f,
                "e3_crisis_part": fourclass.crisis_f1(t_true, t_part), "e3_crisis_root": fourclass.crisis_f1(t_true, t_pool),
                "e3_fourclass_part": fourclass.macro_f1(t_true, t_part), "e3_fourclass_root": fourclass.macro_f1(t_true, t_pool),
                "val_crisis_final": fourclass.crisis_f1(v_true, v_final), "val_crisis_root": fourclass.crisis_f1(v_true, v_root),
            })
            key = (c["threshold_family"], c["local_config"])
            acc4 = conf.setdefault(key, {"target_part": np.zeros((4, 4), int), "target_root": np.zeros((4, 4), int),
                                         "val_final": np.zeros((4, 4), int), "val_root": np.zeros((4, 4), int)})
            acc4["target_part"] += confusion4(t_true, t_part)
            acc4["target_root"] += confusion4(t_true, t_pool)
            acc4["val_final"] += confusion4(v_true, v_final)
            acc4["val_root"] += confusion4(v_true, v_root)
    out.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "candidates.csv", index=False, float_format="%.17g")
    done = frame[frame.status == "completed"]

    def summarise(g):
        split = g[g.n_terminal > 1]
        e2pos = g[g.e2_crisis_gain > 0]
        return {"candidates": int(len(g)), "split": int(len(split)),
                "mean_e2_crisis": float(g.e2_crisis_gain.mean()), "mean_e3_crisis": float(g.e3_crisis_gain.mean()),
                "mean_e2_fourclass": float(g.e2_fourclass_gain.mean()), "mean_e3_fourclass": float(g.e3_fourclass_gain.mean()),
                "e3_crisis_positive": int((g.e3_crisis_gain > 0).sum()),
                "e2_positive": int(len(e2pos)), "e2_positive_e3_negative": int((e2pos.e3_crisis_gain < 0).sum()),
                "spearman_e2_e3_crisis": float(g.e2_crisis_gain.corr(g.e3_crisis_gain, method="spearman"))
                if len(g) > 2 and g.e2_crisis_gain.nunique() > 1 and g.e3_crisis_gain.nunique() > 1 else None}
    summary = {"endpoint": "crisis_f1 (four-class argmax collapsed to IPC>=3); fixed-four macro F1 secondary",
               "roots": names, "overall": summarise(done) if len(done) else None,
               "by_family_local": {f"{f}/{l}": summarise(g) for (f, l), g in done.groupby(["threshold_family", "local_config"])},
               "by_horizon": {str(h): summarise(g) for h, g in done.groupby("horizon")},
               "by_target": {t: summarise(g) for t, g in done.groupby("target_month")},
               "by_ratio": {r: summarise(g) for r, g in done.groupby("ratio")}}
    conf_rows = []
    for (f, l), mats in conf.items():
        for which, m in mats.items():
            crisis = np.array([[m[:2, :2].sum(), m[:2, 2:].sum()], [m[2:, :2].sum(), m[2:, 2:].sum()]])
            conf_rows.append({"family": f, "local": l, "which": which, "fourclass": m.tolist(),
                              "crisis_[[nn,nc],[cn,cc]]": crisis.tolist(),
                              "per_class_f1": np.round(2 * np.diag(m) / np.maximum(m.sum(0) + m.sum(1), 1), 4).tolist(),
                              "crisis_f1": float(2 * crisis[1, 1] / max(2 * crisis[1, 1] + crisis[0, 1] + crisis[1, 0], 1))})
    summary["pooled_confusion"] = conf_rows
    write_json_atomic(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("overall", "by_family_local", "by_horizon")}, indent=1))


if __name__ == "__main__":
    main()
