"""D28 keyed comparison: six shared-root increment candidates vs the six D26 r80 controls (never fits).

python scripts/stage1_rootinc_compare.py --run-dir RUN --control-run D26_RUN [--control-code-rev 87513eb]

Reuses the D27 comparison's acceptance and keyed joins (scripts/stage1_tb3_compare.py).
Both arms use the same r80/seed42 split, G and L1/gt0, so in addition to identical target
keys/truth it REQUIRES identical fitting/validation membership keys, identical root target
predictions and the same root booster SHA-256; a difference stops the comparison.
Writes RUN/stage1_rootinc_compare/.
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from scripts import stage1_tb3_compare as tc  # noqa: E402
from scripts.run_stage1 import SPLIT_MODES, scheduled_candidates, scheduled_roots  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.utils import acceptance as acc  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402

CompareError = tc.CompareError
KEY = ["area", "target_month"]


def accept_rootinc(run: Path):
    prepared = acc.accept_prepared(run)
    g_of, g_record = acc.accept_g_selection(run)
    if dict(g_of) != plan.TB3_G:
        raise CompareError(f"rootinc requires the locked G {plan.TB3_G}")
    sched = acc.schedule(run)
    roots, cands = scheduled_roots(sched, g_of, plan.ROOTINC), scheduled_candidates(sched, g_of, plan.ROOTINC)
    if len(roots) != 6 or len(cands) != 6:
        raise CompareError("the schedule does not give exactly six rootinc roots / candidates")
    stage = run / SPLIT_MODES[plan.ROOTINC][2]
    present = {p.name for p in (stage / "roots").iterdir() if p.is_dir()} if (stage / "roots").is_dir() else set()
    if present != set(roots):
        raise CompareError(f"rootinc roots: missing {sorted(set(roots) - present)}, unexpected {sorted(present - set(roots))}")
    cand_dirs = {p.name for p in (stage / "candidates").iterdir() if p.is_dir()} if (stage / "candidates").is_dir() else set()
    if cand_dirs != set(cands):
        raise CompareError(f"rootinc candidates: dirs {sorted(cand_dirs)} != scheduled {sorted(cands)}")
    code, runtime = rid.code_identity(), rid.runtime_identity()
    for name, entry in roots.items():
        want = sorted(c for c, e in cands.items() if e["root"] == name)
        if want != [plan.rootinc_candidate_name(entry["horizon"], entry["target_month"], entry["g_config"])]:
            raise CompareError(f"{name}: expected exactly its single L1/gt0 rootinc candidate, got {want}")
        record = tc.accept_root_record(stage, name, want)
        if record.get("prepared") != prepared["outputs_sha256"] or record.get("g_selection") != g_record:
            raise CompareError(f"{name}: produced on another preparation or G selection")
        if record.get("code") != code or record.get("runtime") != runtime:
            raise CompareError(f"{name}: produced by other package code or runtime")
        root = tc._json(stage / "roots" / name / "root.json")
        if (root.get("increment_source"), root.get("ratio"), root.get("split_seed"), root.get("horizon"),
                root.get("target_month"), root.get("g_config")) != \
                ("root", plan.ROOTINC_RATIO, plan.ROOTINC_SEED, entry["horizon"], entry["target_month"], entry["g_config"]):
            raise CompareError(f"{name}: root.json does not describe the scheduled rootinc root")
        for c in want:
            if tc._json(stage / "candidates" / c / "candidate.json").get("increment_source") != "root":
                raise CompareError(f"{c}: candidate is not a shared-root increment candidate")
    return roots, cands, {"prepared": prepared["outputs_sha256"], "g_selection": g_record, "g_of": g_of,
                          "code": code, "runtime": runtime}


def same_roots(new_stage, new_root, old_stage, old_root, t_new, t_old) -> None:
    """Fitting/validation membership, target truth, root target predictions, root booster."""
    a = tc.read(new_stage / "roots" / new_root / "fold_membership.csv.gz")
    b = tc.read(old_stage / "roots" / old_root / "fold_membership.csv.gz")
    m = a.merge(b, on=KEY, how="outer", suffixes=("_new", "_old"), indicator=True, validate="one_to_one")
    if (m["_merge"] != "both").any() or not (m["role_new"] == m["role_old"]).all() \
            or not (m["class_code_new"] == m["class_code_old"]).all():
        raise CompareError(f"{new_root}: fitting/validation/target membership differs from the control")
    j = t_new.merge(t_old, on=KEY, how="outer", suffixes=("_new", "_old"), indicator=True, validate="one_to_one")
    if (j["_merge"] != "both").any() or not (j["y_true_new"] == j["y_true_old"]).all() \
            or not (j["y_root_new"] == j["y_root_old"]).all():
        raise CompareError(f"{new_root}: target keys/truth/root predictions differ from the control")
    ra, rb = (tc._json(s / "roots" / r / "root.json")["root_booster_sha256"]
              for s, r in ((new_stage, new_root), (old_stage, old_root)))
    if ra != rb:
        raise CompareError(f"{new_root}: root booster differs from the control ({ra} vs {rb})")


def round_columns(stage: Path, cand: str) -> dict:
    """Rounds DEPLOYED after the root in the routed terminal models (rounds_total − root
    rounds; both modes) and the search budget used (rootinc: path_selection_rounds;
    parent mode: cumulative path_rounds_added, which is also its deployed depth)."""
    c = tc._json(stage / "candidates" / cand / "candidate.json")
    saved = {e["saved_as"]: e for e in c["fits"]["saved_log"]}
    root_rounds = int(saved["root"]["rounds_total"])
    terminal = [saved[b] for b in c["partition"]["terminal_partitions"]]
    return {"deployed_rounds_after_root_max": max(int(e["rounds_total"]) - root_rounds for e in terminal),
            "search_budget_rounds_max": max(int(e.get("path_selection_rounds", e.get("path_rounds_added", 0)))
                                            for e in terminal),
            "fits_attempted": sum(1 for e in c["fits"]["fit_log"] if e.get("kind") == "continuation")}


def crisis_counts(conf, label) -> dict:
    cm = tc.crisis_matrix(conf)
    return {f"{label}_tp": cm[1][1], f"{label}_fp": cm[0][1], f"{label}_fn": cm[1][0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--control-code-rev", default="87513eb", help="committed producer of the D26 controls")
    args = parser.parse_args()
    run, control = args.run_dir.resolve(), args.control_run.resolve()
    out = run / "stage1_rootinc_compare"
    rid.refuse_existing(out, "the rootinc comparison")
    roots, cands, ident = accept_rootinc(run)
    controls, control_ident = tc.accept_control(control, run, ident["g_of"], args.control_code_rev)
    stage = run / SPLIT_MODES[plan.ROOTINC][2]
    rows, pairs, partitions, confusions = [], [], {}, []
    for name, entry in sorted(roots.items(), key=lambda kv: (kv[1]["horizon"], kv[1]["target_month"])):
        h, t = entry["horizon"], entry["target_month"]
        cand = next(c for c, e in cands.items() if e["root"] == name)
        ctl = controls[(h, t)]
        rb = tc._json(stage / "roots" / name / "root.json")["root_booster_sha256"]
        rb_ctl = tc._json(ctl["stage"] / "roots" / ctl["root"] / "root.json")["root_booster_sha256"]
        new, t_new, c_new, p_new = tc.method_row(stage, name, cand, h, t, plan.ROOTINC, rb)
        old, t_old, c_old, p_old = tc.method_row(ctl["stage"], ctl["root"], ctl["candidate"], h, t, "r80_control", rb_ctl)
        same_roots(stage, name, ctl["stage"], ctl["root"], t_new, t_old)
        for row, conf, st, cn in ((new, c_new, stage, cand), (old, c_old, ctl["stage"], ctl["candidate"])):
            row.update(round_columns(st, cn), **crisis_counts(conf["target_root"], "e3_root"),
                       **crisis_counts(conf["target_local"], "e3_local"))
            rows.append(row)
        partitions[new["candidate"]], partitions[old["candidate"]] = p_new, p_old
        for method, conf in ((plan.ROOTINC, c_new), ("r80_control", c_old)):
            for which, mtx in conf.items():   # E2 = validation_*, E3 = target_*
                confusions.append({"method": method, "horizon": h, "target_month": t, "which": which,
                                   "fourclass": [[int(x) for x in r] for r in mtx], "crisis_[[nn,nc],[cn,cc]]": tc.crisis_matrix(mtx)})
        gain_new = Fraction(new["e3_local_minus_root_exact"])
        gain_old = Fraction(old["e3_local_minus_root_exact"])
        pairs.append({"horizon": h, "target_month": t, "rootinc_candidate": new["candidate"],
                      "r80_candidate": old["candidate"], "root_crisis_f1": new["e3_root_crisis_f1"],
                      "local_rootinc": new["e3_local_crisis_f1"], "local_r80": old["e3_local_crisis_f1"],
                      "e3_gain_rootinc": float(gain_new), "e3_gain_r80": float(gain_old),
                      "e3_gain_difference_rootinc_minus_r80": float(gain_new - gain_old),
                      "local_fourclass_difference": new["e3_local_fourclass"] - old["e3_local_fourclass"],
                      **{f"{k}_change": new[k] - old[k] for k in ("e3_local_tp", "e3_local_fp", "e3_local_fn")},
                      "e2_gain_rootinc": new["e2_final_minus_root"], "e2_gain_r80": old["e2_final_minus_root"],
                      "n_terminal_rootinc": new["n_terminal"], "n_terminal_r80": old["n_terminal"],
                      "deployed_rounds_max_rootinc": new["deployed_rounds_after_root_max"],
                      "deployed_rounds_max_r80": old["deployed_rounds_after_root_max"],
                      "identical_partition": p_new == p_old,
                      "e4_weight_rootinc": new["e4_weight"], "e4_weight_r80": old["e4_weight"]})
    frame, pair_frame = pd.DataFrame(rows), pd.DataFrame(pairs)

    def dup(names):
        by_cov, n = {}, 0
        for x in names:
            cov, labels = partitions[x]
            seen = by_cov.setdefault(cov, set())
            n += labels in seen
            seen.add(labels)
        return {"candidates": len(names), "coverages": len(by_cov), "same_coverage_duplicates": n}

    summary = {
        "contrast": "D28 (experiment-plan A3): shared-root single L1 increment vs D26 parent-accumulated r80; "
                    "r80, seed 42, L1, gt0, same H/T/G, identical roots",
        "this_run": {k: v for k, v in ident.items() if k != "g_of"}, "g_selected": ident["g_of"],
        "control": control_ident,
        "by_method": {m: {"mean_e3_local_minus_root": float(g["e3_local_minus_root"].mean()),
                          "e3_positive": int((g["e3_local_minus_root"] > 0).sum()),
                          "e3_negative": int((g["e3_local_minus_root"] < 0).sum()),
                          "split": int((g["n_terminal"] > 1).sum()),
                          "positive_e4_weights": int((g["e4_weight"] > 0).sum()),
                          "partitions": dup(list(g["candidate"]))} for m, g in frame.groupby("method")},
        "pairs": {"mean_e3_gain_difference_rootinc_minus_r80":
                  float(pair_frame["e3_gain_difference_rootinc_minus_r80"].mean()),
                  "identical_partitions": int(pair_frame["identical_partition"].sum())},
        "caveats": ["six related candidates, not six independent experiments; development evidence only",
                    "E2 reuses the random validation rows for search and acceptance in both arms",
                    "a root-only candidate (no split) is reported as such, not as a generalisation gain"],
    }
    out.mkdir(parents=True)
    frame.to_csv(out / "candidates.csv", index=False, float_format="%.17g")
    pair_frame.to_csv(out / "pairs.csv", index=False, float_format="%.17g")
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "confusions.json", confusions)
    rid.write_json_atomic(out / "completion.json", {
        "stage": "stage1_rootinc_compare", "code": rid.code_identity(), "runtime": rid.runtime_identity(),
        "outputs": {rel: sha for rel, sha in rid.output_hashes(out).items() if rel != "completion.json"}})
    print(json.dumps({k: summary[k] for k in ("by_method", "pairs")}, indent=1))


if __name__ == "__main__":
    main()
