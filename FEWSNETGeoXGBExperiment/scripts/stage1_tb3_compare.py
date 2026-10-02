"""D27 keyed comparison: six tb3 time-block candidates vs the six D26 r80 controls (never fits).

python scripts/stage1_tb3_compare.py --run-dir RUN --control-run D26_RUN [--control-code-rev 87513eb]

Accepts (experiment-plan A2 section 6):

* the six tb3 roots of RUN/stage1_tb3: completion record, current preparation, G
  selection, producer code/runtime, every recorded output hash, candidate list equal to
  the schedule, validation months equal to the frozen A2 table;
* read-only, the six D26 controls (r80, seed 42, L1, gt0, same H/T/G) of the control run:
  completion record, every recorded output hash, its own preparation and G selection
  records, producer code equal to the committed ``--control-code-rev``, and snapshot /
  development-truth hashes equal to this run's.

Every comparison joins by explicit keys, never by row order: target rows by
(area, target month T), validation rows by (area, label month). Target keys and truth
must be identical between candidate and root files and between the two methods.
Recomputes per method E3 crisis F1 (root, local, local-root), E2 (final vs root on the
method's own validation rows), cross-method differences and the root change; reports
support, terminals, distinct boosters, coverage-aware canonical partition duplicates,
the E4 weight and four-class / binary confusions. E2 populations differ between methods,
so their E2 difference is not a paired effect. Writes RUN/stage1_tb3_compare/.
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from scripts.run_stage1 import CANDIDATE_FILES, ROOT_FILES, SPLIT_MODES, scheduled_candidates, scheduled_roots  # noqa: E402
from scripts.run_stage2 import canonical_partition  # noqa: E402
from scripts.step4_similarity_matrix import compute_plan_weights  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.utils import acceptance as acc  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402

CONTROL = {"ratio": "r80", "split_seed": 42, "local": "L1", "family": "gt0"}
MATCHED_PREPARED = ("snapshot_h4.parquet", "snapshot_h8.parquet", "snapshot_h12.parquet", "ledgers/dev_baselines.csv")


class CompareError(RuntimeError):
    pass


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read(path: Path, str_cols=()) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", low_memory=False,
                       converters={c: str for c in str_cols})


def accept_root_record(stage: Path, name: str, candidates: list) -> dict:
    """Completion record of one root: names, status, required files, every recorded hash."""
    path = stage / "roots" / name / "completion.json"
    if not path.is_file():
        raise CompareError(f"{name}: no completion record (incomplete)")
    record = _json(path)
    if record.get("root") != name:
        raise CompareError(f"{name}: completion record belongs to {record.get('root')!r}")
    if record.get("status") != "completed":
        raise CompareError(f"{name}: status {record.get('status')!r}; the planned contrast is incomplete")
    missing = sorted(set(candidates) - set(record.get("candidates", [])))
    if missing:
        raise CompareError(f"{name}: candidates {missing} not in the completion record")
    required = [f"roots/{name}/{f}" for f in ROOT_FILES]
    required += [f"candidates/{c}/{f}" for c in candidates for f in CANDIDATE_FILES]
    problems = rid.check_inventory(stage, record.get("outputs") or {}, required)
    if problems:
        raise CompareError(f"{name}: {problems[:5]}")
    return record


def accept_tb3(run: Path) -> tuple[dict, dict, dict]:
    """(roots, candidates, identity) of the six scheduled tb3 roots of this run."""
    prepared = acc.accept_prepared(run)
    g_of, g_record = acc.accept_g_selection(run)
    sched = acc.schedule(run)
    roots = scheduled_roots(sched, g_of, plan.TIME_BLOCK)
    cands = scheduled_candidates(sched, g_of, plan.TIME_BLOCK)
    if len(roots) != 6 or len(cands) != 6:
        raise CompareError("the schedule does not list exactly six tb3 roots / candidates")
    stage = run / SPLIT_MODES[plan.TIME_BLOCK][2]
    present = {p.name for p in (stage / "roots").iterdir() if p.is_dir()} if (stage / "roots").is_dir() else set()
    if present != set(roots):
        raise CompareError(f"tb3 roots: missing {sorted(set(roots) - present)}, unexpected {sorted(present - set(roots))}")
    code, runtime = rid.code_identity(), rid.runtime_identity()
    for name, entry in roots.items():
        want = sorted(c for c, e in cands.items() if e["root"] == name)
        record = accept_root_record(stage, name, want)
        if sorted(record["candidates"]) != want:
            raise CompareError(f"{name}: candidate list differs from the schedule")
        if record.get("prepared") != prepared["outputs_sha256"] or record.get("g_selection") != g_record:
            raise CompareError(f"{name}: produced on another preparation or G selection")
        if record.get("code") != code or record.get("runtime") != runtime:
            raise CompareError(f"{name}: produced by other package code or runtime")
        root = _json(stage / "roots" / name / "root.json")
        expected = list(plan.TB3_VALIDATION_MONTHS[(entry["horizon"], entry["target_month"])])
        if (root.get("split_mode"), root.get("ratio"), root.get("horizon"), root.get("target_month"),
                root.get("g_config"), root.get("validation_months")) != \
                (plan.TIME_BLOCK, plan.TIME_BLOCK, entry["horizon"], entry["target_month"], entry["g_config"], expected):
            raise CompareError(f"{name}: root.json does not describe the scheduled tb3 root")
        snapshot = pd.read_parquet(run / "prepared" / f"snapshot_h{entry['horizon']}.parquet",
                                   columns=["area", "target_month"])
        problems = split_problems(stage, name, snapshot, entry["horizon"], entry["target_month"])
        if problems:
            raise CompareError(problems[0])
        entry["record"] = record
    return roots, cands, {"prepared": prepared["outputs_sha256"], "g_selection": g_record, "g_of": g_of,
                          "code": code, "runtime": runtime}


def rederived_tb3_roles(snapshot: pd.DataFrame, horizon: int, target: str) -> pd.DataFrame:
    """Independent rebuild of one tb3 root's split from the snapshot: rows with labels in
    [O-59, O) of the areas present at T; the latest 3 distinct label months are
    validation, all earlier rows fitting (keyed (area, target_month label, role))."""
    t = int(target[:4]) * 12 + int(target[5:7]) - 1
    origin = t - horizon
    snap = snapshot.sort_values(["area", "target_month"])
    pool = snap[(snap.target_month >= origin - plan.WINDOW) & (snap.target_month < origin)]
    pool = pool[pool.area.isin(set(snap.loc[snap.target_month == t, "area"]))]
    block = sorted(pool.target_month.unique())[-plan.TIME_BLOCK_MONTHS:]
    labels = [f"{m // 12:04d}-{m % 12 + 1:02d}" for m in pool.target_month]
    return pd.DataFrame({"area": pool.area.to_numpy(), "target_month": labels,
                         "role": np.where(pool.target_month.isin(block), "validation", "fitting")})


def split_problems(stage: Path, root: str, snapshot: pd.DataFrame, horizon: int, target: str) -> list:
    """Saved membership roles must equal the independent rebuild; validation months equal
    the frozen A2 table and every fitting month precedes every validation month."""
    members = read(stage / "roots" / root / "fold_membership.csv.gz")
    inner = members[members["role"] != "heldout_target"][["area", "target_month", "role"]].reset_index(drop=True)
    expected = rederived_tb3_roles(snapshot, horizon, target).reset_index(drop=True)
    problems = []
    if not inner.equals(expected.astype(inner.dtypes.to_dict())):
        problems.append(f"{root}: saved fitting/validation roles differ from the rebuilt time-block split")
    val = sorted(inner.loc[inner.role == "validation", "target_month"].unique())
    fit = sorted(inner.loc[inner.role == "fitting", "target_month"].unique())
    if val != list(plan.TB3_VALIDATION_MONTHS[(horizon, target)]):
        problems.append(f"{root}: membership validation months {val} differ from the frozen table")
    if not fit or fit[-1] >= val[0]:
        problems.append(f"{root}: a fitting month is not earlier than every validation month")
    return problems


def accept_control(control: Path, run: Path, g_of: dict, code_rev: str) -> tuple[dict, dict]:
    """The six D26 r80/L1/gt0 controls (read-only): (h, T) -> entry, plus identity."""
    identity = _json(control / "prepared" / "manifests" / "identity.json")
    outputs_path = control / "prepared" / "manifests" / "outputs.json"
    if rid.file_sha256(outputs_path) != identity["outputs_sha256"]:
        raise CompareError("control prepared outputs.json differs from its completion record")
    theirs, mine = _json(outputs_path), _json(run / "prepared" / "manifests" / "outputs.json")
    differ = [f for f in MATCHED_PREPARED if theirs.get(f) is None or theirs.get(f) != mine.get(f)]
    if differ:
        raise CompareError(f"control run was prepared from different data: {differ}")
    selection = control / "gscreen" / "selection.json"
    g_record = rid.file_sha256(selection)
    if _json(selection)["selected"] != {str(h): g for h, g in g_of.items()}:
        raise CompareError("control G selection differs from this run's")
    expected_code = rid.code_identity_at(code_rev)
    stage = control / "stage1"
    out = {}
    for h in plan.HORIZONS:
        for t in plan.TB3_TARGETS:
            g = g_of[str(h)]
            name = plan.root_name(h, t, g, CONTROL["ratio"], CONTROL["split_seed"])
            cand = plan.candidate_name(h, t, g, CONTROL["local"], CONTROL["ratio"], CONTROL["split_seed"],
                                       CONTROL["family"])
            record = accept_root_record(stage, name, [cand])
            if record.get("prepared") != identity["outputs_sha256"] or record.get("g_selection") != g_record:
                raise CompareError(f"control {name}: record belongs to another preparation or G selection")
            if record.get("code") != expected_code:
                raise CompareError(f"control {name}: producer code is not {code_rev}")
            root = _json(stage / "roots" / name / "root.json")
            if (root["ratio"], root["split_seed"], root["horizon"], root["target_month"], root["g_config"]) != \
                    (CONTROL["ratio"], CONTROL["split_seed"], h, t, g):
                raise CompareError(f"control {name}: root.json describes another root")
            out[(h, t)] = {"root": name, "candidate": cand, "stage": stage, "record": record}
    return out, {"run": str(control), "prepared": identity["outputs_sha256"], "g_selection": g_record,
                 "code_rev": code_rev, "code": expected_code,
                 "runtime": out[(plan.HORIZONS[0], plan.TB3_TARGETS[0])]["record"].get("runtime"),
                 "matched_prepared_outputs": {f: mine[f] for f in MATCHED_PREPARED}}


def keyed_target(stage: Path, root: str, cand: str, target: str) -> pd.DataFrame:
    """Target rows keyed by (area, T): truth, root and local codes; candidate and root
    files joined by key with identical key sets, truth and pooled codes."""
    pooled = read(stage / "roots" / root / "root_target_predictions.csv")
    local = read(stage / "candidates" / cand / "target_predictions.csv", ("branch_id",))
    for frame, what in ((pooled, "root"), (local, "candidate")):
        if frame["FEWSNET_admin_code"].duplicated().any():
            raise CompareError(f"{cand}: duplicate target keys in the {what} file")
    merged = local.merge(pooled, on="FEWSNET_admin_code", how="outer", suffixes=("", "_root"),
                         indicator=True, validate="one_to_one")
    if (merged["_merge"] != "both").any():
        raise CompareError(f"{cand}: candidate and root target keys differ")
    if not (merged["y_true_code"] == merged["y_true_code_root"]).all() or \
            not (merged["y_pred_pooled_code"] == merged["y_pred_pooled_code_root"]).all():
        raise CompareError(f"{cand}: candidate truth/root predictions differ from the root export")
    members = read(stage / "roots" / root / "fold_membership.csv.gz")
    held = members[members["role"] == "heldout_target"]
    if set(held["target_month"]) != {target}:
        raise CompareError(f"{root}: held-out rows are not the target month {target}")
    check = merged.merge(held, left_on="FEWSNET_admin_code", right_on="area", how="outer", indicator="m2",
                         validate="one_to_one")
    if (check["m2"] != "both").any() or not (check["y_true_code"] == check["class_code"]).all():
        raise CompareError(f"{root}: target predictions differ from the held-out membership")
    return pd.DataFrame({"area": merged["FEWSNET_admin_code"].astype(np.int64), "target_month": target,
                         "y_true": merged["y_true_code"].astype(np.int64),
                         "y_root": merged["y_pred_pooled_code"].astype(np.int64),
                         "y_local": merged["y_pred_partitioned_code"].astype(np.int64)})


def keyed_validation(stage: Path, root: str, cand: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(validation predictions keyed by (area, month), membership) with exact key equality
    to the membership validation rows and no fitting-row overlap."""
    members = read(stage / "roots" / root / "fold_membership.csv.gz")
    val = read(stage / "candidates" / cand / "validation_predictions.csv.gz", ("branch_id",))
    key = ["area", "target_month"]
    if val.duplicated(key).any():
        raise CompareError(f"{cand}: duplicate validation keys")
    vm = members[members["role"] == "validation"]
    merged = val.merge(vm, on=key, how="outer", indicator=True, validate="one_to_one")
    if (merged["_merge"] != "both").any() or not (merged["y_true"] == merged["class_code"]).all():
        raise CompareError(f"{cand}: validation predictions differ from the membership validation rows")
    fit = members[members["role"] == "fitting"]
    if len(fit.merge(vm, on=key)):
        raise CompareError(f"{root}: a key is both fitting and validation")
    return merged.drop(columns=["_merge", "role", "class_code"]), members


def support(rows: pd.DataFrame, label: str) -> dict:
    y = rows["class_code"] if "class_code" in rows else rows["y_true"]
    return {f"{label}_rows": int(len(rows)), f"{label}_areas": int(rows["area"].nunique()),
            f"{label}_dates": int(rows["target_month"].nunique()),
            f"{label}_crisis_positives": int((np.asarray(y) >= fourclass.CRISIS_MIN_CODE).sum())}


def crisis_matrix(m: np.ndarray) -> list:
    m = np.asarray(m)
    return [[int(m[:2, :2].sum()), int(m[:2, 2:].sum())], [int(m[2:, :2].sum()), int(m[2:, 2:].sum())]]


def e4_weight(f_part: float, f_root: float) -> float:
    return float(compute_plan_weights(pd.DataFrame({"macro_f1": [f_part], "macro_f1_base": [f_root]}))["weight"].iloc[0])


def method_row(stage: Path, root: str, cand: str, h: int, t: str, method: str, root_booster: str):
    target = keyed_target(stage, root, cand, t)
    val, members = keyed_validation(stage, root, cand)
    c = _json(stage / "candidates" / cand / "candidate.json")
    yt, yr, yl = target["y_true"].to_numpy(), target["y_root"].to_numpy(), target["y_local"].to_numpy()
    vt, vr, vf = val["y_true"].to_numpy(), val["y_root"].to_numpy(), val["y_final"].to_numpy()
    e3_root, e3_local = fourclass.crisis_f1_exact(yt, yr), fourclass.crisis_f1_exact(yt, yl)
    e2_root, e2_final = fourclass.crisis_f1_exact(vt, vr), fourclass.crisis_f1_exact(vt, vf)
    if float(e3_local) != c["scores"]["score"] or float(e3_root) != c["scores"]["score_base"]:
        raise CompareError(f"{cand}: recorded E3 scores do not recompute from keyed predictions")
    fitting, validation = members[members["role"] == "fitting"], members[members["role"] == "validation"]
    last_save = {e["saved_as"]: e for e in c["fits"]["saved_log"]}
    terminals = c["partition"]["terminal_partitions"]
    boosters = {(last_save.get(b) or {}).get("booster_sha256") or (root_booster if b == "root" else f"unresolved:{b}")
                for b in terminals}
    row = {"method": method, "horizon": h, "target_month": t, "root": root, "candidate": cand,
           "n_target": int(len(target)), "n_validation": int(len(val)),
           "e3_root_crisis_f1": float(e3_root), "e3_local_crisis_f1": float(e3_local),
           "e3_local_minus_root": float(e3_local - e3_root), "e3_local_minus_root_exact": str(e3_local - e3_root),
           "e3_root_crisis_f1_exact": str(e3_root), "e3_local_crisis_f1_exact": str(e3_local),
           "e3_root_fourclass": fourclass.macro_f1(yt, yr), "e3_local_fourclass": fourclass.macro_f1(yt, yl),
           "e2_root_crisis_f1": float(e2_root), "e2_final_crisis_f1": float(e2_final),
           "e2_final_minus_root": float(e2_final - e2_root),
           "e2_root_fourclass": fourclass.macro_f1(vt, vr), "e2_final_fourclass": fourclass.macro_f1(vt, vf),
           "e4_weight": e4_weight(float(e3_local), float(e3_root)),
           "n_terminal": int(c["partition"]["n_terminal"]),
           "accepted_splits": int(c["partition"]["accepted_splits"]),
           "child_fits_including_rejected": int(c["fits"]["child_fits"]),
           "distinct_terminal_boosters": len(boosters),
           "unresolved_boosters": sum(str(b).startswith("unresolved:") for b in boosters),
           "validation_months": ",".join(sorted(validation["target_month"].unique())),
           "fitting_months_first_last": f"{fitting['target_month'].min()}..{fitting['target_month'].max()}",
           "validation_only_areas": int(len(set(validation["area"]) - set(fitting["area"]))),
           **support(fitting, "fitting"), **support(validation, "validation"),
           **support(target.assign(class_code=target["y_true"]), "target")}
    conf = {"target_root": fourclass.confusion(yt, yr), "target_local": fourclass.confusion(yt, yl),
            "validation_root": fourclass.confusion(vt, vr), "validation_final": fourclass.confusion(vt, vf)}
    return row, target, conf, canonical_partition(stage / "candidates" / cand / "correspondence_table.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--control-code-rev", default="87513eb", help="committed producer of the D26 controls")
    args = parser.parse_args()
    run, control = args.run_dir.resolve(), args.control_run.resolve()
    out = run / "stage1_tb3_compare"
    rid.refuse_existing(out, "the tb3 comparison")
    roots, cands, ident = accept_tb3(run)
    controls, control_ident = accept_control(control, run, ident["g_of"], args.control_code_rev)
    tb3_stage = run / SPLIT_MODES[plan.TIME_BLOCK][2]
    rows, pairs, confusions, partitions = [], [], [], {}
    for name, entry in sorted(roots.items(), key=lambda kv: (kv[1]["horizon"], kv[1]["target_month"])):
        h, t = entry["horizon"], entry["target_month"]
        cand = next(c for c, e in cands.items() if e["root"] == name)
        ctl = controls[(h, t)]
        root_booster = {"tb3": _json(tb3_stage / "roots" / name / "root.json")["root_booster_sha256"],
                        "r80": _json(ctl["stage"] / "roots" / ctl["root"] / "root.json")["root_booster_sha256"]}
        new, t_new, c_new, p_new = method_row(tb3_stage, name, cand, h, t, plan.TIME_BLOCK, root_booster["tb3"])
        old, t_old, c_old, p_old = method_row(ctl["stage"], ctl["root"], ctl["candidate"], h, t, "r80_control",
                                              root_booster["r80"])
        both = t_new.merge(t_old, on=["area", "target_month"], how="outer", suffixes=("_tb3", "_r80"),
                           indicator=True, validate="one_to_one")
        if (both["_merge"] != "both").any() or not (both["y_true_tb3"] == both["y_true_r80"]).all():
            raise CompareError(f"h{h} {t}: target keys or truth differ between tb3 and the r80 control")
        rows += [new, old]
        partitions[new["candidate"]], partitions[old["candidate"]] = p_new, p_old
        for method, conf in ((plan.TIME_BLOCK, c_new), ("r80_control", c_old)):
            for which, m in conf.items():
                confusions.append({"method": method, "horizon": h, "target_month": t, "which": which,
                                   "fourclass": np.asarray(m, dtype=int).tolist(), "crisis_[[nn,nc],[cn,cc]]": crisis_matrix(m),
                                   "per_class_f1": fourclass.per_class_f1(m).round(6).tolist()})
        e3_gain_new = Fraction(new["e3_local_minus_root_exact"])
        e3_gain_old = Fraction(old["e3_local_minus_root_exact"])
        pairs.append({
            "horizon": h, "target_month": t, "tb3_candidate": new["candidate"], "r80_candidate": old["candidate"],
            "target_keys_identical": True, "n_target": new["n_target"],
            "root_change_tb3_minus_r80": float(Fraction(new["e3_root_crisis_f1_exact"]) - Fraction(old["e3_root_crisis_f1_exact"])),
            "local_change_tb3_minus_r80": float(Fraction(new["e3_local_crisis_f1_exact"]) - Fraction(old["e3_local_crisis_f1_exact"])),
            "e3_gain_tb3": float(e3_gain_new), "e3_gain_r80": float(e3_gain_old),
            "e3_gain_difference_tb3_minus_r80": float(e3_gain_new - e3_gain_old),
            "e2_gain_tb3_own_rows": new["e2_final_minus_root"], "e2_gain_r80_own_rows": old["e2_final_minus_root"],
            "n_terminal_tb3": new["n_terminal"], "n_terminal_r80": old["n_terminal"],
            "same_coverage": p_new[0] == p_old[0], "identical_partition": p_new == p_old,
            "e4_weight_tb3": new["e4_weight"], "e4_weight_r80": old["e4_weight"]})
    frame, pair_frame = pd.DataFrame(rows), pd.DataFrame(pairs)

    def duplicates(names):
        by_cov, dup = {}, 0
        for n in names:
            cov, labels = partitions[n]
            seen = by_cov.setdefault(cov, set())
            dup += labels in seen
            seen.add(labels)
        return {"candidates": len(names), "coverages": len(by_cov), "same_coverage_duplicates": dup}

    def method_summary(g):
        split = g[g["n_terminal"] > 1]
        return {"candidates": int(len(g)), "split": int(len(split)),
                "mean_e3_root_crisis_f1": float(g["e3_root_crisis_f1"].mean()),
                "mean_e3_local_minus_root": float(g["e3_local_minus_root"].mean()),
                "e3_positive": int((g["e3_local_minus_root"] > 0).sum()),
                "e3_negative": int((g["e3_local_minus_root"] < 0).sum()),
                "mean_e2_final_minus_root": float(g["e2_final_minus_root"].mean()),
                "positive_e4_weights": int((g["e4_weight"] > 0).sum()),
                "partitions": duplicates(list(g["candidate"]))}

    summary = {
        "contrast": "D27 (experiment-plan A2): tb3 time block vs D26 r80 random split; L1, gt0, seed 42, same H/T/G",
        "endpoint": "crisis_f1 (four-class argmax collapsed to IPC>=3); fixed-four macro F1 secondary",
        "this_run": {k: v for k, v in ident.items() if k != "g_of"}, "g_selected": ident["g_of"],
        "control": control_ident,
        "by_method": {m: method_summary(g) for m, g in frame.groupby("method")},
        "by_horizon": {str(h): {m: method_summary(gg) for m, gg in g.groupby("method")} for h, g in frame.groupby("horizon")},
        "by_target": {t: {m: method_summary(gg) for m, gg in g.groupby("method")} for t, g in frame.groupby("target_month")},
        "pairs": {"mean_root_change_tb3_minus_r80": float(pair_frame["root_change_tb3_minus_r80"].mean()),
                  "mean_e3_gain_difference_tb3_minus_r80": float(pair_frame["e3_gain_difference_tb3_minus_r80"].mean()),
                  "identical_partitions": int(pair_frame["identical_partition"].sum())},
        "confusions": confusions,
        "caveats": ["six related candidates (two targets x three H from shared data), not six independent experiments",
                    "E2 populations differ between methods; their E2 difference is not a paired effect",
                    "the tb3 block is reused for E1 scan and E2 acceptance; not an independent confirmation set or an "
                    "internal rolling-origin replay",
                    "root changes come from different fitting rows/periods; local-root gains are reported separately",
                    "G was selected on the whole development period (D24/D26); development evidence only"],
    }
    out.mkdir(parents=True)
    frame.to_csv(out / "candidates.csv", index=False, float_format="%.17g")
    pair_frame.to_csv(out / "pairs.csv", index=False, float_format="%.17g")
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "completion.json", {
        "stage": "stage1_tb3_compare", "code": rid.code_identity(), "runtime": rid.runtime_identity(),
        "outputs": {rel: sha for rel, sha in rid.output_hashes(out).items() if rel != "completion.json"}})
    print(json.dumps({k: summary[k] for k in ("by_method", "pairs")}, indent=1))


if __name__ == "__main__":
    main()
