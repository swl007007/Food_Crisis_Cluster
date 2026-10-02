"""D29 / A4 keyed diagnostic: six rootconf candidates vs the six D28 rootinc candidates (never fits).

python scripts/stage1_rootconf_compare.py --run-dir RUN --control-run D28_RUN [--control-code-rev 98adf48]
    [--code-rev PRODUCER_REV]
python scripts/stage1_rootconf_compare.py --mode recentsearch --run-dir RUN --control-run D29_RUN
    [--control-code-rev ab1ac83] [--code-rev PRODUCER_REV]      (D30 / A5; writes RUN/stage1_recentsearch_compare/)

Reuses the D27/D28 acceptance and keyed joins (scripts/stage1_tb3_compare.py,
scripts/stage1_rootinc_compare.py). REQUIRES, per (H, T): identical fitting keys, the new
S ∪ C == the D28 validation keys (S ∩ C empty), identical target keys/truth, identical
root target predictions and root booster SHA-256, and S/C roles equal to
confirmation_split re-derived on the D28 validation keys. Reports S / C / E3 root and local
crisis F1 (four-class secondary), confusions, S and C support (also per frozen terminal branch:
terminal_support.csv) and C routing; D28's saved
validation predictions are rescored on the same S and C keys and labelled EXPOSED (the
old search used all of them). Writes RUN/stage1_rootconf_compare/.
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
from scripts.stage1_rootinc_compare import crisis_counts, round_columns  # noqa: E402
from scripts.run_stage1 import NAMERS, SPLIT_MODES, scheduled_candidates, scheduled_roots  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.utils import acceptance as acc  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402
from src.utils.split import confirmation_split  # noqa: E402

CompareError = tc.CompareError
KEY = ["area", "target_month"]
#: modes with a frozen-candidate confirmation set C
CONF_MODES = (plan.ROOTCONF, plan.RECENTSEARCH)
#: new mode -> (control mode, default control producer, column prefix of the control)
CONTROLS = {plan.ROOTCONF: (plan.ROOTINC, "98adf48", "d28"), plan.RECENTSEARCH: (plan.ROOTCONF, "ab1ac83", "d29")}


def accept_mode(run: Path, mode: str, code=None, code_rev=None):
    """(roots, candidates, identity) of the six scheduled roots of ``mode`` in ``run``."""
    if code_rev is None:   # this run: full current-code acceptance
        prepared = acc.accept_prepared(run)
        g_of, g_record = acc.accept_g_selection(run)
    else:   # committed producer (tc.accept_control convention): record hashes and producer code
        prepared = tc._json(run / "prepared" / "manifests" / "identity.json")
        if rid.file_sha256(run / "prepared" / "manifests" / "outputs.json") != prepared["outputs_sha256"]:
            raise CompareError(f"{run.name}: prepared outputs.json differs from its completion record")
        recorded = tc._json(run / "prepared" / "manifests" / "outputs.json")
        problems = rid.verify_outputs(run / "prepared", recorded) + \
            rid.check_inventory(run / "prepared", recorded, rid.REQUIRED_PREPARED)
        if problems or prepared.get("stage") != "prepare":
            raise CompareError(f"{run.name}: prepared outputs do not match their record: {problems[:5]}")
        if prepared.get("code") != code or prepared.get("runtime") != rid.runtime_identity():
            raise CompareError(f"{run.name}: preparation was not produced by {code_rev} on this runtime")
        selection = run / "gscreen" / "selection.json"
        g_record, g_rec = rid.file_sha256(selection), tc._json(selection)
        if g_rec.get("prepared") != prepared["outputs_sha256"]:
            raise CompareError(f"{run.name}: G selection was made on another preparation")
        g_of = g_rec["selected"]
    if dict(g_of) != plan.TB3_G:
        raise CompareError(f"{mode} requires the locked G {plan.TB3_G}")
    sched = acc.schedule(run)
    roots, cands = scheduled_roots(sched, g_of, mode), scheduled_candidates(sched, g_of, mode)
    if len(roots) != 6 or len(cands) != 6:
        raise CompareError(f"the schedule does not give exactly six {mode} roots / candidates")
    stage = run / SPLIT_MODES[mode][2]
    present = {p.name for p in (stage / "roots").iterdir() if p.is_dir()} if (stage / "roots").is_dir() else set()
    if present != set(roots):
        raise CompareError(f"{mode} roots: missing {sorted(set(roots) - present)}, unexpected {sorted(present - set(roots))}")
    cand_dirs = {p.name for p in (stage / "candidates").iterdir() if p.is_dir()} if (stage / "candidates").is_dir() else set()
    if cand_dirs != set(cands):
        raise CompareError(f"{mode} candidates: dirs {sorted(cand_dirs)} != scheduled {sorted(cands)}")
    name_of = NAMERS[mode][1]
    code = code or rid.code_identity()
    runtime = None
    for name, entry in roots.items():
        want = sorted(c for c, e in cands.items() if e["root"] == name)
        if want != [name_of(entry["horizon"], entry["target_month"], entry["g_config"])]:
            raise CompareError(f"{name}: expected exactly its single L1/gt0 {mode} candidate, got {want}")
        record = tc.accept_root_record(stage, name, want)
        if list(record.get("candidates", [])) != want:
            raise CompareError(f"{name}: completion lists {record.get('candidates')}, not exactly {want}")
        if record.get("prepared") != prepared["outputs_sha256"] or record.get("g_selection") != g_record:
            raise CompareError(f"{name}: produced on another preparation or G selection")
        if record.get("code") != code:
            raise CompareError(f"{name}: produced by other package code" + (f" than {code_rev}" if code_rev else ""))
        runtime = record.get("runtime")
        if runtime != rid.runtime_identity():
            raise CompareError(f"{name}: produced by another runtime")
        root = tc._json(stage / "roots" / name / "root.json")
        if (root.get("increment_source"), root.get("ratio"), root.get("split_seed"), root.get("horizon"),
                root.get("target_month"), root.get("g_config")) != \
                ("root", plan.ROOTINC_RATIO, plan.ROOTINC_SEED, entry["horizon"], entry["target_month"], entry["g_config"]):
            raise CompareError(f"{name}: root.json does not describe the scheduled r80/seed42 shared-root root")
        if tc._json(stage / "candidates" / want[0] / "candidate.json").get("increment_source") != "root":
            raise CompareError(f"{want[0]}: candidate is not a shared-root increment candidate")
        if (mode in CONF_MODES) != ("confirmation_split" in root) or \
                (mode == plan.RECENTSEARCH) != ("recent_search" in root):
            raise CompareError(f"{name}: confirmation split / recent search presence does not match mode {mode}")
        if mode in CONF_MODES and not (stage / "candidates" / want[0] / "confirmation_predictions.csv.gz").is_file():
            raise CompareError(f"{want[0]}: no confirmation predictions")
    return roots, cands, {"run": str(run), "stage": stage, "prepared": prepared["outputs_sha256"],
                          "g_selection": g_record, "g_of": g_of, "code": code, "code_rev": code_rev,
                          "runtime": runtime}


def same_roots(new_stage, new_root, old_stage, old_root, t_new, t_old) -> pd.DataFrame:
    """Return the new membership after checking it against the D28 root it must reproduce."""
    a = tc.read(new_stage / "roots" / new_root / "fold_membership.csv.gz")
    b = tc.read(old_stage / "roots" / old_root / "fold_membership.csv.gz")
    if a.duplicated(KEY).any():
        raise CompareError(f"{new_root}: duplicate membership keys")
    a_cmp = a.assign(role=a["role"].replace({"confirmation": "validation"}))   # S ∪ C == original validation
    m = a_cmp.merge(b, on=KEY, how="outer", suffixes=("_new", "_old"), indicator=True, validate="one_to_one")
    if (m["_merge"] != "both").any() or not (m["role_new"] == m["role_old"]).all() \
            or not (m["class_code_new"] == m["class_code_old"]).all():
        raise CompareError(f"{new_root}: fitting / S∪C / target membership differs from the D28 control")
    # Re-derive S/C label-blind from the D28 validation keys (A4 section 2); order is numeric area/month.
    ov = b[b["role"] == "validation"].reset_index(drop=True)
    want_c = confirmation_split(ov["area"].to_numpy(int), pd.PeriodIndex(ov["target_month"], freq="M").asi8,
                                plan.CONFIRMATION_SEED) == 1
    got = ov[KEY].merge(a[KEY + ["role"]], on=KEY, how="left", validate="one_to_one")
    if not ((got["role"] == "confirmation").to_numpy() == want_c).all():
        raise CompareError(f"{new_root}: S/C roles differ from confirmation_split re-derived on the D28 validation keys")
    same_target_and_root(new_stage, new_root, old_stage, old_root, t_new, t_old)
    return a


def same_target_and_root(new_stage, new_root, old_stage, old_root, t_new, t_old) -> None:
    j = t_new.merge(t_old, on=KEY, how="outer", suffixes=("_new", "_old"), indicator=True, validate="one_to_one")
    if (j["_merge"] != "both").any() or not (j["y_true_new"] == j["y_true_old"]).all() \
            or not (j["y_root_new"] == j["y_root_old"]).all():
        raise CompareError(f"{new_root}: target keys/truth/root predictions differ from the control")
    ra, rb = (tc._json(s / "roots" / r / "root.json")["root_booster_sha256"]
              for s, r in ((new_stage, new_root), (old_stage, old_root)))
    if ra != rb:
        raise CompareError(f"{new_root}: root booster differs from the control ({ra} vs {rb})")


def same_roots_recent(new_stage, new_root, old_stage, old_root, t_new, t_old, h, t) -> pd.DataFrame:
    """D30/A5: the new membership equals the D29 one except S -> S_recent + unused_search_history,
    with the recent months re-derived from the D29 original validation (S ∪ C) dates."""
    a = tc.read(new_stage / "roots" / new_root / "fold_membership.csv.gz")
    b = tc.read(old_stage / "roots" / old_root / "fold_membership.csv.gz")
    if a.duplicated(KEY).any():
        raise CompareError(f"{new_root}: duplicate membership keys")
    months = sorted(b.loc[b["role"].isin(["validation", "confirmation"]), "target_month"].unique())[-plan.RECENT_SEARCH_MONTHS:]
    if tuple(months) != tuple(plan.RECENT_SEARCH_DATES[(h, t)]):
        raise CompareError(f"{new_root}: D29 original validation gives recent months {months}, not the A5 table")
    want = b["role"].where(~((b["role"] == "validation") & ~b["target_month"].isin(months)), "unused_search_history")
    m = a.merge(b.assign(role=want), on=KEY, how="outer", suffixes=("_new", "_old"), indicator=True, validate="one_to_one")
    if (m["_merge"] != "both").any() or not (m["role_new"] == m["role_old"]).all() \
            or not (m["class_code_new"] == m["class_code_old"]).all():
        raise CompareError(f"{new_root}: fitting / C / S_recent+unused / target membership differs from the D29 control")
    same_target_and_root(new_stage, new_root, old_stage, old_root, t_new, t_old)
    return a


def keyed_confirmation(stage, cand, members) -> pd.DataFrame:
    c = tc.read(stage / "candidates" / cand / "confirmation_predictions.csv.gz", ("branch_id",))
    if c.duplicated(KEY).any():
        raise CompareError(f"{cand}: duplicate confirmation keys")
    cm = members[members["role"] == "confirmation"]
    merged = c.merge(cm, on=KEY, how="outer", indicator=True, validate="one_to_one")
    if (merged["_merge"] != "both").any() or not (merged["y_true"] == merged["class_code"]).all():
        raise CompareError(f"{cand}: confirmation predictions do not cover exactly the C membership keys")
    return merged.drop(columns=["_merge", "role", "class_code"])


def terminal_support(frame, which) -> list:
    """S or C support per frozen terminal branch: rows/areas/dates/class counts/crisis positives."""
    out = []
    for branch, g in frame.groupby("branch_id", sort=True):
        y = g["y_true"].to_numpy()
        out.append({"set": which, "branch_id": branch, "rows": int(len(g)), "areas": int(g["area"].nunique()),
                    "dates": int(g["target_month"].nunique()),
                    **{f"class_{k}": int((y == k).sum()) for k in range(plan.N_CLASSES)},
                    "crisis_positives": int((y >= fourclass.CRISIS_MIN_CODE).sum())})
    return out


def scored(frame, pred, label) -> tuple[dict, dict]:
    y, p = frame["y_true"].to_numpy(), frame[pred].to_numpy()
    f = fourclass.crisis_f1_exact(y, p)
    conf = fourclass.confusion(y, p)
    return {f"{label}_crisis_f1": float(f), f"{label}_crisis_f1_exact": str(f),
            f"{label}_fourclass": fourclass.macro_f1(y, p), **crisis_counts(conf, label)}, conf


def pool_block(frame, prefix, root_col="y_root", local_col="y_final") -> tuple[dict, dict]:
    r, cr = scored(frame, root_col, f"{prefix}_root")
    loc, cl = scored(frame, local_col, f"{prefix}_local")
    gain = Fraction(loc[f"{prefix}_local_crisis_f1_exact"]) - Fraction(r[f"{prefix}_root_crisis_f1_exact"])
    return {**r, **loc, f"{prefix}_local_minus_root": float(gain), f"{prefix}_local_minus_root_exact": str(gain),
            f"{prefix}_fourclass_local_minus_root": loc[f"{prefix}_local_fourclass"] - r[f"{prefix}_root_fourclass"]}, \
        {f"{prefix}_root": cr, f"{prefix}_local": cl}


def recent_row(stage, name, cand, cstage, croot, ccand, h, t):
    """D30/A5 row: new recentsearch candidate vs the D29 rootconf candidate on identical keys."""
    rb = tc._json(stage / "roots" / name / "root.json")["root_booster_sha256"]
    new, t_new, c_new, p_new = tc.method_row(stage, name, cand, h, t, plan.RECENTSEARCH, rb)
    old, t_old, conf_old, p_old = tc.method_row(cstage, croot, ccand, h, t, plan.ROOTCONF, rb)
    members = same_roots_recent(stage, name, cstage, croot, t_new, t_old, h, t)
    cmembers = tc.read(cstage / "roots" / croot / "fold_membership.csv.gz")
    s_frame, _ = tc.keyed_validation(stage, name, cand)                   # S_recent
    old_val, _ = tc.keyed_validation(cstage, croot, ccand)                 # D29 S (its search used it all)
    old_s = old_val.merge(s_frame[KEY], on=KEY, validate="one_to_one")
    if len(old_s) != len(s_frame) or not (old_s.sort_values(KEY)["y_root"].to_numpy()
                                          == s_frame.sort_values(KEY)["y_root"].to_numpy()).all():
        raise CompareError(f"{cand}: S_recent keys/root predictions differ from D29 on the same keys")
    c_frame, c_old = keyed_confirmation(stage, cand, members), keyed_confirmation(cstage, ccand, cmembers)
    chk = c_frame.merge(c_old, on=KEY, how="outer", suffixes=("", "_d29"), indicator=True, validate="one_to_one")
    if (chk["_merge"] != "both").any() or not (chk["y_root"] == chk["y_root_d29"]).all() \
            or not (chk["y_true"] == chk["y_true_d29"]).all():
        raise CompareError(f"{cand}: C keys/truth/root predictions differ from D29")
    recent = list(plan.RECENT_SEARCH_DATES[(h, t)])
    blocks, mats = {}, {}
    for prefix, frame in (("S_recent", s_frame), ("d29_exposed_S_recent", old_s),
                          ("C", c_frame), ("C_recent6", c_frame[c_frame["target_month"].isin(recent)]),
                          ("C_older", c_frame[~c_frame["target_month"].isin(recent)]),
                          ("d29_C", c_old), ("d29_C_recent6", c_old[c_old["target_month"].isin(recent)]),
                          ("d29_C_older", c_old[~c_old["target_month"].isin(recent)])):
        b, mt = pool_block(frame, prefix)
        blocks.update(b); mats.update(mt)
    mats.update({k: v for k, v in c_new.items() if k.startswith("target_")})
    mats["d29_target_local"] = conf_old["target_local"]   # same root, so only D29's local E3 confusion is new
    role = lambda r: members[members["role"] == r]   # noqa: E731
    row = {"horizon": h, "target_month": t, "candidate": cand, "d29_candidate": ccand,
           "recent_months": ",".join(recent),
           "e3_root_crisis_f1": new["e3_root_crisis_f1"], "e3_local_crisis_f1": new["e3_local_crisis_f1"],
           "e3_local_minus_root": new["e3_local_minus_root"],
           "e3_local_minus_root_exact": new["e3_local_minus_root_exact"],
           "e3_fourclass_local_minus_root": new["e3_local_fourclass"] - new["e3_root_fourclass"],
           **crisis_counts(c_new["target_root"], "e3_root"), **crisis_counts(c_new["target_local"], "e3_local"),
           "d29_e3_local_minus_root": old["e3_local_minus_root"],
           "e3_minus_d29_e3": float(Fraction(new["e3_local_minus_root_exact"]) - Fraction(old["e3_local_minus_root_exact"])),
           **blocks,
           **{f"{k}_minus_d29": blocks[f"{k}_local_minus_root"] - blocks[f"d29_{k}_local_minus_root"]
              for k in ("C", "C_recent6", "C_older")},
           "S_recent_minus_d29_exposed": blocks["S_recent_local_minus_root"] - blocks["d29_exposed_S_recent_local_minus_root"],
           **tc.support(role("validation"), "S_recent"), **tc.support(role("unused_search_history"), "unused"),
           **tc.support(role("confirmation"), "C"),
           "C_rows_root_unassigned": int((c_frame["routing"] != "terminal_branch").sum()),
           "C_rows_on_root_branch": int((c_frame["branch_id"] == "root").sum()),
           "n_terminal": new["n_terminal"], "accepted_splits": new["accepted_splits"],
           "distinct_terminal_boosters": new["distinct_terminal_boosters"],
           **round_columns(stage, cand), "e4_weight": new["e4_weight"],
           "d29_n_terminal": old["n_terminal"], "d29_distinct_terminal_boosters": old["distinct_terminal_boosters"],
           "d29_e4_weight": old["e4_weight"], "identical_partition_to_d29": p_new == p_old}
    terminals = [{"horizon": h, "target_month": t, "candidate": cand, **r}
                 for which, fr in (("S_recent", s_frame), ("C", c_frame),
                                   ("C_recent6", c_frame[c_frame["target_month"].isin(recent)]),
                                   ("C_older", c_frame[~c_frame["target_month"].isin(recent)]))
                 for r in terminal_support(fr, which)]
    return row, mats, terminals, p_new, p_old


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(CONTROLS), default=plan.ROOTCONF,
                        help="rootconf (D29 vs D28 rootinc) or recentsearch (D30 vs D29 rootconf)")
    parser.add_argument("--control-run", type=Path, required=True, help="the D28 rootinc / D29 rootconf run")
    parser.add_argument("--control-code-rev", default=None,
                        help="committed producer of the controls (default: 98adf48 for rootconf, ab1ac83 for recentsearch)")
    parser.add_argument("--code-rev", default=None,
                        help="committed producer of this run when it differs from the current code (default: current)")
    args = parser.parse_args()
    mode = args.mode
    cmode, default_rev, cpre = CONTROLS[mode]
    control_rev = args.control_code_rev or default_rev
    run, control = args.run_dir.resolve(), args.control_run.resolve()
    out = run / f"stage1_{mode}_compare"
    rid.refuse_existing(out, f"the {mode} comparison")
    roots, cands, ident = accept_mode(run, mode, rid.code_identity_at(args.code_rev) if args.code_rev else None,
                                      args.code_rev)
    croots, ccands, cident = accept_mode(control, cmode, rid.code_identity_at(control_rev), control_rev)
    if cident["g_of"] != ident["g_of"]:
        raise CompareError("control G selection differs from this run's")
    theirs = tc._json(control / "prepared" / "manifests" / "outputs.json")
    mine = tc._json(run / "prepared" / "manifests" / "outputs.json")
    differ = [f for f in tc.MATCHED_PREPARED if theirs.get(f) is None or theirs.get(f) != mine.get(f)]
    if differ:
        raise CompareError(f"control run was prepared from different data: {differ}")
    stage, cstage = ident["stage"], cident["stage"]
    ctl_of = {(e["horizon"], e["target_month"]): n for n, e in croots.items()}
    rows, confusions, partitions, terminals = [], [], {}, []
    for name, entry in sorted(roots.items(), key=lambda kv: (kv[1]["horizon"], kv[1]["target_month"])):
        h, t = entry["horizon"], entry["target_month"]
        cand = next(c for c, e in cands.items() if e["root"] == name)
        croot = ctl_of[(h, t)]
        ccand = next(c for c, e in ccands.items() if e["root"] == croot)
        if mode == plan.RECENTSEARCH:
            row, mats, terms, p_new, p_old = recent_row(stage, name, cand, cstage, croot, ccand, h, t)
            rows.append(row)
            terminals += terms
            partitions[cand], partitions[ccand] = p_new, p_old
            for which, mtx in mats.items():
                confusions.append({"horizon": h, "target_month": t, "candidate": cand, "which": which,
                                   "exposed": which.startswith("d29_exposed"),
                                   "fourclass": [[int(x) for x in r] for r in mtx],
                                   "crisis_[[nn,nc],[cn,cc]]": tc.crisis_matrix(mtx)})
            continue
        rb = tc._json(stage / "roots" / name / "root.json")["root_booster_sha256"]
        new, t_new, c_new, p_new = tc.method_row(stage, name, cand, h, t, plan.ROOTCONF, rb)
        old, t_old, c_old, p_old = tc.method_row(cstage, croot, ccand, h, t, plan.ROOTINC, rb)
        members = same_roots(stage, name, cstage, croot, t_new, t_old)
        s_keys = members[members["role"] == "validation"][KEY]
        c_frame = keyed_confirmation(stage, cand, members)
        s_frame, _ = tc.keyed_validation(stage, name, cand)
        old_val, _ = tc.keyed_validation(cstage, croot, ccand)
        # Same root booster -> identical root codes on every C key (consistency of the join).
        chk = c_frame.merge(old_val, on=KEY, suffixes=("", "_d28"), validate="one_to_one")
        if len(chk) != len(c_frame) or not (chk["y_root"] == chk["y_root_d28"]).all():
            raise CompareError(f"{cand}: C root predictions differ from the D28 root on the same keys")
        old_s = old_val.merge(s_keys, on=KEY, validate="one_to_one")
        old_c = old_val.merge(c_frame[KEY], on=KEY, validate="one_to_one")
        blocks, mats = {}, {}
        for prefix, frame in (("S", s_frame), ("C", c_frame), ("d28_exposed_S", old_s), ("d28_exposed_C", old_c)):
            b, mt = pool_block(frame, prefix)
            blocks.update(b); mats.update(mt)
        mats.update({k: v for k, v in c_new.items() if k.startswith("target_")})
        cmem = members[members["role"] == "confirmation"]
        smem = members[members["role"] == "validation"]
        row = {"horizon": h, "target_month": t, "candidate": cand, "d28_candidate": ccand,
               "e3_root_crisis_f1": new["e3_root_crisis_f1"], "e3_local_crisis_f1": new["e3_local_crisis_f1"],
               "e3_local_minus_root": new["e3_local_minus_root"],
               "e3_local_minus_root_exact": new["e3_local_minus_root_exact"],
               "e3_fourclass_local_minus_root": new["e3_local_fourclass"] - new["e3_root_fourclass"],
               **crisis_counts(c_new["target_root"], "e3_root"), **crisis_counts(c_new["target_local"], "e3_local"),
               **blocks,
               "delta_S_minus_delta_C": blocks["S_local_minus_root"] - blocks["C_local_minus_root"],
               "delta_C_minus_delta_E3": blocks["C_local_minus_root"] - new["e3_local_minus_root"],
               **tc.support(smem, "S"), **tc.support(cmem, "C"),
               "C_rows_terminal_branch": int((c_frame["routing"] == "terminal_branch").sum()),
               "C_rows_root_unassigned": int((c_frame["routing"] != "terminal_branch").sum()),
               "C_rows_on_root_branch": int((c_frame["branch_id"] == "root").sum()),
               "n_terminal": new["n_terminal"], "accepted_splits": new["accepted_splits"],
               "distinct_terminal_boosters": new["distinct_terminal_boosters"],
               **round_columns(stage, cand), "e4_weight": new["e4_weight"],
               "d28_n_terminal": old["n_terminal"], "d28_e3_local_minus_root": old["e3_local_minus_root"],
               "d28_e4_weight": old["e4_weight"], "identical_partition_to_d28": p_new == p_old}
        rows.append(row)
        for which, fr in (("S", s_frame), ("C", c_frame)):
            terminals += [{"horizon": h, "target_month": t, "candidate": cand, **r} for r in terminal_support(fr, which)]
        partitions[cand], partitions[ccand] = p_new, p_old
        for which, mtx in mats.items():
            confusions.append({"horizon": h, "target_month": t, "candidate": cand, "which": which,
                               "exposed": which.startswith("d28_exposed"),
                               "fourclass": [[int(x) for x in r] for r in mtx],
                               "crisis_[[nn,nc],[cn,cc]]": tc.crisis_matrix(mtx)})
    frame = pd.DataFrame(rows)

    def dup(names):
        by_cov, n = {}, 0
        for x in names:
            cov, labels = partitions[x]
            seen = by_cov.setdefault(cov, set())
            n += labels in seen
            seen.add(labels)
        return {"candidates": len(names), "coverages": len(by_cov), "same_coverage_duplicates": n}

    if mode == plan.RECENTSEARCH:
        mean_keys = ("S_recent_local_minus_root", "d29_exposed_S_recent_local_minus_root", "C_local_minus_root",
                     "C_recent6_local_minus_root", "C_older_local_minus_root", "d29_C_local_minus_root",
                     "d29_C_recent6_local_minus_root", "d29_C_older_local_minus_root",
                     "e3_local_minus_root", "d29_e3_local_minus_root", "e3_minus_d29_e3")
        summary = {
            "contrast": "D30 (A5): search on S_recent (latest six observed original-validation months) only, same "
                        "fitting/root/C as D29; frozen, then scored on full C (diagnostic) and E3; D29 rootconf control",
            "this_run": {k: v for k, v in ident.items() if k not in ("g_of", "stage")}, "g_selected": ident["g_of"],
            "control": {k: v for k, v in cident.items() if k not in ("g_of", "stage")},
            "means": {k: float(frame[k].mean()) for k in mean_keys},
            "split_candidates": int((frame["n_terminal"] > 1).sum()),
            "positive_e4_weights": int((frame["e4_weight"] > 0).sum()),
            "partitions": {"recentsearch": dup(list(frame["candidate"])), "d29": dup(list(frame["d29_candidate"]))},
            "caveats": ["six related candidates, not six independent experiments; development evidence only",
                        "S_recent is smaller than D29 S; differences are not a pure time effect",
                        "d29_exposed_S_recent rescores D29 predictions whose search used those rows",
                        "C was used by neither search but is time/space correlated with S and fitting; not an untouched test",
                        "C recent/older strata are groupings of the same frozen predictions; descriptive only",
                        "C has no gate: no pruning, root fallback or E4 use"],
        }
    else:
        summary = None
    summary = summary or {
        "contrast": "D29 (experiment-plan A4): rootconf search on S, frozen, then scored on C (diagnostic) and E3; "
                    "D28 rootinc same roots as the control",
        "this_run": {k: v for k, v in ident.items() if k not in ("g_of", "stage")}, "g_selected": ident["g_of"],
        "control": {k: v for k, v in cident.items() if k not in ("g_of", "stage")},
        "means": {k: float(frame[k].mean()) for k in ("S_local_minus_root", "C_local_minus_root", "e3_local_minus_root",
                                                      "d28_exposed_S_local_minus_root", "d28_exposed_C_local_minus_root",
                                                      "d28_e3_local_minus_root")},
        "split_candidates": int((frame["n_terminal"] > 1).sum()),
        "positive_e4_weights": int((frame["e4_weight"] > 0).sum()),
        "partitions": {"rootconf": dup(list(frame["candidate"])), "d28": dup(list(frame["d28_candidate"]))},
        "caveats": ["six related candidates, not six independent experiments; development evidence only",
                    "C is isolated only from this new search; time/space correlation with S and fitting remains",
                    "d28_exposed_* rescore D28 predictions whose search used ALL of S and C: not independent",
                    "new S vs D28 full validation are not the same sample; ΔS−ΔC and ΔC−ΔE3 are descriptive only",
                    "C has no gate: no pruning, root fallback or E4 use"],
    }
    out.mkdir(parents=True)
    frame.to_csv(out / "candidates.csv", index=False, float_format="%.17g")
    pd.DataFrame(terminals).to_csv(out / "terminal_support.csv", index=False)
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "confusions.json", confusions)
    rid.write_json_atomic(out / "completion.json", {
        "stage": f"stage1_{mode}_compare", "code": rid.code_identity(), "runtime": rid.runtime_identity(),
        "outputs": {rel: sha for rel, sha in rid.output_hashes(out).items() if rel != "completion.json"}})
    print(json.dumps({k: summary[k] for k in ("means", "split_candidates", "positive_e4_weights")}, indent=1))


if __name__ == "__main__":
    main()
