"""D29 / A4 keyed diagnostic: six rootconf candidates vs the six D28 rootinc candidates (never fits).

python scripts/stage1_rootconf_compare.py --run-dir RUN --control-run D28_RUN [--control-code-rev 98adf48]
    [--code-rev PRODUCER_REV]
python scripts/stage1_rootconf_compare.py --mode recentsearch --run-dir RUN --control-run D29_RUN
    [--control-code-rev ab1ac83] [--code-rev PRODUCER_REV]      (D30 / A5; writes RUN/stage1_recentsearch_compare/)
python scripts/stage1_rootconf_compare.py --mode matchedsize --run-dir RUN --control-run D30_RUN
    --reference-run D29_RUN [--control-code-rev 5517fb4] [--reference-code-rev ab1ac83] [--code-rev PRODUCER_REV]
    (D31 / A6; D30 compared on complete C and E3 only; writes RUN/stage1_matchedsize_compare/)
python scripts/stage1_rootconf_compare.py --mode e1pair --run-dir RUN [--code-rev PRODUCER_REV]
    (D34 / A9; in-run hard_f1 vs brier_crisis on the 21 shared roots; writes RUN/stage1_e1pair_compare/)

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

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from scripts import stage1_tb3_compare as tc  # noqa: E402
from scripts.stage1_rootinc_compare import crisis_counts, round_columns  # noqa: E402
from scripts.run_stage1 import SPLIT_MODES, _names, scheduled_candidates, scheduled_roots  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.metrics import fourclass  # noqa: E402
from src.utils import acceptance as acc  # noqa: E402
from src.utils import run_identity as rid  # noqa: E402
from src.utils.split import confirmation_split, matched_size_sample  # noqa: E402

CompareError = tc.CompareError
KEY = ["area", "target_month"]
#: modes with a frozen-candidate confirmation set C
CONF_MODES = (plan.ROOTCONF, plan.RECENTSEARCH, plan.MATCHEDSIZE, plan.E1PAIR)
#: new mode -> (control mode, default control producer, column prefix of the control)
CONTROLS = {plan.ROOTCONF: (plan.ROOTINC, "98adf48", "d28"), plan.RECENTSEARCH: (plan.ROOTCONF, "ab1ac83", "d29"),
            plan.MATCHEDSIZE: (plan.RECENTSEARCH, "5517fb4", "d30")}
#: D31/A6: the D29 rootconf reference of the matched-size contrast
MATCHED_REFERENCE = (plan.ROOTCONF, "ab1ac83")
#: scheduled roots per mode (D31: six H/T x three search seeds)
EXPECTED_ROOTS = {plan.MATCHEDSIZE: 18, plan.E1PAIR: 21}
#: scheduled candidates per mode when it differs from the root count (D34: two E1 variants per root)
EXPECTED_CANDIDATES = {plan.E1PAIR: 42}


def expected_candidates(mode: str, entry: dict) -> list:
    """Scheduled candidate names of one root in the producer's canonical order (D34: hard_f1, brier_crisis)."""
    if mode == plan.E1PAIR:
        return [n for n, _ in plan.e1pair_candidate_names(entry["horizon"], entry["target_month"], entry["g_config"])]
    return [_names(mode, entry, entry["g_config"])[1]]


def check_completion_candidates(name: str, listed, expected: list) -> None:
    """The completion record lists exactly the expected candidates, in canonical order, no duplicates."""
    if list(listed) != list(expected):
        raise CompareError(f"{name}: completion lists {list(listed)}, not exactly {list(expected)} in producer order")


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
    n_expected = EXPECTED_ROOTS.get(mode, 6)
    n_cands = EXPECTED_CANDIDATES.get(mode, n_expected)
    if len(roots) != n_expected or len(cands) != n_cands:
        raise CompareError(f"the schedule does not give exactly {n_expected} {mode} roots / {n_cands} candidates")
    stage = run / SPLIT_MODES[mode][2]
    present = {p.name for p in (stage / "roots").iterdir() if p.is_dir()} if (stage / "roots").is_dir() else set()
    if present != set(roots):
        raise CompareError(f"{mode} roots: missing {sorted(set(roots) - present)}, unexpected {sorted(present - set(roots))}")
    cand_dirs = {p.name for p in (stage / "candidates").iterdir() if p.is_dir()} if (stage / "candidates").is_dir() else set()
    if cand_dirs != set(cands):
        raise CompareError(f"{mode} candidates: dirs {sorted(cand_dirs)} != scheduled {sorted(cands)}")
    code = code or rid.code_identity()
    runtime = None
    for name, entry in roots.items():
        want = sorted(c for c, e in cands.items() if e["root"] == name)
        expected = expected_candidates(mode, entry)          # canonical producer order
        if want != sorted(expected) or len(set(expected)) != len(expected):
            raise CompareError(f"{name}: expected exactly {expected} for {mode}, got {want}")
        record = tc.accept_root_record(stage, name, expected)
        check_completion_candidates(name, record.get("candidates", []), expected)
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
        for c in expected:
            cj = tc._json(stage / "candidates" / c / "candidate.json")
            if cj.get("increment_source") != "root":
                raise CompareError(f"{c}: candidate is not a shared-root increment candidate")
            if mode == plan.E1PAIR and cj.get("e1") != cands[c]["e1"]:
                raise CompareError(f"{c}: candidate.json e1 {cj.get('e1')!r} differs from the schedule")
        if (mode in CONF_MODES) != ("confirmation_split" in root) or \
                (mode == plan.RECENTSEARCH) != ("recent_search" in root) or \
                (mode == plan.MATCHEDSIZE) != ("matched_size" in root):
            raise CompareError(f"{name}: confirmation split / recent search presence does not match mode {mode}")
        for c in expected:
            if mode in CONF_MODES and not (stage / "candidates" / c / "confirmation_predictions.csv.gz").is_file():
                raise CompareError(f"{c}: no confirmation predictions")
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


def matched_roles(d29_members: pd.DataFrame, h, t, seed) -> pd.DataFrame:
    """D31/A6: re-derive the expected roles from the D29 membership: k_a from the A5 recent
    months on original S (dates only), the per-area draw ``matched_size_sample`` (numeric
    area order, date-sorted keys, one shuffle per area), the rest of S unused."""
    b = d29_members
    months = sorted(b.loc[b["role"].isin(["validation", "confirmation"]), "target_month"].unique())[-plan.RECENT_SEARCH_MONTHS:]
    if tuple(months) != tuple(plan.RECENT_SEARCH_DATES[(h, t)]):
        raise CompareError(f"h{h} {t}: D29 original validation gives recent months {months}, not the A5 table")
    s = b[b["role"] == "validation"]
    k = s[s["target_month"].isin(months)].groupby("area").size()
    drawn = matched_size_sample(s["area"].to_numpy(int), pd.PeriodIndex(s["target_month"], freq="M").asi8,
                                {int(a): int(n) for a, n in k.items()}, seed)
    role = b["role"].copy()
    role.loc[s.index[~drawn]] = "unused_search_history"
    return b.assign(role=role)


def same_roots_matched(new_members, d29_members, d30_members, h, t, seed, name) -> None:
    """New membership == D29 except S -> drawn sample + unused (re-drawn independently);
    per-area sample counts == the D30 recent-S counts; C/fitting/target identical to D30."""
    if new_members.duplicated(KEY).any():
        raise CompareError(f"{name}: duplicate membership keys")
    m = new_members.merge(matched_roles(d29_members, h, t, seed), on=KEY, how="outer", suffixes=("_new", "_old"),
                          indicator=True, validate="one_to_one")
    if (m["_merge"] != "both").any() or not (m["role_new"] == m["role_old"]).all() \
            or not (m["class_code_new"] == m["class_code_old"]).all():
        raise CompareError(f"{name}: membership differs from the A6 draw re-derived on the D29 keys")
    per_area = lambda mem: mem[mem["role"] == "validation"].groupby("area").size()   # noqa: E731
    if not per_area(new_members).equals(per_area(d30_members)):
        raise CompareError(f"{name}: per-area search counts differ from the D30 recent-S counts")
    fixed = lambda mem: mem[~mem["role"].isin(["validation", "unused_search_history"])].sort_values(KEY).reset_index(drop=True)   # noqa: E731
    if not fixed(new_members).equals(fixed(d30_members)):
        raise CompareError(f"{name}: fitting / C / target membership differs from D30")


def load_matched(stage, name, cand, d30, d29, h, t) -> dict:
    """Read every input of one matched row (``d30``/``d29`` = (stage, root, candidate))."""
    rb = tc._json(stage / "roots" / name / "root.json")["root_booster_sha256"]
    out = {"seed": int(tc._json(stage / "roots" / name / "root.json")["matched_size"]["search_seed"])}
    for key, (st, rt, cd, mode) in (("new", (stage, name, cand, plan.MATCHEDSIZE)),
                                    ("d30", (*d30, plan.RECENTSEARCH)), ("d29", (*d29, plan.ROOTCONF))):
        row, target, conf, part = tc.method_row(st, rt, cd, h, t, mode, rb)
        members = tc.read(st / "roots" / rt / "fold_membership.csv.gz")
        out[key] = {"row": row, "target": target, "conf": conf, "partition": part, "members": members,
                    "confirmation": keyed_confirmation(st, cd, members)}
        if key != "d30":
            out[key]["validation"] = tc.keyed_validation(st, rt, cd)[0]
    for key, (st, rt, _cd) in (("d30", d30), ("d29", d29)):
        same_target_and_root(stage, name, st, rt, out["new"]["target"], out[key]["target"])
    c = tc._json(stage / "candidates" / cand / "candidate.json")
    last_save = {e["saved_as"]: e for e in c["fits"]["saved_log"]}
    out["new"]["rounds"] = round_columns(stage, cand)
    out["new"]["root_booster"] = rb
    out["new"]["branch_booster"] = {b: (last_save.get(b) or {}).get("booster_sha256") or (rb if b == "root" else None)
                                    for b in c["partition"]["terminal_partitions"]}
    return out


def matched_row(data: dict, h, t, cand) -> tuple:
    """Pure D31 row: joins, identity checks and scores on loaded frames (no file access).

    Primary D30 comparisons on complete C and E3 only (D30 saved no predictions for its
    unused older S); own search-sample scores are descriptive; D29 rescored on the sample
    keys is labelled exposed."""
    new, d30, d29 = data["new"], data["d30"], data["d29"]
    same_roots_matched(new["members"], d29["members"], d30["members"], h, t, data["seed"], cand)
    s_frame = new["validation"]
    d29_s = d29["validation"].merge(s_frame[KEY], on=KEY, validate="one_to_one")
    if len(d29_s) != len(s_frame) or not (d29_s.sort_values(KEY)["y_root"].to_numpy()
                                          == s_frame.sort_values(KEY)["y_root"].to_numpy()).all():
        raise CompareError(f"{cand}: search-sample keys/root predictions differ from D29 on the same keys")
    c_new = new["confirmation"]
    for key in ("d30", "d29"):
        chk = c_new.merge(data[key]["confirmation"], on=KEY, how="outer", suffixes=("", "_o"), indicator=True,
                          validate="one_to_one")
        if (chk["_merge"] != "both").any() or not (chk["y_root"] == chk["y_root_o"]).all() \
                or not (chk["y_true"] == chk["y_true_o"]).all():
            raise CompareError(f"{cand}: C keys/truth/root predictions differ from {key}")
    recent = list(plan.RECENT_SEARCH_DATES[(h, t)])
    strata = lambda c, p: ((p, c), (f"{p}_recent6", c[c["target_month"].isin(recent)]),   # noqa: E731
                           (f"{p}_older", c[~c["target_month"].isin(recent)]))
    blocks, mats = {}, {}
    for prefix, frame in (("S_sample", s_frame), ("d29_exposed_S_sample", d29_s), *strata(c_new, "C"),
                          *strata(d30["confirmation"], "d30_C"), *strata(d29["confirmation"], "d29_C")):
        b, mt = pool_block(frame, prefix)
        blocks.update(b); mats.update(mt)
    mats.update({k: v for k, v in new["conf"].items() if k.startswith("target_")})
    mats["d30_target_local"], mats["d29_target_local"] = d30["conf"]["target_local"], d29["conf"]["target_local"]
    e3 = lambda r: Fraction(r["e3_local_minus_root_exact"])   # noqa: E731
    nr, members = new["row"], new["members"]
    role = lambda r: members[members["role"] == r]   # noqa: E731
    sample = role("validation")
    row = {"horizon": h, "target_month": t, "search_seed": data["seed"], "candidate": cand,
           "d30_candidate": d30["row"]["candidate"], "d29_candidate": d29["row"]["candidate"],
           "e3_root_crisis_f1": nr["e3_root_crisis_f1"], "e3_local_crisis_f1": nr["e3_local_crisis_f1"],
           "e3_local_minus_root": nr["e3_local_minus_root"], "e3_local_minus_root_exact": nr["e3_local_minus_root_exact"],
           "e3_fourclass_local_minus_root": nr["e3_local_fourclass"] - nr["e3_root_fourclass"],
           **crisis_counts(new["conf"]["target_root"], "e3_root"), **crisis_counts(new["conf"]["target_local"], "e3_local"),
           "d30_e3_local_minus_root": d30["row"]["e3_local_minus_root"],
           "d29_e3_local_minus_root": d29["row"]["e3_local_minus_root"],
           "e3_minus_d30_e3": float(e3(nr) - e3(d30["row"])), "e3_minus_d29_e3": float(e3(nr) - e3(d29["row"])),
           **blocks,
           **{f"{k}_minus_d30": blocks[f"{k}_local_minus_root"] - blocks[f"d30_{k}_local_minus_root"]
              for k in ("C", "C_recent6", "C_older")},
           **{f"{k}_minus_d29": blocks[f"{k}_local_minus_root"] - blocks[f"d29_{k}_local_minus_root"]
              for k in ("C", "C_recent6", "C_older")},
           **tc.support(sample, "S_sample"), **tc.support(role("unused_search_history"), "unused"),
           **tc.support(role("confirmation"), "C"),
           "S_sample_recent_share": float(sample["target_month"].isin(recent).mean()),
           "S_sample_recent_rows": int(sample["target_month"].isin(recent).sum()),
           "C_rows_root_unassigned": int((c_new["routing"] != "terminal_branch").sum()),
           "C_rows_on_root_branch_name": int((c_new["branch_id"] == "root").sum()),   # literal branch name only
           **root_booster_rows(c_new, new["branch_booster"], new["root_booster"]),
           "unsplit": int(nr["n_terminal"] == 1),
           **new["rounds"],
           "n_terminal": nr["n_terminal"], "accepted_splits": nr["accepted_splits"],
           "distinct_terminal_boosters": nr["distinct_terminal_boosters"], "e4_weight": nr["e4_weight"],
           "d30_n_terminal": d30["row"]["n_terminal"], "d30_unsplit": int(d30["row"]["n_terminal"] == 1),
           "d30_e4_weight": d30["row"]["e4_weight"],
           "identical_partition_to_d30": new["partition"] == d30["partition"]}
    terminals = [{"horizon": h, "target_month": t, "search_seed": data["seed"], "candidate": cand, **r}
                 for which, fr in (("S_sample", s_frame), *strata(c_new, "C")) for r in terminal_support(fr, which)]
    return row, mats, terminals


def root_booster_rows(c_frame, branch_booster: dict, root_booster: str) -> dict:
    """C rows actually predicted by the root booster: routed to a terminal whose saved booster
    is the root booster (incl. inherited parent copies), or not routed to a terminal."""
    booster = c_frame["branch_id"].map(branch_booster)
    on_root = (c_frame["routing"] != "terminal_branch") | (booster == root_booster)
    return {"C_rows_root_booster": int(on_root.sum()), "C_rows_booster_unresolved": int(booster.isna().sum())}


def ari_common(a, b):
    """Adjusted Rand index of two canonical partitions on their common area coverage."""
    from sklearn.metrics import adjusted_rand_score
    da, db = dict(zip(*a)), dict(zip(*b))
    common = sorted(set(da) & set(db))
    return float(adjusted_rand_score([da[x] for x in common], [db[x] for x in common])), len(common)


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


def matched_summary(frame, partitions, ident, cident, dup) -> dict:
    """D31 descriptive summary: per seed and per seed pair, all-root and split-only means,
    fallback (unsplit) counts, date support, and ARI between seeds and vs D30. No gates."""
    keys = ("e3_local_minus_root", "e3_minus_d30_e3", "C_local_minus_root", "C_recent6_local_minus_root",
            "C_older_local_minus_root", "C_minus_d30", "C_recent6_minus_d30", "C_older_minus_d30",
            "S_sample_local_minus_root", "d29_exposed_S_sample_local_minus_root")
    means = lambda g: {k: float(g[k].mean()) for k in keys} if len(g) else None   # noqa: E731
    seeds = sorted(frame["search_seed"].unique())
    d30 = frame.drop_duplicates(["horizon", "target_month"])
    per_seed = {}
    for seed in seeds:
        g = frame[frame["search_seed"] == seed]
        per_seed[str(seed)] = {"all_roots": means(g), "split_only": means(g[g["unsplit"] == 0]),
                               "unsplit_roots": int(g["unsplit"].sum()), "positive_e3": int((g["e3_local_minus_root"] > 0).sum()),
                               "positive_e4_weights": int((g["e4_weight"] > 0).sum()),
                               "mean_sample_recent_share": float(g["S_sample_recent_share"].mean()),
                               "sample_dates": [int(x) for x in g["S_sample_dates"]],
                               "partitions": dup(list(g["candidate"]))}
    pairs = {}
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            ga = frame[frame["search_seed"] == a].set_index(["horizon", "target_month"]).sort_index()
            gb = frame[frame["search_seed"] == b].set_index(["horizon", "target_month"]).sort_index()
            ari = [dict(zip(("ari", "common_areas"), ari_common(partitions[x], partitions[y])))
                   for x, y in zip(ga["candidate"], gb["candidate"])]
            pairs[f"{a}-{b}"] = {"mean_e3_difference": float((ga["e3_local_minus_root"] - gb["e3_local_minus_root"]).mean()),
                                 "mean_C_difference": float((ga["C_local_minus_root"] - gb["C_local_minus_root"]).mean()),
                                 "ari_per_root": ari}
    ari_d30 = {str(seed): [dict(zip(("ari", "common_areas"), ari_common(partitions[r["candidate"]], partitions[r["d30_candidate"]])))
                           for _, r in frame[frame["search_seed"] == seed].sort_values(["horizon", "target_month"]).iterrows()]
               for seed in seeds}
    return {
        "contrast": "D31 (A6): per-area matched search counts (= D30 recent-six S) drawn label-blind from all original S "
                    "dates, three search seeds; same D29 fitting/root/C; frozen, then scored on full C (diagnostic) and "
                    "E3; D30 recentsearch primary control (C/E3 only), D29 rootconf reference",
        "this_run": {k: v for k, v in ident.items() if k not in ("g_of", "stage")}, "g_selected": ident["g_of"],
        "control": {k: v for k, v in cident.items() if k not in ("g_of", "stage")},
        "per_seed": per_seed, "seed_pairs": pairs, "ari_vs_d30_per_root": ari_d30,
        "d30_reference": {"all_roots_mean_e3": float(d30["d30_e3_local_minus_root"].mean()),
                          "split_only_mean_e3": float(d30.loc[d30["d30_unsplit"] == 0, "d30_e3_local_minus_root"].mean()),
                          "unsplit_roots": int(d30["d30_unsplit"].sum()),
                          "mean_C": float(d30["d30_C_local_minus_root"].mean())},
        "caveats": ["18 related candidates (six repeatedly exposed targets x three search seeds); development evidence only",
                    "per-area counts force 34-57% of D30's recent rows into every draw (expected overlap 67-79%): limited "
                    "contrast; no observed difference is inconclusive, not attribution to size or fallback",
                    "no D30 search-score comparison: D30 saved predictions only for its S_recent",
                    "search-sample scores are adaptively reused (descriptive); d29_exposed_* used those rows in D29's search",
                    "C was used by no search but is time/space correlated with S and fitting; not an untouched test",
                    "ARI is descriptive only; no gates, hypothesis tests or automatic recipe selection"],
    }


# ------------------------------------------------------------------ D34 / A9 e1pair (in-run hard vs Brier)

E1_TOKENS = dict((e1, tok) for tok, e1 in plan.E1PAIR_VARIANTS)      # hard_f1 -> e1hard, brier_crisis -> e1brier


def load_e1pair(stage, name, pair: dict, h, t) -> dict:
    """File reads for one D34 root: ``pair`` maps e1 variant -> candidate name."""
    rb = tc._json(stage / "roots" / name / "root.json")["root_booster_sha256"]
    out = {"root_booster": rb}
    for e1, cand in pair.items():
        row, target, conf, part = tc.method_row(stage, name, cand, h, t, e1, rb)
        s_frame, members = tc.keyed_validation(stage, name, cand)
        cj = tc._json(stage / "candidates" / cand / "candidate.json")
        last = {e["saved_as"]: e for e in cj["fits"]["saved_log"]}
        out[e1] = {"candidate": cand, "row": row, "target": target, "conf": conf, "partition": part,
                   "S": s_frame, "C": keyed_confirmation(stage, cand, members), "record": cj,
                   "branch_booster": {b: (last.get(b) or {}).get("booster_sha256") or (rb if b == "root" else None)
                                      for b in cj["partition"]["terminal_partitions"]},
                   "checkpoint_training_rows": [int((last.get(b) or {}).get("rows") or 0)
                                                for b in cj["partition"]["terminal_partitions"]],
                   "assigned": assigned_pools(tc.read(stage / "candidates" / cand / "assignment_evidence.csv",
                                                      ("prediction_branch_id", "spatial_partition_id"))),
                   "rounds": round_columns(stage, cand)}
    return out


def assigned_pools(evidence: pd.DataFrame) -> dict:
    """Per spatial terminal: assigned fitting rows and areas (D32 evidence; s-1 = no search rows)."""
    g = evidence.groupby("spatial_partition_id").agg(fitting_rows=("fitting_rows", "sum"), areas=("FEWSNET_admin_code", "size"))
    spatial = g.drop(index=[i for i in ("s-1",) if i in g.index])
    summary = {"assigned_terminals": int(len(spatial)),
               "assigned_fitting_rows_min": int(spatial["fitting_rows"].min()) if len(spatial) else None,
               "assigned_fitting_rows_max": int(spatial["fitting_rows"].max()) if len(spatial) else None,
               "assigned_areas_min": int(spatial["areas"].min()) if len(spatial) else None,
               "assigned_areas_max": int(spatial["areas"].max()) if len(spatial) else None,
               "unsearched_s1_areas": int(g.loc["s-1", "areas"]) if "s-1" in g.index else 0,
               "unsearched_s1_fitting_rows": int(g.loc["s-1", "fitting_rows"]) if "s-1" in g.index else 0}
    return {"by_terminal": {k: {"fitting_rows": int(v.fitting_rows), "areas": int(v.areas)} for k, v in g.iterrows()},
            "summary": summary}


def e1pair_row(data: dict, h, t) -> tuple:
    """Pure D34 row: same-root identity checks, S/C/E3 root vs hard vs Brier, diagnostics, structure."""
    hard, brier = data["hard_f1"], data["brier_crisis"]
    for part, cols in (("target", ["y_true", "y_root"]), ("S", ["y_true", "y_root"]), ("C", ["y_true", "y_root"])):
        a, b = hard[part].sort_values(KEY).reset_index(drop=True), brier[part].sort_values(KEY).reset_index(drop=True)
        if not (a[KEY].equals(b[KEY]) and all((a[c].to_numpy() == b[c].to_numpy()).all() for c in cols)):
            raise CompareError(f"h{h} {t}: {part} keys/truth/root predictions differ between the paired candidates")
    blocks, mats = {}, {}
    for part in ("S", "C"):
        for e1, d in (("hard_f1", hard), ("brier_crisis", brier)):
            b, mt = pool_block(d[part], f"{part}_{E1_TOKENS[e1]}")
            blocks.update(b); mats.update(mt)
        blocks[f"{part}_brier_minus_hard"] = float(Fraction(blocks[f"{part}_e1brier_local_crisis_f1_exact"])
                                                   - Fraction(blocks[f"{part}_e1hard_local_crisis_f1_exact"]))
    e3 = lambda d, k: Fraction(d["row"][k])   # noqa: E731  (d = candidate dict)
    row = {"horizon": h, "target_month": t, "hard_candidate": hard["candidate"], "brier_candidate": brier["candidate"],
           "e3_root_crisis_f1": hard["row"]["e3_root_crisis_f1"],
           **crisis_counts(hard["conf"]["target_root"], "e3_root")}
    for e1, d in (("hard_f1", hard), ("brier_crisis", brier)):
        tok = E1_TOKENS[e1]
        r, rec = d["row"], d["record"]
        root_dec = [x for x in rec["partition"]["decisions"] if (x.get("branch_id") or "") == ""]
        diag = (root_dec[0].get("scan_diagnostics") if root_dec else None) or {}
        pools, ck = d["assigned"], d["checkpoint_training_rows"]
        row.update({f"e3_{tok}_local_crisis_f1": r["e3_local_crisis_f1"],
                    f"e3_{tok}_local_minus_root": r["e3_local_minus_root"],
                    f"e3_{tok}_local_minus_root_exact": r["e3_local_minus_root_exact"],
                    f"e3_{tok}_fourclass_local_minus_root": r["e3_local_fourclass"] - r["e3_root_fourclass"],
                    **crisis_counts(d["conf"]["target_local"], f"e3_{tok}_local"),
                    f"{tok}_n_terminal": r["n_terminal"], f"{tok}_unsplit": int(r["n_terminal"] == 1),
                    f"{tok}_accepted_splits": r["accepted_splits"],
                    f"{tok}_distinct_terminal_boosters": r["distinct_terminal_boosters"],
                    f"{tok}_decisions": len(rec["partition"]["decisions"]),
                    f"{tok}_root_outcome": root_dec[0].get("outcome") if root_dec else None,
                    f"{tok}_root_parent_kept_fitting_areas": (root_dec[0].get("parent_kept_fitting") or {}).get("areas") if root_dec else None,
                    **{f"{tok}_{k}": v for k, v in pools["summary"].items()},
                    f"{tok}_checkpoint_training_rows_min": min(ck) if ck else None,   # booster fit rows (retained-root = global)
                    f"{tok}_checkpoint_training_rows_max": max(ck) if ck else None,
                    **{f"{tok}_root_scan_{k}": v for k, v in diag.items() if k != "e1"},
                    **{f"{tok}_{k}": v for k, v in root_booster_rows(d["C"], d["branch_booster"], data["root_booster"]).items()},
                    **{f"{tok}_d32_{k}": v for k, v in ((rec.get("assignment_evidence") or {}).get("areas_by_status") or {}).items()},
                    **{f"{tok}_{k}": v for k, v in d["rounds"].items()}})
        mats[f"target_{tok}_local"] = d["conf"]["target_local"]
    mats["target_root"] = hard["conf"]["target_root"]
    row["e3_brier_minus_hard"] = float(e3(brier, "e3_local_crisis_f1_exact") - e3(hard, "e3_local_crisis_f1_exact"))
    row.update(blocks)
    row["identical_partition"] = hard["partition"] == brier["partition"]
    diagnostics = {E1_TOKENS[e1]: {"scan": [x.get("scan_diagnostics") for x in d["record"]["partition"]["decisions"]
                                            if x.get("scan_diagnostics")],
                                   "assigned_pools": d["assigned"]["by_terminal"]}
                   for e1, d in (("hard_f1", hard), ("brier_crisis", brier))}
    return row, mats, diagnostics


def e1pair_summary(frame, confusions) -> dict:
    """Per-H, per-target and pooled descriptions; mean-fold deltas kept separate from pooled confusion."""
    deltas = ["e3_e1hard_local_minus_root", "e3_e1brier_local_minus_root", "e3_brier_minus_hard",
              "C_e1hard_local_minus_root", "C_e1brier_local_minus_root", "C_brier_minus_hard",
              "S_e1hard_local_minus_root", "S_e1brier_local_minus_root", "S_brier_minus_hard"]
    desc = lambda g: {k: {"mean": float(g[k].mean()), "positive": int((g[k] > 0).sum()),   # noqa: E731
                          "negative": int((g[k] < 0).sum()), "zero": int((g[k] == 0).sum())} for k in deltas}
    pooled = {}
    for which in sorted({c["which"] for c in confusions}):
        m = np.sum([np.asarray(c["fourclass"]) for c in confusions if c["which"] == which], axis=0)
        tp, fp, fn = int(m[2:, 2:].sum()), int(m[:2, 2:].sum()), int(m[2:, :2].sum())
        pooled[which] = {"fourclass": m.astype(int).tolist(), "crisis_tp_fp_fn": [tp, fp, fn],
                         "crisis_f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None}
    return {"mean_fold_all": desc(frame),
            "by_horizon": {str(h): desc(g) for h, g in frame.groupby("horizon")},
            "by_target": {t: desc(g) for t, g in frame.groupby("target_month")},
            "pooled_confusion": pooled,
            "unsplit": {"e1hard": int(frame["e1hard_unsplit"].sum()), "e1brier": int(frame["e1brier_unsplit"].sum())},
            "identical_partitions": int(frame["identical_partition"].sum())}


def e1pair_main(args) -> None:
    run = args.run_dir.resolve()
    out = run / f"stage1_{plan.E1PAIR}_compare"
    rid.refuse_existing(out, "the e1pair comparison")
    roots, cands, ident = accept_mode(run, plan.E1PAIR, rid.code_identity_at(args.code_rev) if args.code_rev else None,
                                      args.code_rev)
    stage = ident["stage"]
    rows, confusions, diagnostics = [], [], {}
    for name, entry in sorted(roots.items(), key=lambda kv: (kv[1]["horizon"], kv[1]["target_month"])):
        h, t = entry["horizon"], entry["target_month"]
        pair = {e["e1"]: c for c, e in cands.items() if e["root"] == name}
        row, mats, diag = e1pair_row(load_e1pair(stage, name, pair, h, t), h, t)
        rows.append(row)
        diagnostics[name] = diag
        for which, mtx in mats.items():
            confusions.append({"horizon": h, "target_month": t, "root": name, "which": which,
                               "fourclass": [[int(x) for x in r] for r in mtx],
                               "crisis_[[nn,nc],[cn,cc]]": tc.crisis_matrix(mtx)})
    frame = pd.DataFrame(rows)
    summary = {
        "contrast": "D34 (A9): 21 frozen roots, one root fit each shared by hard_f1 and brier_crisis E1 (42 candidates); "
                    "E2/C/E3 argmax crisis F1; development contrast, not independent validation",
        "this_run": {k: v for k, v in ident.items() if k not in ("g_of", "stage")}, "g_selected": ident["g_of"],
        **e1pair_summary(frame, confusions),
        "caveats": ["21 related roots on dates previously used by v1/v7 RF/G selection; development evidence only",
                    "scan diagnostics are descriptive; Brier does not guarantee distinct or nonzero scores and a "
                    "nonzero g is not independent evidence",
                    "S is adaptively reused; C was used by no search but is correlated with S/fitting; E3 descriptive",
                    "no E4 weights, no maps for Stage2, no selection of a family"]}
    out.mkdir(parents=True)
    frame.to_csv(out / "candidates.csv", index=False, float_format="%.17g")
    rid.write_json_atomic(out / "summary.json", summary)
    rid.write_json_atomic(out / "confusions.json", confusions)
    rid.write_json_atomic(out / "scan_diagnostics.json", diagnostics)
    rid.write_json_atomic(out / "completion.json", {
        "stage": "stage1_e1pair_compare", "code": rid.code_identity(), "runtime": rid.runtime_identity(),
        "outputs": {rel: sha for rel, sha in rid.output_hashes(out).items() if rel != "completion.json"}})
    print(json.dumps({k: summary[k] for k in ("mean_fold_all", "unsplit")}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(CONTROLS) + [plan.E1PAIR], default=plan.ROOTCONF,
                        help="rootconf (D29 vs D28 rootinc), recentsearch (D30 vs D29 rootconf) or "
                             "matchedsize (D31 vs D30 recentsearch, D29 reference)")
    parser.add_argument("--reference-run", type=Path, default=None, help="matchedsize only: the D29 rootconf run")
    parser.add_argument("--reference-code-rev", default=MATCHED_REFERENCE[1],
                        help="matchedsize only: committed producer of the D29 reference (default ab1ac83)")
    parser.add_argument("--control-run", type=Path, default=None,
                        help="the D28 rootinc / D29 rootconf / D30 run (not used by e1pair: hard_f1 is the in-run pair)")
    parser.add_argument("--control-code-rev", default=None,
                        help="committed producer of the controls (default: 98adf48 for rootconf, ab1ac83 for recentsearch)")
    parser.add_argument("--code-rev", default=None,
                        help="committed producer of this run when it differs from the current code (default: current)")
    args = parser.parse_args()
    mode = args.mode
    if mode == plan.E1PAIR:
        return e1pair_main(args)
    if args.control_run is None:
        raise CompareError(f"{mode} needs --control-run")
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
    if mode == plan.MATCHEDSIZE:
        if args.reference_run is None:
            raise CompareError("matchedsize needs --reference-run (the D29 rootconf run)")
        reference = args.reference_run.resolve()
        rroots, rcands, rident = accept_mode(reference, MATCHED_REFERENCE[0], rid.code_identity_at(args.reference_code_rev),
                                             args.reference_code_rev)
        if rident["g_of"] != ident["g_of"]:
            raise CompareError("reference G selection differs from this run's")
        ref_of = {(e["horizon"], e["target_month"]): n for n, e in rroots.items()}
        for name, entry in sorted(roots.items(), key=lambda kv: (kv[1]["horizon"], kv[1]["target_month"],
                                                                 kv[1]["matched_size_seed"])):
            h, t = entry["horizon"], entry["target_month"]
            cand = next(c for c, e in cands.items() if e["root"] == name)
            croot, rroot = ctl_of[(h, t)], ref_of[(h, t)]
            d30 = (cstage, croot, next(c for c, e in ccands.items() if e["root"] == croot))
            d29 = (rident["stage"], rroot, next(c for c, e in rcands.items() if e["root"] == rroot))
            data = load_matched(stage, name, cand, d30, d29, h, t)
            if data["seed"] != entry["matched_size_seed"]:
                raise CompareError(f"{name}: root.json search seed differs from the schedule")
            row, mats, terms = matched_row(data, h, t, cand)
            rows.append(row)
            terminals += terms
            partitions[cand], partitions[d30[2]] = data["new"]["partition"], data["d30"]["partition"]
            for which, mtx in mats.items():
                confusions.append({"horizon": h, "target_month": t, "search_seed": data["seed"], "candidate": cand,
                                   "which": which, "exposed": which.startswith("d29_exposed"),
                                   "fourclass": [[int(x) for x in r] for r in mtx],
                                   "crisis_[[nn,nc],[cn,cc]]": tc.crisis_matrix(mtx)})
        cident["reference"] = {k: v for k, v in rident.items() if k not in ("g_of", "stage")}
    for name, entry in sorted(roots.items(), key=lambda kv: (kv[1]["horizon"], kv[1]["target_month"])):
        h, t = entry["horizon"], entry["target_month"]
        cand = next(c for c, e in cands.items() if e["root"] == name)
        croot = ctl_of[(h, t)]
        ccand = next(c for c, e in ccands.items() if e["root"] == croot)
        if mode == plan.MATCHEDSIZE:
            break
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

    if mode == plan.MATCHEDSIZE:
        summary = matched_summary(frame, partitions, ident, cident, dup)
    elif mode == plan.RECENTSEARCH:
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
    shown = ("per_seed", "d30_reference") if mode == plan.MATCHEDSIZE else ("means", "split_candidates", "positive_e4_weights")
    print(json.dumps({k: summary[k] for k in shown}, indent=1))


if __name__ == "__main__":
    main()
