"""Scenario-path C-label permutation check (implement.md §5; synthetic only, no real data).

Reuses the existing synthetic ``scenario_fixture`` and the same small-fixture patches as
tests/test_baseline.py::ScenarioStage1.test_real_scenario_root_splits_original_keys_and_weights_children.
For each strategy (A, B) it runs ``scenario_root`` twice in scratch dirs:
  1. on the fixture's Stage 1 input;
  2. after permuting ONLY the class labels of the confirmation (C) original keys among themselves,
     applied to every row of those keys (eval row and, for B, all fit_variant copies).
C keys come from run 1's label-blind fold_membership. The check verifies that C keys' variant
copies are never F/S rows, and that every artifact except the C diagnostics is byte-identical:
root and candidate files, checkpoints, partitions, maps/routes, fit logs and support. No product
or test module is changed; nothing is written outside temporary dirs and this probe's summary.
Content comparison: gzip members decompressed (headers carry an mtime); JSON with the scratch\ncheckpoint dir normalised. A same-input rerun is the determinism control. Comparator-only corrections vs the first version
(see c_permutation_check_initial_failed.json): gzip container metadata and the scratch checkpoint dir path;
all content, row order, checkpoint model bytes and non-C predictions are still compared.\nUsage: python c_permutation_check.py
"""
import gzip
import hashlib
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve()
PKG = HERE.parents[5] / "FEWSNETGeoXGBExperiment"
sys.path.insert(0, str(PKG))
os.chdir(PKG)
import pandas as pd  # noqa: E402

from app import main_model_GF as mgf  # noqa: E402
from src.experiment import plan  # noqa: E402
from src.partition import transformation as trans  # noqa: E402
sys.path.insert(0, str(PKG / "tests"))
from test_baseline import SMALL_G, m, scenario_fixture  # noqa: E402

C_DIAGNOSTIC_FILES = {"confirmation_predictions.csv.gz"}      # C scored after freeze (D29)
C_DIAGNOSTIC_KEYS = {"confirmation", "timings"}               # candidate.json blocks excluded
T = "2020-10"


def run_root(frame, features, workdir):
    with patch.object(trans, "CONTIGUITY", False), \
            patch.object(trans, "generate_count_grid", return_value=(None, 0, 1)), \
            patch.dict(plan.G_CONFIGS, SMALL_G), \
            patch.dict(plan.FIT_SUPPORT, {"rows": 20, "areas": 5, "dates": 3, "classes": 2}), \
            patch.dict(plan.STAGE1_VAL_SUPPORT, {"rows": 5, "areas": 3, "dates": 2}), \
            patch.object(mgf, "MAX_DEPTH", 3), redirect_stdout(StringIO()):
        here = os.getcwd()
        os.chdir(workdir)
        try:
            rec = mgf.scenario_root(frame, "r50", 42, 4, T, Path(workdir), Path(workdir) / "ck", None, features)
        finally:
            os.chdir(here)
    return rec


def inventory(root: Path):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if p.suffix == ".json":
                d = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    d = {k: v for k, v in d.items() if k not in C_DIAGNOSTIC_KEYS}
                    if isinstance(d.get("checkpoints"), dict):   # scratch-dir path only; file hashes kept
                        d["checkpoints"] = {**d["checkpoints"], "dir": Path(d["checkpoints"]["dir"]).name}
                text = json.dumps(d, sort_keys=True, default=str).replace(str(root), "<ROOT>")
                out[rel] = hashlib.sha256(text.encode()).hexdigest()
            elif p.name == "fold_membership.csv.gz":
                f = pd.read_csv(p).drop(columns=["class_code"])   # roles/keys only; C labels differ by design
                out[rel] = hashlib.sha256(f.to_csv(index=False).encode()).hexdigest()
            elif p.suffix == ".gz":   # gzip headers carry an mtime: compare decompressed content
                out[rel] = hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()
            else:
                out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


_, ctx, _ = scenario_fixture()
summary = {"fixture": "tests/test_baseline.py::scenario_fixture (synthetic)", "target": T, "results": {}}
for strategy in ("A", "B"):
    frame = ctx.stage1_input(m(T), 1, strategy)
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2, \
            tempfile.TemporaryDirectory() as t0:
        rec1 = run_root(frame, ctx.features, t1)
        run_root(frame.copy(), ctx.features, t0)
        inv0 = inventory(Path(t0))
        roles = pd.read_csv(Path(t1) / "fold_membership.csv.gz")
        ev = frame[frame["role"] == "eval"]
        key_of = dict(zip(zip(ev["area"], ev["target_month"].map(lambda v: f"{int(v) // 12}-{int(v) % 12 + 1:02d}")),
                          ev["orig_key"]))
        role_of = {key_of[(a, tm)]: r for a, tm, r in zip(roles["area"], roles["target_month"], roles["role"])
                   if r != "heldout_target"}
        c_keys = sorted(k for k, r in role_of.items() if r == "confirmation")
        f_keys = {k for k, r in role_of.items() if r == "fitting"}
        s_keys = {k for k, r in role_of.items() if r == "validation"}
        labels = ev.set_index("orig_key")["class_code"]
        c_labels = labels.reindex(c_keys).to_numpy()
        perm = dict(zip(c_keys, list(c_labels[1:]) + list(c_labels[:1])))   # deterministic rotation
        frame2 = frame.copy()
        hit = frame2["orig_key"].isin(c_keys) & frame2["role"].isin(["eval", "fit_variant"])
        frame2.loc[hit, "class_code"] = frame2.loc[hit, "orig_key"].map(perm)
        changed = int((frame2["class_code"] != frame["class_code"]).sum())
        rec2 = run_root(frame2, ctx.features, t2)
        # F/S labels and all roles unchanged; C labels actually changed
        non_c = ~frame["orig_key"].isin(c_keys)
        fs_labels_unchanged = bool((frame2.loc[non_c, "class_code"] == frame.loc[non_c, "class_code"]).all())
        r2 = pd.read_csv(Path(t2) / "fold_membership.csv.gz")
        roles_equal = roles.drop(columns=["class_code"]).equals(r2.drop(columns=["class_code"]))
        fs_mask = roles["role"].isin(["fitting", "validation", "heldout_target"])
        fs_membership_labels_equal = roles.loc[fs_mask, "class_code"].equals(r2.loc[fs_mask, "class_code"])
        c_membership_labels_changed = int((roles.loc[~fs_mask, "class_code"].to_numpy()
                                           != r2.loc[~fs_mask, "class_code"].to_numpy()).sum())
        cand = rec1["candidates"][0]
        fits1, fits2 = (json.loads((Path(t) / cand / "candidate.json").read_text(encoding="utf-8"))["fits"]
                        for t in (t1, t2))
        inv1, inv2 = inventory(Path(t1)), inventory(Path(t2))
        same_input_diff = sorted(k for k in set(inv1) | set(inv0) if inv1.get(k) != inv0.get(k))
        diff = sorted(k for k in set(inv1) | set(inv2) if inv1.get(k) != inv2.get(k))
        unexpected = [d for d in diff if Path(d).name not in C_DIAGNOSTIC_FILES]
        checkpoint_files = sorted(k for k in inv1 if k.startswith("ck/"))
        model_bytes_equal = bool(checkpoint_files) and all(inv1[k] == inv2.get(k) for k in checkpoint_files)
        variant_rows_of_c = int((frame["orig_key"].isin(c_keys) & (frame["role"] == "fit_variant")).sum())
        summary["results"][strategy] = {
            "c_keys": len(c_keys), "f_keys": len(f_keys), "s_keys": len(s_keys),
            "c_overlaps_f_or_s": len(set(c_keys) & (f_keys | s_keys)),
            "c_fit_variant_rows_in_input": variant_rows_of_c, "label_cells_changed": changed,
            "files_compared": len(inv1), "same_input_control_differences": same_input_diff,
            "differing_files": diff, "unexpected_differences": unexpected,
            "root_status": [rec1.get("status"), rec2.get("status")],
            "fs_input_labels_unchanged": fs_labels_unchanged, "roles_equal": roles_equal,
            "fs_membership_labels_equal": bool(fs_membership_labels_equal),
            "c_membership_labels_changed": c_membership_labels_changed,
            "child_fits": [fits1.get("child_fits"), fits2.get("child_fits")],
            "checkpoint_model_files_compared": len(checkpoint_files), "checkpoint_model_bytes_equal": model_bytes_equal,
            "pass": (changed > 0 and c_membership_labels_changed > 0 and fs_labels_unchanged and roles_equal
                     and bool(fs_membership_labels_equal) and min(fits1.get("child_fits", 0), fits2.get("child_fits", 0)) > 0
                     and model_bytes_equal and not unexpected and not same_input_diff
                     and not (set(c_keys) & (f_keys | s_keys)))}
summary["pass"] = all(r["pass"] for r in summary["results"].values())
HERE.with_name("c_permutation_check_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
if not summary["pass"]:
    raise SystemExit(1)
