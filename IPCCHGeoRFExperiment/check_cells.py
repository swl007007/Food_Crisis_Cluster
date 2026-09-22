"""Cross-cell verification for the cohort x gate factorial (R2, R3/A2, R4/A3, R6).

The ablation's whole claim is that the only things that differ between cells are the
cohort filter and the split gate. Recording each cell's effective configuration does not
establish that; comparing them does. This script is that comparison, and it fails loudly
rather than printing a warning.

Checks, per the task's PRD:

* R2  - every REPORTED_CONFIG_KEYS value must match cell C1's, except the declared gate.
* R3/A2 - each cell carries a three-namespace split-gate readback agreeing with its
          declared gate, including the default-gate cells.
* R4/A3 - cohort counts reconcile to the `admin_code >= 100000` rule and sum back to the
          audited source totals.
* R6  - candidate split evaluation counts are published per cell.

    python IPCCHGeoRFExperiment/check_cells.py --runs-dir IPCCHGeoRFExperiment/runs
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

#: The factorial, in reporting order. C1 is the reference every other cell is compared
#: against, and the cell that must reproduce the prior baseline.
CELLS = (
    ("C1", "abl-C1-all-gate010", "all", 0.010),
    ("C2", "abl-C2-all-gate005", "all", 0.005),
    ("C3", "abl-C3-nonch-gate010", "non_ch", 0.010),
    ("C4", "abl-C4-nonch-gate005", "non_ch", 0.005),
    ("C5", "abl-C5-chonly-gate010", "ch_only", 0.010),
    ("C6", "abl-C6-chonly-gate005", "ch_only", 0.005),
)

GATE_KEY = "MIN_CLASS_1_IMPROVEMENT_THRESHOLD"
GATE_NAMESPACES = ("config", "src.tests.sig_test", "src.partition.transformation")
CH_FLOOR = 100000
#: The audited full-source totals both cohort halves must sum back to.
EXPECTED_VALID = 42695
EXPECTED_AREAS = 6227


def load(run: Path) -> dict:
    return json.loads((run / "manifest.json").read_text())


def check_settings_drift(cells: dict) -> list[str]:
    """R2: only the gate may differ from C1."""
    problems = []
    reference = cells["C1"]["manifest"]["stages"]["stage1_fit"]["effective_config"]["config_module"]
    for cid, info in cells.items():
        if cid == "C1":
            continue
        actual = info["manifest"]["stages"]["stage1_fit"]["effective_config"]["config_module"]
        for key in sorted(set(reference) | set(actual)):
            if key == GATE_KEY:
                continue
            if reference.get(key) != actual.get(key):
                problems.append(
                    f"{cid}: {key} drifted from C1 "
                    f"({reference.get(key)!r} -> {actual.get(key)!r})"
                )
        missing = sorted(set(reference) - set(actual))
        if missing:
            problems.append(f"{cid}: missing recorded settings {missing}")
    return problems


def check_gate(cells: dict) -> list[str]:
    """R3/A2: every cell, default or overridden, proves its gate in all namespaces."""
    problems = []
    for cid, info in cells.items():
        declared = info["declared_gate"]
        evidence = info["manifest"]["stages"]["stage1_fit"].get("split_gate")
        if not evidence:
            problems.append(
                f"{cid}: no split-gate readback recorded; a default-gate cell needs "
                "run-bound proof too"
            )
            continue
        readback = evidence.get("readback", {})
        if set(readback) != set(GATE_NAMESPACES):
            problems.append(
                f"{cid}: readback covers {sorted(readback)}, expected "
                f"{sorted(GATE_NAMESPACES)}"
            )
        for namespace, value in readback.items():
            if value != declared:
                problems.append(
                    f"{cid}: {namespace} gated at {value}, declared {declared}"
                )
        if evidence.get("effective") != declared:
            problems.append(
                f"{cid}: effective gate {evidence.get('effective')} != declared {declared}"
            )
    return problems


def check_cohort(cells: dict) -> list[str]:
    """R4/A3: the filter matches its rule and reconciles to the audited totals."""
    problems = []
    halves = {}
    for cid, info in cells.items():
        declared = info["declared_cohort"]
        gate = info["manifest"]["stages"]["target"]["gate"]
        evidence = gate.get("cohort_filter")
        if declared == "all":
            if evidence is not None:
                problems.append(f"{cid}: declared cohort 'all' but a filter was applied")
            if gate.get("valid") != EXPECTED_VALID:
                problems.append(
                    f"{cid}: full-cohort valid rows {gate.get('valid')} != "
                    f"{EXPECTED_VALID}"
                )
            continue
        if evidence is None:
            problems.append(f"{cid}: declared cohort {declared} but no filter recorded")
            continue
        if evidence["filter"] != declared:
            problems.append(
                f"{cid}: recorded filter {evidence['filter']} != declared {declared}"
            )
        if str(CH_FLOOR) not in evidence.get("rule", ""):
            problems.append(f"{cid}: filter rule does not cite the {CH_FLOOR} floor")
        kept, dropped = evidence["kept"], evidence["dropped"]
        if kept["valid"] + dropped["valid"] != EXPECTED_VALID:
            problems.append(
                f"{cid}: kept+dropped valid {kept['valid']}+{dropped['valid']} != "
                f"{EXPECTED_VALID}"
            )
        if kept["areas_total"] + dropped["areas"] != EXPECTED_AREAS:
            problems.append(
                f"{cid}: kept+dropped areas {kept['areas_total']}+{dropped['areas']} "
                f"!= {EXPECTED_AREAS}"
            )
        halves[declared] = (kept["valid"], kept["areas_total"])

    # The two restricted cohorts must be exact complements of one another.
    if {"non_ch", "ch_only"} <= set(halves):
        nv, na = halves["non_ch"]
        cv, ca = halves["ch_only"]
        if nv + cv != EXPECTED_VALID or na + ca != EXPECTED_AREAS:
            problems.append(
                f"non_ch and ch_only are not complements: {nv}+{cv} rows, {na}+{ca} areas"
            )
    return problems


#: The learner prints one of these per evaluated candidate split, with the parent and
#: candidate class-1 F1 and its verdict. Their margins are what make "the strict gate
#: rejected splits worth rejecting" a measurement rather than an inference.
GATE_LINE = re.compile(
    r"F1 performance gate: parent=([0-9.]+), candidate=([0-9.]+), accepted=(True|False)"
)


def split_evaluation_counts(run: Path) -> dict:
    """R6: how many candidate splits were considered, how many cleared, and by how much.

    Parsed from the retained Stage 1 fitting transcript. The accepted count is
    cross-checked against the manifest's own tally by the caller, so a parsing change
    cannot silently invent numbers.
    """
    out = {
        "considered": None, "cleared": None, "rejected": None, "source": None,
        "rejected_margins": None, "max_rejected_margin": None,
        "min_cleared_margin": None,
    }
    for name in ("georf_fit_stdout.txt.gz", "georf_fit_stdout.txt"):
        path = run / "stage1" / name
        if not path.is_file():
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        text = opener(path, "rt", errors="replace").read()
        rows = [
            (float(parent), float(candidate), verdict == "True")
            for parent, candidate, verdict in GATE_LINE.findall(text)
        ]
        cleared = [c - p for p, c, ok in rows if ok]
        rejected = [c - p for p, c, ok in rows if not ok]
        out.update(
            considered=len(rows), cleared=len(cleared), rejected=len(rejected),
            source=name,
            rejected_margins=[round(m, 6) for m in sorted(rejected, reverse=True)[:10]],
            max_rejected_margin=round(max(rejected), 6) if rejected else None,
            min_cleared_margin=round(min(cleared), 6) if cleared else None,
        )
        break
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="IPCCHGeoRFExperiment/runs")
    parser.add_argument("--out", default=None, help="write the verification JSON here")
    args = parser.parse_args(argv)

    runs = Path(args.runs_dir)
    cells = {}
    missing = []
    for cid, name, cohort, gate in CELLS:
        run = runs / name
        if not (run / "manifest.json").is_file():
            missing.append(f"{cid} ({name})")
            continue
        cells[cid] = {
            "run": name, "declared_cohort": cohort, "declared_gate": gate,
            "manifest": load(run), "splits": split_evaluation_counts(run),
        }
    if missing:
        print(f"FAIL: cells absent from {runs}: {missing}")
        return 1

    checks = {
        "R2_settings_drift": check_settings_drift(cells),
        "R3_gate_readback": check_gate(cells),
        "R4_cohort_reconciliation": check_cohort(cells),
    }
    r6 = []
    for cid, c in cells.items():
        s6 = c["splits"]
        if s6["considered"] is None:
            r6.append(f"{cid}: no fitting transcript, so no evaluation counts")
            continue
        accepted = c["manifest"]["stages"]["stage1_fit"]["learned_map"]["accepted_splits"]
        if s6["cleared"] != accepted:
            r6.append(
                f"{cid}: transcript shows {s6['cleared']} cleared but the manifest "
                f"records {accepted} accepted splits"
            )
    checks["R6_split_evaluation_counts"] = r6

    report = {
        "cells": {
            cid: {
                "run": c["run"], "cohort": c["declared_cohort"], "gate": c["declared_gate"],
                "split_gate_evidence": c["manifest"]["stages"]["stage1_fit"].get("split_gate"),
                "accepted_splits": c["manifest"]["stages"]["stage1_fit"]["learned_map"][
                    "accepted_splits"
                ],
                "split_evaluation": c["splits"],
            }
            for cid, c in cells.items()
        },
        "checks": checks,
        "all_passed": not any(checks.values()),
    }

    width = 74
    print("=" * width)
    print("cross-cell verification".center(width))
    print("=" * width)
    print(f"{'cell':>5} {'cohort':>8} {'gate':>6} {'cleared':>8} {'considered':>11} "
          f"{'max rej':>9}  gate proven")
    print("-" * width)
    for cid, c in report["cells"].items():
        ev = c["split_gate_evidence"] or {}
        proven = "yes (3 ns)" if ev.get("readback") and len(ev["readback"]) == 3 else "NO"
        se = c["split_evaluation"]
        print(
            f"{cid:>5} {c['cohort']:>8} {c['gate']:>6.3f} {c['accepted_splits']:>8} "
            f"{str(se['considered']):>11} {str(se['max_rejected_margin']):>9}  {proven}"
        )
    print("-" * width)
    for name, problems in checks.items():
        print(f"  {name:<28} {'PASS' if not problems else 'FAIL'}")
        for problem in problems[:6]:
            print(f"      - {problem}")
    print("=" * width)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"written: {args.out}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
