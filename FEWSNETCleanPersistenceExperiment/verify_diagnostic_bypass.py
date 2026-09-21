"""Retained A/B control: does bypassing the discarded CV diagnostic change Stage 1?

`GeoRF.fit` unconditionally calls `create_pre_partition_diagnostics_cv`, fits five extra
cross-validation forests and then discards the return value. `run_pipeline` skips that
work by registering an attribute-less stub under `DIAGNOSTIC_MODULE`, so the release
takes its own supported `except ImportError` branch.

That is not obviously inert. The diagnostic has real global side effects:

* `src/diagnostics/pre_partition_diagnostic.py:912` calls ``np.random.seed(final_seed)``,
  resetting NumPy's global RNG.
* `:837,853` draw from ``np.random.choice``, consuming global randomness.

So the two arms genuinely leave different global RNG state behind. This script settles
empirically whether that reaches any scientific output, by running the same
``job.json`` both ways and hashing every artifact.

Run (from the repository root, pinned Windows Python 3.12.10):

    python FEWSNETCleanPersistenceExperiment/verify_diagnostic_bypass.py \
        --job-dir FEWSNETCleanPersistenceExperiment/runs/<run>/stage1/reference/<candidate> \
        --out FEWSNETCleanPersistenceExperiment/runs/<run>/verification/diagnostic_bypass

Evidence is retained under ``--out``: both arms' artifacts, their SHA-256 digests and a
comparison manifest. Note this establishes equivalence for the tested candidate and
runtime, not a proof for every possible input.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_pipeline as rp  # noqa: E402

#: The artifacts that carry scientific meaning. stdout and the worker log are excluded:
#: they legitimately differ, because the diagnostics-on arm prints more.
SCIENTIFIC_ARTIFACTS = (
    "target_predictions.csv",
    "branch_table.npy",
    "s_branch.pkl",
    "correspondence_table.csv",
    "imputer_fill_values.csv",
    "val_coverage_by_group.csv",
)

#: `candidate.json` fields that are expected to differ between the arms.
#: ``diagnostics_bypassed`` is the knob under test. ``n_jobs`` is chosen at runtime by
#: the release's own memory-pressure branch and varies with whatever else the machine is
#: doing, so it is not treated as a scientific difference. This script makes no claim
#: about thread-count invariance: establishing that would need a run in which the arms
#: provably selected different values.
EXPECTED_DIFFERENCES = ("diagnostics_bypassed", "n_jobs")
VOLATILE_TOKENS = (
    "seconds", "started", "finished", "timestamp", "elapsed", "duration",
    "job_dir", "memory", "host", "run_dir", "pythonhashseed",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_volatile(payload):
    if isinstance(payload, dict):
        return {
            key: strip_volatile(value) for key, value in payload.items()
            if not any(token in key.lower() for token in VOLATILE_TOKENS)
        }
    if isinstance(payload, list):
        return [strip_volatile(item) for item in payload]
    return payload


def differences(left, right, path: str = ""):
    if isinstance(left, dict) and isinstance(right, dict):
        out = []
        for key in sorted(set(left) | set(right)):
            out += differences(left.get(key), right.get(key), f"{path}.{key}")
        return out
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [(path, f"length {len(left)} vs {len(right)}")]
        out = []
        for index, (a, b) in enumerate(zip(left, right)):
            out += differences(a, b, f"{path}[{index}]")
        return out
    return [] if left == right else [(path, f"{left!r} vs {right!r}")]


def run_arm(job_dir: Path, out_dir: Path, arm: str) -> Path:
    """Run the worker once with diagnostics stubbed or genuinely enabled."""
    target = out_dir / arm
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy2(job_dir / "job.json", target / "job.json")

    original = rp.DIAGNOSTIC_MODULE
    if arm == "diagnostics_on":
        # Park the stub on a name nothing imports so the real module loads normally.
        rp.DIAGNOSTIC_MODULE = "src.diagnostics.__bypass_control_unused__"
    try:
        rc = rp.run_worker(target)
    finally:
        rp.DIAGNOSTIC_MODULE = original
    if rc != 0:
        raise SystemExit(f"{arm}: worker returned {rc}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True,
                        help="A completed Stage 1 candidate directory (supplies job.json).")
    parser.add_argument("--out", required=True, help="Retained evidence directory.")
    parser.add_argument("--reference", default=None,
                        help="Optional third directory (e.g. the original pilot run) to "
                             "include in the comparison.")
    args = parser.parse_args()

    job_dir = Path(args.job_dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = {arm: run_arm(job_dir, out_dir, arm) for arm in ("stubbed", "diagnostics_on")}
    if args.reference:
        arms["reference_run"] = Path(args.reference).resolve()

    # Prove the diagnostics-on arm really executed the diagnostic.
    executed = {}
    for arm, directory in arms.items():
        stdout = directory / "georf_fit_stdout.txt.gz"
        if not stdout.is_file():
            executed[arm] = None
            continue
        text = gzip.open(stdout, "rt", errors="replace").read()
        # GeoRF.py:336 prints the banner *before* calling the diagnostic and :392-396
        # catches any failure and continues, so the banner alone only proves it started.
        # Completion is the SUCCESS line at :388; a failure warning must disqualify it.
        started = "=== Generating Pre-Partitioning CV Diagnostic Maps ===" in text
        completed = "SUCCESS: Pre-partitioning CV diagnostics completed successfully" in text
        failed = "WARNING: Pre-partitioning CV diagnostics failed" in text
        skipped = "WARNING: Pre-partitioning diagnostic module not available" in text
        executed[arm] = {
            "stdout_chars": len(text),
            "diagnostic_started": started,
            "diagnostic_completed": completed and not failed,
            "diagnostic_failed": failed,
            "diagnostic_module_absent": skipped,
            "pre_partitioning_cv_ran": completed and not failed,
        }

    # Every artifact must be present in every arm. Skipping missing files would let an
    # empty or partial comparison report "all identical", and GeoRF.py:392-396 swallows
    # diagnostic exceptions, so a half-run ON arm is reachable.
    missing = {
        arm: [name for name in SCIENTIFIC_ARTIFACTS if not (directory / name).is_file()]
        for arm, directory in arms.items()
    }
    missing = {arm: names for arm, names in missing.items() if names}

    digests = {
        arm: {name: sha256(directory / name) for name in SCIENTIFIC_ARTIFACTS
              if (directory / name).is_file()}
        for arm, directory in arms.items()
    }
    baseline = digests["stubbed"]
    identical = {
        arm: {name: value == baseline.get(name) for name, value in table.items()}
        for arm, table in digests.items()
    }

    candidates = {
        arm: strip_volatile(json.loads((directory / "candidate.json").read_text()))
        for arm, directory in arms.items()
    }
    json_diffs = {
        arm: [
            {"field": field, "difference": text}
            for field, text in differences(candidates["stubbed"], payload)
        ]
        for arm, payload in candidates.items() if arm != "stubbed"
    }

    all_identical = (
        not missing
        and all(len(table) == len(SCIENTIFIC_ARTIFACTS) for table in digests.values())
        and all(
            table and all(table.values())
            for arm, table in identical.items() if arm != "stubbed"
        )
    )
    unexpected = {
        arm: [entry for entry in entries
              if not any(token in entry["field"] for token in EXPECTED_DIFFERENCES)]
        for arm, entries in json_diffs.items()
    }
    unexpected = {arm: entries for arm, entries in unexpected.items() if entries}

    # The ON arm must have run the diagnostic *to completion* and the OFF arm must have
    # taken the module-absent branch; otherwise the two arms are not the comparison this
    # script claims to make, and a silently failed diagnostic would masquerade as ON.
    on = executed.get("diagnostics_on") or {}
    off = executed.get("stubbed") or {}
    arms_behaved = bool(
        on.get("diagnostic_completed") and not on.get("diagnostic_failed")
        and not off.get("diagnostic_started")
        and off.get("diagnostic_module_absent")
    )
    verified = bool(all_identical and arms_behaved and not unexpected)

    manifest = {
        "question": (
            "Does bypassing the discarded pre-partition CV diagnostic change any "
            "scientific Stage 1 output?"
        ),
        "why_it_is_not_obvious": (
            "pre_partition_diagnostic.py:912 resets NumPy's global RNG with "
            "np.random.seed(final_seed), and :837,853 can draw from np.random.choice on "
            "fallback paths. The two arms can therefore leave different global RNG "
            "state behind, so the discarded return value proves nothing on its own."
        ),
        "job_dir": str(job_dir),
        "runtime": rp.runtime_identity(),
        "diagnostic_module": rp.DIAGNOSTIC_MODULE,
        "arms": {arm: str(directory) for arm, directory in arms.items()},
        "diagnostic_execution": executed,
        "required_artifacts": list(SCIENTIFIC_ARTIFACTS),
        "missing_artifacts": missing,
        "arms_behaved_as_labelled": arms_behaved,
        "artifact_sha256": digests,
        "artifact_identical_to_stubbed": identical,
        "candidate_json_differences_vs_stubbed": json_diffs,
        "unexpected_candidate_json_differences": unexpected,
        "all_scientific_artifacts_identical": all_identical,
        "verified": verified,
        "conclusion": (
            "bypass is inert for this candidate and runtime" if verified
            else "NOT VERIFIED - see missing_artifacts, arms_behaved_as_labelled and "
                 "unexpected_candidate_json_differences"
        ),
        "scope_limit": (
            "Equivalence is demonstrated for the tested candidate, arm and pinned "
            "runtime. It is evidence, not a proof for every possible input."
        ),
    }
    path = out_dir / "diagnostic_bypass_control.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(json.dumps({
        "diagnostic_execution": executed,
        "missing_artifacts": missing,
        "arms_behaved_as_labelled": arms_behaved,
        "all_scientific_artifacts_identical": all_identical,
        "unexpected_candidate_json_differences": unexpected,
        "verified": verified,
        "manifest": str(path),
    }, indent=2))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
