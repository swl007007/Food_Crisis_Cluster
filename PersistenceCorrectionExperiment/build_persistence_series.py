#!/usr/bin/env python3
"""Phase 1 entrypoint: build and validate the layer-1 persistence series.

Run from the repository root::

    PYTHONPATH="$PWD/PersistenceCorrectionExperiment" python3 \
        PersistenceCorrectionExperiment/build_persistence_series.py --run-id phase1

Writes ``persistence_series_fs{1,2}.csv``, ``persistence_run_manifest.json`` and a
before/after protected-hash report under
``PersistenceCorrectionExperiment/outputs/<run-id>/``.  Run IDs are immutable: an
existing run directory raises ``FileExistsError`` (PRD R26).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from persistencecorrection import persistence, protected  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="immutable output directory name")
    parser.add_argument(
        "--scopes", nargs="+", type=int, default=[1, 2], help="forecasting scopes (fs1/fs2)"
    )
    args = parser.parse_args(argv)

    # Refuses any path outside PersistenceCorrectionExperiment/outputs/.
    output_dir = protected.resolve_output_path(args.run_id)
    if output_dir.exists():
        raise FileExistsError(f"Run directory already exists (run IDs are immutable): {output_dir}")

    before = protected.hash_protected()
    result = persistence.build_persistence_series(output_dir, args.scopes)
    after = protected.hash_protected()
    protected.assert_unchanged(before)
    from step3correction.protected import write_hash_report  # noqa: E402

    write_hash_report(output_dir / "protected_hashes.json", before, after)

    failures = []
    for scope, payload in sorted(result["manifest"]["scopes"].items()):
        coverage = payload["coverage"]
        check = payload["reference_check"]
        print(
            f"fs{scope}: rows={coverage['rows']} coverage={coverage['coverage']:.4f} "
            f"f1(1)={check['computed_f1_class1']:.6f} "
            f"reference={check['reference_f1_class1']:.4f} "
            f"|delta|={check['abs_delta']:.2e} "
            f"reproduced={check['reproduced']}"
        )
        if coverage["coverage"] != 1.0 or not check["reproduced"]:
            failures.append(scope)

    print(f"protected inputs hashed: {len(before)}; unchanged: {before == after}")
    if failures:
        print(f"PHASE 1 GATE FAILED for scopes {failures}", file=sys.stderr)
        return 1
    print("PHASE 1 GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
