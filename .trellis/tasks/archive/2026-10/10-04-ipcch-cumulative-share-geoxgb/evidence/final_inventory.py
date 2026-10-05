"""P6 post-replay final inventory: run-relative path, size and sha256 of every run file.

Run from IPCCHGeoXGBExperiment/:  python <this file> runs/<run-id> <out.csv>
Writes outside the run directory (no self-reference).
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(run: Path, out: Path) -> None:
    files = sorted((p for p in run.rglob("*") if p.is_file()), key=lambda p: p.relative_to(run).as_posix())
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["path", "bytes", "sha256"])
        for p in files:
            w.writerow([p.relative_to(run).as_posix(), p.stat().st_size, sha(p)])
    print(len(files), sum(p.stat().st_size for p in files))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
