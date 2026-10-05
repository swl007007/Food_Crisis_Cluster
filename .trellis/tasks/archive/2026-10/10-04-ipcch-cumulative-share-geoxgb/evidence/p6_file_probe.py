"""P6 restart probe: no-fit write / directory-rename / read check under runs/.

Mirrors ModelStore._get_or_fit persistence (sibling ``<name>.tmp`` directory,
four .ubj payloads + record.json, ``os.replace`` to the final directory) for a
handful of entries in one pass. Stops at the first failure; no retries.
Run from IPCCHGeoXGBExperiment/ with the pinned runtime:
    python <this file> runs/<probe-id>
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ENTRIES = 8
PAYLOAD_BYTES = 256 * 1024


def main(root: Path) -> int:
    if root.exists():
        print(f"refusing existing probe dir {root}")
        return 2
    models = root / "models"
    models.mkdir(parents=True)
    result = {"probe_dir": str(root), "pid": os.getpid(), "entries": [], "status": "passed"}
    for i in range(ENTRIES):
        directory = models / f"probe_{i:02d}"
        tmp = directory.with_name(directory.name + ".tmp")
        step = "mkdir"
        try:
            tmp.mkdir()
            expected = {}
            step = "write"
            for q in ("q2", "q3", "q4", "q5"):
                data = os.urandom(PAYLOAD_BYTES)
                (tmp / f"{q}.ubj").write_bytes(data)
                expected[f"{q}.ubj"] = hashlib.sha256(data).hexdigest()
            rec = json.dumps({"entry": i, "sha256": expected}, sort_keys=True)
            (tmp / "record.json").write_text(rec, encoding="utf-8")
            expected["record.json"] = hashlib.sha256(rec.encode()).hexdigest()
            step = "rename"
            t0 = time.perf_counter()
            os.replace(tmp, directory)
            rename_s = time.perf_counter() - t0
            step = "read"
            got = {n: hashlib.sha256((directory / n).read_bytes()).hexdigest() for n in expected}
            if got != expected:
                raise RuntimeError("read-back digest mismatch")
            result["entries"].append({"entry": i, "status": "ok", "rename_seconds": round(rename_s, 6)})
        except Exception as exc:  # noqa: BLE001 - record and stop
            result["status"] = "failed"
            result["entries"].append({"entry": i, "status": "failed", "step": step,
                                      "error": f"{type(exc).__name__}: {exc}",
                                      "traceback": traceback.format_exc()})
            break
    (root / "probe-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
