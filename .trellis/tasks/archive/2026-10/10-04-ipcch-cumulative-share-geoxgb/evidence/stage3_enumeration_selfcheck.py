"""Cross-check stage3_enumeration.py against production Stage3 on a synthetic run.

Builds the adopted-local synthetic world from tests/test_p5_e2e.py (west_areas=56)
in a temp dir, runs production learn-map + predict, then compares the
enumerator's counts with the production Stage3 request ledger. Exact categories
must match exactly; current locals must be <= the enumerated upper bound.
Run from IPCCHGeoXGBExperiment/:  PYTHONPATH=.;tests python <this file>
"""

from __future__ import annotations

import collections
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import test_p5_e2e as p5
from ipcch_geoxgb import learnmap, predict

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("enum", here / "stage3_enumeration.py")
enum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enum)

for module in (learnmap, predict, enum):
    module.load_experiment_contract = p5._contract

with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp) / "run"
    p5._write_prepared(run, west_areas=56)
    learnmap.run_learn_map(run)
    predict.run_predict(run)
    # the synthetic contract has a single H; restrict the enumerator to it
    contract = p5._contract()
    led = [json.loads(l) for l in (run / "stage3" / "model_requests.jsonl").read_text().splitlines()]
    out = enum.enumerate_run(run)
    e = out["horizons"][str(p5.H)]
    by = collections.Counter((r["purpose"], r["use"]) for r in led)
    prod = {
        "current_global": by[("global", "current")],
        "historical_global": by[("global", "gate")],
        "historical_local": by[("local", "gate")],
        "current_local": by[("local", "current")],
        "distinct_global": len({r["identity_sha256"] for r in led if r["purpose"] == "global"}),
        "distinct_local": len({r["identity_sha256"] for r in led if r["purpose"] == "local"}),
        "statuses": dict(collections.Counter(r["status"] for r in led)),
    }
    hist_local_ids = {r["identity_sha256"] for r in led if r["purpose"] == "local" and r["use"] == "gate"}
    cur_local_ids = {r["identity_sha256"] for r in led if r["purpose"] == "local" and r["use"] == "current"}
    q = e["requests_quartets"]
    checks = {
        "current_global_exact": q["current_global"] == prod["current_global"],
        "historical_global_exact": q["historical_global"] == prod["historical_global"],
        "historical_local_exact": q["historical_local_exact"] == prod["historical_local"],
        "current_local_within_upper": prod["current_local"] <= q["current_local_upper"],
        "distinct_global_exact": e["distinct_global_quartets_exact"] == prod["distinct_global"],
        "distinct_hist_local_exact": e["distinct_local_quartets_exact"] == len(hist_local_ids),
        "distinct_local_within_upper": prod["distinct_local"]
        <= e["distinct_local_quartets_exact"] + e["distinct_current_local_quartets_upper_not_shared"],
    }
    result = {"enumerated": e, "production": prod,
              "production_current_local_distinct": len(cur_local_ids),
              "checks": checks, "status": "passed" if all(checks.values()) else "failed"}
    print(json.dumps(result, indent=1))
    sys.exit(0 if result["status"] == "passed" else 1)
