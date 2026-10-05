"""P6: reconcile the actual Stage3 request ledger with the pre-predict enumeration.

Run from IPCCHGeoXGBExperiment/:
    python <this file> runs/<run-id> <enumeration.json>
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def main(run: Path, enum_path: Path) -> dict:
    enum = json.loads(enum_path.read_text())
    led = [json.loads(l) for l in (run / "stage3" / "model_requests.jsonl").read_text().splitlines()]
    out, ok = {"horizons": {}}, True
    tot = collections.Counter()
    for h, e in enum["horizons"].items():
        rows = [r for r in led if str(r["H"]) == h]
        by = collections.Counter((r["purpose"], r["use"]) for r in rows)
        st = collections.Counter(r["status"] for r in rows)
        dg = {r["identity_sha256"] for r in rows if r["purpose"] == "global"}
        dl_hist = {r["identity_sha256"] for r in rows if r["purpose"] == "local" and r["use"] == "gate"}
        dl_cur = {r["identity_sha256"] for r in rows if r["purpose"] == "local" and r["use"] == "current"}
        q = e["requests_quartets"]
        actual = {
            "current_global": by[("global", "current")], "historical_global": by[("global", "gate")],
            "historical_local": by[("local", "gate")], "current_local": by[("local", "current")],
            "distinct_global": len(dg), "distinct_local_hist": len(dl_hist),
            "distinct_local_current_only": len(dl_cur - dl_hist),
            "fits": st["fit"], "hits": st["hit"], "failed": st.get("failed", 0),
        }
        actual["distinct_scalar_fits"] = 4 * (actual["distinct_global"] + actual["distinct_local_hist"]
                                              + actual["distinct_local_current_only"])
        checks = {
            "current_global_exact": actual["current_global"] == q["current_global"],
            "historical_global_exact": actual["historical_global"] == q["historical_global"],
            "historical_local_exact": actual["historical_local"] == q["historical_local_exact"],
            "current_local_within_upper": actual["current_local"] <= q["current_local_upper"],
            "distinct_global_exact": actual["distinct_global"] == e["distinct_global_quartets_exact"],
            "distinct_local_hist_exact": actual["distinct_local_hist"] == e["distinct_local_quartets_exact"],
            "current_only_within_upper": actual["distinct_local_current_only"]
            <= e["distinct_current_local_quartets_upper_not_shared"],
            "fits_equal_distinct": actual["fits"] * 4 == actual["distinct_scalar_fits"],
            "no_failed": actual["failed"] == 0,
        }
        ok &= all(checks.values())
        out["horizons"][h] = {"actual": actual, "checks": checks}
        tot.update({k: v for k, v in actual.items()})
    out["totals"] = dict(tot)
    out["totals"]["enumerated_scalar_fits_exact_upper"] = [enum["totals"]["distinct_scalar_fits_exact"],
                                                           enum["totals"]["distinct_scalar_fits_upper"]]
    out["status"] = "passed" if ok else "failed"
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]), Path(sys.argv[2])), indent=1))
