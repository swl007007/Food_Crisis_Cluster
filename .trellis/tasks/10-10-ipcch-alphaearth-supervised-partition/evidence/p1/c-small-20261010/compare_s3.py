"""S3: keyed comparison of batched (batch_size 4) vs single responses for the same 8 pieces; timing from request logs.
Timing is order/cache-conditioned (batch ran first, singles second), not an unbiased speed ratio."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
def load(stage):
    out = {}
    for p in (root / stage / "pieces").glob("*.json"):
        rec = json.loads(p.read_text())
        out[rec["piece_id"]] = rec
    return out
b, s = load("S3_batching"), load("S3_single")
assert set(b) == set(s) and len(b) == 8
rows, n_diff_fields, max_rel = [], 0, 0.0
for pid in sorted(b):
    rb, rs = b[pid]["response"], s[pid]["response"]
    identical = rb == rs
    diffs = {}
    for k in sorted(set(rb["sums"]) | set(rs["sums"])):
        x, y = rb["sums"].get(k), rs["sums"].get(k)
        if x != y:
            rel = abs(x - y) / max(abs(x), abs(y), 1e-300)
            diffs[k] = {"batch": x, "single": y, "rel": rel}
            max_rel = max(max_rel, rel)
    n_diff_fields += len(diffs)
    rows.append({"piece_id": pid, "identical_response": identical, "sums_differing": len(diffs),
                 "count_batch": rb["count"].get("valid_cells"), "count_single": rs["count"].get("valid_cells"),
                 "rank_identical": rb["area_by_rank"] == rs["area_by_rank"], "diffs": diffs,
                 "batch_call_seconds": b[pid]["seconds"], "single_seconds": s[pid]["seconds"]})
def call_seconds(stage):
    log = [json.loads(x) for x in (root / stage / "request_log.jsonl").read_text().splitlines()]
    return [x["seconds"] for x in log if x["status"] == "ok" and not x["purpose"].startswith(("initialize", "projection"))]
bs, ss = call_seconds("S3_batching"), call_seconds("S3_single")
summary = {"pieces": 8, "all_responses_identical": all(r["identical_response"] for r in rows), "differing_sum_fields": n_diff_fields,
           "max_rel_diff": max_rel, "counts_identical": all(r["count_batch"] == r["count_single"] for r in rows),
           "batch_calls": len(bs), "batch_call_seconds": bs, "batch_total_s": round(sum(bs), 2),
           "single_calls": len(ss), "single_call_seconds": ss, "single_total_s": round(sum(ss), 2),
           "timing_note": "batch ran first, singles second (order/cache-conditioned; not an unbiased speed ratio)", "rows": rows}
print(json.dumps(summary, indent=1))
