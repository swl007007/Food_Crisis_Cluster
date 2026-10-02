"""Compare old vs new hard-path replay outputs (stdlib). Numerical outputs/routes/correspondence/checkpoints
must be byte-equal; gzip CSVs compared decompressed; candidate.json/record compared after removing ONLY the
declared new metadata (e1, decisions[].scan_diagnostics), timings and the temporary checkpoint dir.
Usage: python3 d34_hard_replay_compare.py <old_out> <new_out> <result.json>"""
import gzip, json, os, sys
o, n, dest = sys.argv[1:4]
DECLARED = {"top": ["e1", "timings"], "decision": ["scan_diagnostics"], "checkpoints": ["dir"]}
def strip(j):
    j = {k: v for k, v in j.items() if k not in DECLARED["top"]}
    if "checkpoints" in j: j["checkpoints"] = {k: v for k, v in j["checkpoints"].items() if k not in DECLARED["checkpoints"]}
    p = dict(j.get("partition", {}))
    p["decisions"] = [{k: v for k, v in d.items() if k not in DECLARED["decision"]} for d in p.get("decisions", [])]
    j["partition"] = p
    return j
res = {"declared_exclusions": DECLARED, "files": {}}
for sub in ("candidate", "checkpoints"):
    A, B = set(os.listdir(f"{o}/{sub}")), set(os.listdir(f"{n}/{sub}"))
    res[f"{sub}_only_old"], res[f"{sub}_only_new"] = sorted(A - B), sorted(B - A)
    for f in sorted(A & B):
        a, b = open(f"{o}/{sub}/{f}", "rb").read(), open(f"{n}/{sub}/{f}", "rb").read()
        if f == "candidate.json":
            r = "equal_after_declared_exclusions" if strip(json.loads(a)) == strip(json.loads(b)) else "DIFF"
        elif f.endswith(".gz"):
            r = "raw_bytes_equal" if a == b else ("decompressed_equal" if gzip.decompress(a) == gzip.decompress(b) else "DIFF")
        else:
            r = "bytes_equal" if a == b else "DIFF"
        res["files"][f"{sub}/{f}"] = r
ra, rb = json.load(open(f"{o}/record.json")), json.load(open(f"{n}/record.json"))
res["returned_record"] = "equal_after_declared_exclusions" if strip(ra) == strip(rb) else "DIFF"
res["passed"] = all(v != "DIFF" for v in res["files"].values()) and res["returned_record"] != "DIFF" \
    and not any(res[k] for k in ("candidate_only_old", "candidate_only_new", "checkpoints_only_old", "checkpoints_only_new"))
json.dump(res, open(dest, "w"), indent=1)
print(json.dumps(res, indent=1))
