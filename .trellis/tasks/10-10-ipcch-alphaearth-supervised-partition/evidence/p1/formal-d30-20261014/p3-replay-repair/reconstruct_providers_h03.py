"""Zero-fit staged reconstruction of predict/providers_all_h03.csv.gz (read-only on the run; writes ONLY to the staged
path given). Same assembly order as pipeline: historical blocks in schedule order, then bridge, then main/supplement by
fit_origin; each block = rows with target in [target_lo, target_hi], B1 provider = stored booster of the ledger identity."""
import gzip, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from ipcch_alphaearth import backbone, schedule
from ipcch_alphaearth.artifacts import write_csv_gz
from ipcch_alphaearth.contract import load_contract
from ipcch_alphaearth.pipeline import TARGETS
from ipcch_alphaearth.runinputs import load_run_inputs
run, staged = Path(sys.argv[1]), Path(sys.argv[2])
c = load_contract(); inp = load_run_inputs(run, c); h = 3; hi = inp.horizons[h]
store = backbone.ModelStore(run / "models", read_only=True)
led = pd.concat([pd.read_csv(run / s / "request_ledger.csv.gz") for s in ("develop", "predict")], ignore_index=True)
ident = {(r.arm, r.block_id): r.identity for r in led[led.horizon == h].itertuples(index=False)}
blocks = schedule.horizon_blocks(c, h)
order = [b for b in blocks if b.role == "historical"] + [b for b in blocks if b.role == "bridge"] + \
        sorted([b for b in blocks if b.role in ("main", "supplement")], key=lambda x: x.fit_origin)
parts = []
for b in order:
    bst, rec = store.load_by_digest(ident[("B1", b.block_id)])
    rows = np.flatnonzero((hi.target >= b.target_lo) & (hi.target <= b.target_hi))
    raw = backbone.predict_quartet(bst, hi.X("B1")[rows], int(c["backbone"]["fixed"]["n_jobs"]))
    parts.append(pd.DataFrame({"row": rows, "target_ord": hi.target[rows], "provider_block": b.block_id, "provider_fit_origin": b.fit_origin,
                               "provider_identity": ident[("B1", b.block_id)], **{f"p_{q}": raw[:, k] for k, q in enumerate(TARGETS)}}))
write_csv_gz(staged, pd.concat(parts, ignore_index=True))
a = gzip.open(run / "predict" / f"providers_all_h{h:02d}.csv.gz", "rt").read().split("\n")
b = gzip.open(staged, "rt").read().split("\n")
diff = [i for i in range(max(len(a), len(b))) if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None)]
cells = []
for i in diff:
    fa, fb = a[i].split(","), b[i].split(",")
    cells += [{"line": i + 1, "column": a[0].split(",")[j], "original": fa[j], "reconstructed": fb[j]} for j in range(len(fa)) if fa[j] != fb[j]]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
print(json.dumps({"original_sha256": sha(run / "predict/providers_all_h03.csv.gz"), "staged_sha256": sha(staged),
                  "original_lines": len(a), "staged_lines": len(b), "differing_lines": [i + 1 for i in diff], "differing_cells": cells,
                  "staged_text_sha256": hashlib.sha256("\n".join(b).encode()).hexdigest(),
                  "original_text_sha256": hashlib.sha256("\n".join(a).encode()).hexdigest()}, indent=1))
