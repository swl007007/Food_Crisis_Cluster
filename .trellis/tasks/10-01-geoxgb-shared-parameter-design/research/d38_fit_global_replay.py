"""D38 default-path replay: native_xgb.fit_global(X, y, config) and proba(booster, X) without a base
margin, old (HEAD before D38) vs new package. Usage: python d38_fit_global_replay.py PKG OUT_DIR."""
import sys, json, hashlib
from pathlib import Path
pkg, out = Path(sys.argv[1]), Path(sys.argv[2]); sys.path.insert(0, str(pkg)); sys.path.insert(0, str(pkg / "tests"))
import numpy as np
import test_baseline as tb
from src.model import native_xgb as nx
rng, groups, X, y, months, x_set, conf = tb.RecentSearchContrast()._fixture()
booster, record = nx.fit_global(X[x_set == 0], y[x_set == 0], tb.SMALL_G['G1'])
raw = nx.raw(booster)
p = nx.proba(nx.from_raw(raw), X[x_set == 1])
out.mkdir(parents=True, exist_ok=True); (out / "booster.ubj").write_bytes(raw)
np.save(out / "proba.npy", p)
json.dump(record, open(out / "record.json", "w"), indent=1, default=str, sort_keys=True)
print(json.dumps({"package": str(pkg), "booster_sha256": hashlib.sha256(raw).hexdigest(),
                  "proba_sha256": hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest(),
                  "record_keys": sorted(record)}))
