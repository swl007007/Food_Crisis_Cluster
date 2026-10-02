"""D34 legacy hard-path replay runner: one production fixture through run_candidate (default E1).
Usage: python d34_hard_replay_run.py <package root> <out dir>   (run once with the pinned old package, once with new)."""
import sys, os, shutil, tempfile, json
from pathlib import Path
pkg, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(pkg)); sys.path.insert(0, str(pkg / "tests")); os.chdir(pkg)
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
import numpy as np
import test_baseline as tb
from app import main_model_GF as mgf
from src.model import native_xgb as nx
from src.metrics import fourclass
from src.partition import transformation as trans
from src.experiment import plan
rng, groups, X, y, months, x_set, conf = tb.RecentSearchContrast()._fixture()
root = nx.fit_global(X[x_set == 0], y[x_set == 0], tb.SMALL_G['G1'])
first = np.unique(groups, return_index=True)[1]
Xt, yt, gt = X[first], y[first], groups[first]
y_pool = fourclass.argmax_codes(nx.proba(root[0], Xt))
keep = ~conf
data = (X[keep], y[keep], groups[keep], months[keep], x_set[keep], Xt, yt, gt, y_pool)
with tempfile.TemporaryDirectory() as t, patch.object(trans, 'CONTIGUITY', False), \
        patch.object(trans, 'generate_count_grid', return_value=(None, 0, 1)), \
        patch.dict(plan.FIT_SUPPORT, tb.FLOORS['fit_support']), patch.dict(plan.STAGE1_VAL_SUPPORT, tb.FLOORS['val_support']), \
        patch.object(mgf, 'MAX_DEPTH', 3), redirect_stdout(StringIO()):
    work, ck = Path(t) / 'w', Path(t) / 'ck'; work.mkdir()
    rec = mgf.run_candidate('c', 'L1', 'gt0', root, data, work, ck, None, [f'f{i}' for i in range(5)],
                            increment_source='root', confirmation=(X[conf], y[conf], groups[conf], months[conf]))
    shutil.copytree(work / 'c', out / 'candidate'); shutil.copytree(ck / 'c', out / 'checkpoints')
json.dump(rec, open(out / 'record.json', 'w'), indent=1, default=str, sort_keys=True)
print(json.dumps({"package": str(pkg), "files": sorted(p.name for p in (out / 'candidate').iterdir()),
                  "decisions": len(rec['partition']['decisions']), "n_terminal": rec['partition']['n_terminal']}))
