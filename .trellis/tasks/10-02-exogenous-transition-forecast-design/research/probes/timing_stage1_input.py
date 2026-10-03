"""Launch prep TIMING ONLY: cost of Availability.stage1_input on the real pinned panel.
Uses a provisional in-memory ledger rule purely for timing; nothing is written or adopted; no fit."""
import json, sys, time
from pathlib import Path
PKG = Path(__file__).resolve().parents[5] / "FEWSNETGeoXGBExperiment"
sys.path.insert(0, str(PKG))
import pandas as pd
from scripts.prepare_fourclass import load_panel, build_observations, mi
from src.feature import fourclass_features as ff
from src.experiment import availability as av
t0 = time.time()
panel = load_panel(Path(sys.argv[1]))
schema = ff.load_schema(PKG / "feature-schema.json")
obs = build_observations(panel)
scaffold = ff.Scaffold(panel, schema["static_sources"] + schema["dynamic_sources_at_origin"])
cells = obs[["country", "month"]].drop_duplicates()
ledger = pd.DataFrame({"cycle_id": "CS-" + ff.month_label(cells["month"]), "product": "CS",
                       "country": cells["country"].astype(str), "reference_month": ff.month_label(cells["month"]),
                       "release_date": [f"{l}-28" for l in ff.month_label(cells["month"])],
                       "evidence": "reconstructed", "source": "TIMING ONLY provisional"})
alignment = json.loads(Path(sys.argv[2]).read_text())
o = obs[["area", "month", "country", "class_code"]]
ctx = av.Availability(o, av.ReleaseLedger(ledger, real=True), scaffold, schema, 4, alignment, truth=o)
print(f"setup {time.time()-t0:.0f}s areas={o['area'].nunique()}", flush=True)
for strategy, k in (("A", 0), ("B", 2)):
    t = time.time()
    frame = ctx.stage1_input(mi("2018-06"), k, strategy)
    import os, tempfile
    tmp = Path(tempfile.gettempdir()) / f"scratch_timing_{strategy}{k}.parquet"   # scratch only, deleted
    frame.to_parquet(tmp, index=False)
    size = tmp.stat().st_size
    os.remove(tmp)
    print(f"stage1_input {strategy} k{k}: rows={len(frame)} cols={frame.shape[1]} roles={frame['role'].value_counts().to_dict()} "
          f"{time.time()-t:.0f}s parquet={size/1e6:.1f}MB", flush=True)
