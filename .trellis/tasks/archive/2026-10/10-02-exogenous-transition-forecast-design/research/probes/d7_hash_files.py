"""D7 probe: stream SHA256 + size + mtime of raw bytes only (no parsing)."""
import hashlib, os, sys, json, datetime, glob
SRC = "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data"
IPC = f"{SRC}/Outcome/FEWSNET_IPC"
BND = f"{IPC}/FEWS NET Admin Boundaries"
EXPECTED = {
    f"{SRC}/FEWSNET_forecast_unadjusted_bm.csv": "611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651",
    f"{IPC}/FEWSNET.csv": "8fdd4cca6f6ba26b84efc209c8eb36492e1257d51e24edd2c2ad4962df7b38d0",
    f"{SRC}/FEWSNET_admin_code_lat_lon.csv": "a06be85849bb726a4505ed284bed14b100b61f998fb4586a6e14439aca8a4bcb",
    f"{BND}/FEWS_Admin_LZ_v3.shp": "3aba66a6fbf6b2a8beb153df76a67662ce4e0a898fcb11e292a17cc174f5f742",
    f"{SRC}/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025_combined.normalized-v1.csv": "510375f58cd835e694b6e287cce9439bbe1b6246d752daabc8151df8ffdda61d",
}
files = list(EXPECTED) + [
    f"{BND}/FEWS_Admin_LZ_v3.shx", f"{BND}/FEWS_Admin_LZ_v3.dbf", f"{BND}/FEWS_Admin_LZ_v3.prj", f"{BND}/FEWS_Admin_LZ_v3.cpg",
    f"{SRC}/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025.csv",
    f"{SRC}/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025_combined.csv",
    f"{SRC}/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025_combined.normalized-v1.audit.json",
    f"{IPC}/2025_2026_FEWSNET.csv", f"{IPC}/FEWS_2025.csv",
    f"{IPC}/FEWS October 2024 Update TrueBoundaries_12-09-24.csv",
    f"{IPC}/geoidentifier_fews.csv", f"{IPC}/FEWS_scaffold.csv", f"{IPC}/FEWS_scaffold_fixed.csv",
    f"{IPC}/append_2025_2026.ipynb", f"{IPC}/scrape_fewsnet.py",
    f"{IPC}/fewsnet_ipcphase_2025-01_2026-04.geojson",
] + sorted(glob.glob(f"{IPC}/fewsnet_chunks_2025_2026/*"))
out = []
for p in files:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    st = os.stat(p)
    d = h.hexdigest()
    exp = EXPECTED.get(p)
    rec = dict(path=p.replace(SRC, "$SRC"), size=st.st_size,
               mtime=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
               sha256=d, expected=exp, match=(None if exp is None else d == exp))
    out.append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
