"""D7 probe: admin-key / date-key metadata only.
Protection: usecols whitelist only key/date/name/scenario metadata. No IPC value,
description, phase, HA or projection-value column is ever loaded.
"""
import struct, sys, json, collections
import pandas as pd
S = "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data"
I = S + "/Outcome/FEWSNET_IPC"
FORBIDDEN = ("ipc", "value", "description", "phase", "class_code", "fews_ha", "proj", "assistance")
def rd(path, cols, **kw):
    for c in [x for x in (cols or []) if x not in ("projection_start", "projection_end")]:  # date metadata only
        assert not any(t in c.lower() for t in FORBIDDEN), c
    return pd.read_csv(path, usecols=cols, dtype=str, encoding="utf-8-sig", **kw)
def code(s):
    return pd.to_numeric(s, errors="coerce").astype("Int64")
def ex(s, k=8):
    return sorted(list(s), key=str)[:k]
out = collections.OrderedDict()
def say(*a):
    print(*a, flush=True)

# ---------------- load (key columns only) ----------------
P = rd(S + "/FEWSNET_forecast_unadjusted_bm.csv", ["unit_name", "ADMIN0", "FEWSNET_admin_code", "ISO", "date", "month"])
P["id"] = code(P.FEWSNET_admin_code); P["ym"] = P.date.str[:7]
H = rd(I + "/FEWSNET.csv", ["country", "admin_code", "year_month", "year", "month", "admin_name"])
H["id"] = code(H.admin_code); H["ym"] = H.year + "-" + H.month.str.zfill(2)
A = rd(S + "/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025.csv", ["unit_name", "ADMIN0", "admin_code", "ISO", "year", "month"])
A["id"] = code(A.admin_code); A["ym"] = A.year + "-" + A.month.str.zfill(2)
C = rd(S + "/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025_combined.csv", ["unit_name", "ADMIN0", "FEWSNET_admin_code", "ISO", "date", "month"])
C["id"] = code(C.FEWSNET_admin_code); C["ym"] = C.date.str[:7]
N = rd(S + "/assembled_FEWSNET/FEWSNET_forecast_unadjusted_bm_2025_combined.normalized-v1.csv", ["unit_name", "FEWSNET_admin_code", "ISO", "date", "month"])
N["id"] = code(N.FEWSNET_admin_code); N["ym"] = N.date.str[:7]
R = rd(I + "/2025_2026_FEWSNET.csv", ["country", "country_code", "geographic_group", "fewsnet_region", "fnid", "geographic_unit_name",
        "geographic_unit_full_name", "classification_scale", "scenario", "scenario_name", "reporting_date", "projection_start",
        "projection_end", "collection_status", "status", "id", "dataseries_name"])
F = rd(I + "/FEWS_2025.csv", ["admin_code", "year", "month"])
F["id"] = code(F.admin_code)
F["ym"] = (pd.to_numeric(F.year, errors="coerce").astype("Int64").astype(str) + "-" +
           pd.to_numeric(F.month, errors="coerce").astype("Int64").astype(str).str.zfill(2))
F.loc[F.year.isna() | F.month.isna(), "ym"] = "<NA>"
K = rd(S + "/FEWSNET_admin_code_lat_lon.csv", None); K["id"] = code(K.FEWSNET_admin_code)
G = rd(I + "/geoidentifier_fews.csv", ["unit_name", "ADMIN0", "admin_code", "ISO", "lat", "lon"]); G["id"] = code(G.admin_code)
SC = rd(I + "/FEWS_scaffold.csv", ["admin_code", "year", "month"]); SC["id"] = code(SC.admin_code); SC["ym"] = SC.year + "-" + SC.month.str.zfill(2)
SF = rd(I + "/FEWS_scaffold_fixed.csv", ["unit_name", "admin_code", "ISO", "lat", "lon"]); SF["id"] = code(SF.admin_code)

# DBF attributes (no IPC fields exist in this schema; read selected fields)
def read_dbf(path, want):
    f = open(path, "rb"); h = f.read(32)
    n, hl, rl = struct.unpack("<IHH", h[4:12]); fields = []; off = 1
    while True:
        b = f.read(32)
        if b[0] == 0x0D: break
        nm = b[:11].split(b"\0")[0].decode(); ln = b[16]; fields.append((nm, off, ln)); off += ln
    f.seek(hl); rows = []
    for _ in range(n):
        rec = f.read(rl)
        rows.append({nm: rec[o:o + ln].decode("utf-8", "replace").strip() for nm, o, ln in fields if nm in want})
    return pd.DataFrame(rows)
D = read_dbf(I + "/FEWS NET Admin Boundaries/FEWS_Admin_LZ_v3.dbf",
             {"cov_start", "cov_end", "report_mon", "unit_name", "ADMIN0", "LZCODE", "admin_code", "admin_name", "ISO", "adm0_name"})
D["id"] = code(D.admin_code)

# ---------------- per-file key structure ----------------
def keyinfo(name, df, has_ym=True):
    r = dict(rows=len(df), null_id=int(df.id.isna().sum()), distinct_id=int(df.id.nunique()),
             id_min=None if df.id.isna().all() else int(df.id.min()), id_max=None if df.id.isna().all() else int(df.id.max()))
    if has_ym:
        k = df[["id", "ym"]]
        r["dup_key_rows"] = int(k.duplicated(keep=False).sum())
        r["distinct_key"] = int(len(k.drop_duplicates()))
        vc = df.ym.value_counts().sort_index()
        r["n_months"] = int(len(vc)); r["first_ym"] = vc.index[0]; r["last_ym"] = vc.index[-1]
        r["rows_per_month_distinct"] = sorted(set(int(x) for x in vc.values))[:10]
        r["months"] = {k_: int(v) for k_, v in vc.items()}
    else:
        r["dup_id_rows"] = int(df.id.duplicated(keep=False).sum())
    out[name] = r
    say(name, {k_: v for k_, v in r.items() if k_ != "months"})
for nm, df in [("pinned_panel", P), ("FEWSNET.csv", H), ("panel_2025", A), ("panel_2025_combined", C),
               ("panel_2025_combined_normv1", N), ("FEWS_2025.csv", F), ("FEWS_scaffold", SC)]:
    keyinfo(nm, df)
for nm, df in [("coords", K), ("geoidentifier", G), ("scaffold_fixed", SF), ("shapefile_dbf", D)]:
    keyinfo(nm, df, has_ym=False)

# months listing for small/interesting files
for nm in ["FEWSNET.csv", "FEWS_2025.csv", "panel_2025"]:
    say("MONTHS", nm, out[nm]["months"])
for nm in ["pinned_panel", "panel_2025_combined", "panel_2025_combined_normv1", "FEWS_scaffold"]:
    m = out[nm]["months"]; say("MONTHS", nm, "n=", len(m), "first", min(m), "last", max(m), "counts", sorted(set(m.values())))
    # gaps
    allm = pd.period_range(min(m), max(m), freq="M").astype(str)
    say("  missing months:", [x for x in allm if x not in m])
fm = out["FEWSNET.csv"]["months"]; say("FEWSNET.csv distinct months", len(fm))

# ---------------- id set comparisons ----------------
sets = {"pinned_panel": set(P.id.dropna()), "FEWSNET.csv": set(H.id.dropna()), "panel_2025": set(A.id.dropna()),
        "panel_2025_combined": set(C.id.dropna()), "normv1": set(N.id.dropna()), "FEWS_2025.csv": set(F.id.dropna()),
        "coords": set(K.id.dropna()), "geoidentifier": set(G.id.dropna()), "scaffold": set(SC.id.dropna()),
        "scaffold_fixed": set(SF.id.dropna()), "shapefile_dbf": set(D.id.dropna())}
cmp = {}
for ref in ["pinned_panel", "FEWSNET.csv"]:
    for k, s in sets.items():
        r = sets[ref]
        cmp[f"{k}_vs_{ref}"] = dict(n=len(s), inter=len(s & r), only_this=len(s - r), only_ref=len(r - s),
                                   ex_only_this=[int(x) for x in ex(s - r)], ex_only_ref=[int(x) for x in ex(r - s)])
        say("SET", k, "vs", ref, cmp[f"{k}_vs_{ref}"])
out["set_cmp"] = cmp

# ---------------- code-space semantics: same integer, same unit? ----------------
pu = P.drop_duplicates("id").set_index("id")[["unit_name", "ISO"]]
hu = H.drop_duplicates(["id", "admin_name"])[["id", "admin_name", "country"]]
du = D.set_index("id")[["unit_name", "admin_name", "ISO", "LZCODE", "cov_start", "cov_end", "report_mon"]]
gu = G.set_index("id")[["unit_name"]]
say("H: admin_codes with >1 admin_name:", int((hu.groupby("id").admin_name.nunique() > 1).sum()),
    "; admin_names with >1 admin_code:", int((hu.groupby("admin_name").id.nunique() > 1).sum()),
    "; distinct names:", hu.admin_name.nunique())
say("H country count", H.country.nunique())
# panel unit_name vs dbf unit_name by code
j = pu.join(du, rsuffix="_dbf", how="inner")
say("panel∩dbf codes", len(j), "unit_name equal:", int((j.unit_name == j.unit_name_dbf).sum()))
say("examples panel vs dbf:", j.head(3)[["unit_name", "unit_name_dbf", "admin_name"]].to_dict("records"))
# FEWSNET.csv admin_name vs dbf admin_name by code
h1 = hu.drop_duplicates("id").set_index("id")
j2 = h1.join(du, rsuffix="_dbf", how="inner")
say("H∩dbf codes", len(j2), "admin_name equal:", int((j2.admin_name == j2.admin_name_dbf).sum()))
say("examples H vs dbf:", j2.head(3)[["admin_name", "admin_name_dbf", "unit_name"]].to_dict("records"))
# FEWSNET.csv admin_name vs panel unit_name by code (does admin_name end with unit_name?)
j3 = h1.join(pu, how="inner", rsuffix="_p")
say("H∩panel codes", len(j3), "H.admin_name endswith panel.unit_name:",
    int(sum(str(a).endswith(str(b)) for a, b in zip(j3.admin_name, j3.unit_name))))
say("examples H vs panel same code:", j3.head(3)[["admin_name", "unit_name"]].to_dict("records"))
# dbf duplicates
say("dbf admin_code dup:", int(D.id.duplicated().sum()), "dbf unit_name dup rows:", int(D.unit_name.duplicated(keep=False).sum()),
    "dbf admin_name dup rows:", int(D.admin_name.duplicated(keep=False).sum()), "dbf LZCODE examples:", D.LZCODE.head(3).tolist())
say("dbf cov_start/cov_end/report_mon distinct:", D.cov_start.value_counts().head(5).to_dict(), D.cov_end.value_counts().head(5).to_dict(),
    D.report_mon.value_counts().head(5).to_dict())
# geoidentifier / coords / scaffold_fixed consistency
gg = G.set_index("id"); kk = K.set_index("id"); ss = SF.set_index("id")
jj = gg.join(kk, rsuffix="_k", how="inner")
say("geoid∩coords", len(jj), "max|dlat|", float((jj.lat.astype(float) - jj.lat_k.astype(float)).abs().max()),
    "max|dlon|", float((jj.lon.astype(float) - jj.lon_k.astype(float)).abs().max()))
js = gg.join(ss, rsuffix="_s", how="inner")
say("geoid∩scaffold_fixed", len(js), "unit_name equal", int((js.unit_name == js.unit_name_s).sum()))
jg = gu.join(pu, rsuffix="_p", how="inner"); say("geoid∩panel unit_name equal", int((jg.unit_name == jg.unit_name_p).sum()), "of", len(jg))
jd = gu.join(du, rsuffix="_d", how="inner"); say("geoid∩dbf unit_name equal", int((jd.unit_name == jd.unit_name_d).sum()), "of", len(jd))
# panel unit_name uniqueness
say("panel unit_name per id unique:", int((P.groupby("id").unit_name.nunique() > 1).sum()), "ids with >1 name;",
    int((P.drop_duplicates(["id", "unit_name"]).groupby("unit_name").id.nunique() > 1).sum()), "names with >1 id")

# ---------------- 2025_2026 raw file: fnid metadata ----------------
say("RAW rows", len(R), "distinct fnid", R.fnid.nunique(), "null fnid", int(R.fnid.isna().sum()))
say("RAW scenario x reporting_date:", R.groupby(["scenario", "scenario_name", "reporting_date"]).size().to_dict())
say("RAW scenario x projection_start/end:", R.groupby(["scenario", "projection_start", "projection_end"]).size().to_dict())
say("RAW collection_status/status:", R.groupby(["collection_status", "status"]).size().to_dict())
say("RAW classification_scale:", R.classification_scale.value_counts().to_dict())
say("RAW key dup (fnid, scenario, reporting_date, projection_start):",
    int(R.duplicated(["fnid", "scenario", "reporting_date", "projection_start"], keep=False).sum()))
say("RAW fnid -> full_name >1:", int((R.groupby("fnid").geographic_unit_full_name.nunique() > 1).sum()),
    "; full_name -> fnid >1:", int((R.groupby("geographic_unit_full_name").fnid.nunique() > 1).sum()))
fn_multi = R.groupby("geographic_unit_full_name").fnid.unique()
fn_multi = fn_multi[fn_multi.apply(len) > 1]
say("examples full_name with >1 fnid:", {k: list(v) for k, v in fn_multi.head(5).items()})
R["fn_vintage"] = R.fnid.str[2:6]
say("fnid boundary-vintage (chars 3-6) by country_code:", R.drop_duplicates("fnid").groupby(["country_code", "fn_vintage"]).size().to_dict())
say("RAW dataseries 'From' examples:", R.dataseries_name.str.extract(r"\(From ([^)]*)\)")[0].value_counts().head(10).to_dict())
say("RAW country counts (distinct fnid):", R.drop_duplicates("fnid").country.value_counts().to_dict())
# direct fnid -> admin code? (string match to any numeric code impossible; check dbf LZCODE)
say("fnid ∩ dbf LZCODE:", len(set(R.fnid) & set(D.LZCODE)))
# name join used by append_2025_2026.ipynb: geographic_unit_full_name == FEWSNET.csv admin_name
CS = R[R.scenario_name == "Current Situation"]
names_cs = set(CS.geographic_unit_full_name); names_h = set(hu.admin_name); names_d = set(D.admin_name)
say("CS rows", len(CS), "CS distinct full_names", len(names_cs), "matched to FEWSNET.csv admin_name", len(names_cs & names_h),
    "unmatched", len(names_cs - names_h), "; H names not in CS", len(names_h - names_cs))
say("CS full_name matched to dbf admin_name", len(names_cs & names_d), "; to panel unit_name", len(names_cs & set(pu.unit_name)))
say("examples CS-unmatched:", ex(names_cs - names_h, 6))
say("examples H-not-in-CS:", ex(names_h - names_cs, 6))
# per reporting_date
for rdte, g in CS.groupby("reporting_date"):
    nm = set(g.geographic_unit_full_name)
    dupn = int(g.duplicated(["geographic_unit_full_name"], keep=False).sum())
    say(f"CS {rdte}: rows {len(g)}, distinct names {len(nm)}, matched {len(nm & names_h)}, rows sharing a name {dupn}")
# simulate notebook join cardinality (keys only)
prev = H[["admin_name", "id"]].drop_duplicates()
newk = CS[["geographic_unit_full_name", "reporting_date"]].rename(columns={"geographic_unit_full_name": "admin_name"})
m2 = prev.merge(newk, on="admin_name", how="left")
say("notebook-style left join rows (prev->new):", len(m2), "rows with NA date:", int(m2.reporting_date.isna().sum()),
    "dup (id, reporting_date):", int(m2.dropna(subset=["reporting_date"]).duplicated(["id", "reporting_date"], keep=False).sum()))
say("prev (admin_name, admin_code) pairs:", len(prev))
# per-country coverage of the name join
cs_c = CS.drop_duplicates("geographic_unit_full_name")[["country", "geographic_unit_full_name"]]
cs_c["matched"] = cs_c.geographic_unit_full_name.isin(names_h)
say("per-country CS name match (matched/unmatched):", cs_c.groupby("country").matched.agg(["sum", "count"]).assign(un=lambda d: d["count"] - d["sum"]).to_dict("index"))
# FEWS_2025 vs FEWSNET.csv codes per country
hc = H.drop_duplicates("id")[["id", "country"]]
fdated = F[F.ym != "<NA>"]
hc["in_F_dated"] = hc.id.isin(set(fdated.id.dropna()))
say("per-country FEWSNET.csv codes present in dated FEWS_2025 rows:", hc.groupby("country").in_F_dated.agg(["sum", "count"]).to_dict("index"))
# FEWS_2025 vs pinned panel code space
say("FEWS_2025 dated distinct ids", fdated.id.nunique(), "∩ pinned", len(set(fdated.id.dropna()) & sets["pinned_panel"]))

# ---------------- pinned vs 2025-combined panel key agreement ----------------
pk = P[["id", "ym"]]; ck = C[["id", "ym"]]; nk = N[["id", "ym"]]; ak = A[["id", "ym"]]
ov = sorted(set(pk.ym) & set(ck.ym))
say("pinned∩combined months", len(ov), ov[0] if ov else None, ov[-1] if ov else None)
pks = set(map(tuple, pk.drop_duplicates().itertuples(index=False))); cks = set(map(tuple, ck.drop_duplicates().itertuples(index=False)))
nks = set(map(tuple, nk.drop_duplicates().itertuples(index=False))); aks = set(map(tuple, ak.drop_duplicates().itertuples(index=False)))
pko = {k for k in pks if k[1] in ov}; cko = {k for k in cks if k[1] in ov}
say("overlap keys pinned", len(pko), "combined", len(cko), "pinned-only", len(pko - cko), "combined-only", len(cko - pko),
    "ex", ex(pko - cko, 4), ex(cko - pko, 4))
say("combined vs normv1 keys: combined-only", len(cks - nks), "normv1-only", len(nks - cks))
say("combined dup key rows", int(ck.duplicated(keep=False).sum()), "by month (top):",
    ck[ck.duplicated(keep=False)].ym.value_counts().head(10).to_dict())
a_months = set(ak.ym); cka = {k for k in cks if k[1] in a_months}
say("panel_2025 vs combined (2025+ months): a-only", len(aks - cka), "c-only", len(cka - aks))
say("combined months not in pinned:", sorted(set(ck.ym) - set(pk.ym)))
say("pinned months not in combined:", sorted(set(pk.ym) - set(ck.ym)))
# unit_name agreement pinned vs combined
cu = C.drop_duplicates("id").set_index("id")[["unit_name"]]
jc = pu.join(cu, rsuffix="_c", how="inner"); say("pinned vs combined unit_name equal per id:", int((jc.unit_name == jc.unit_name_c).sum()), "of", len(jc))
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
