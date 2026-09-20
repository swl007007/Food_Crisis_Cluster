import pathlib,json,hashlib,sys
import numpy as np,pandas as pd
r=pathlib.Path(r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\2.source_code\Step5_Geo_RF_trial\Food_Crisis_Cluster\IPCCHGeoRFExperiment\runs\ipcch-v1-20260920d")
a=pd.read_csv(r/'stage1/area_assignments.csv',keep_default_na=False);d=pd.read_csv(r/'stage1/eligible_donors.csv',keep_default_na=False);c=pd.read_csv(r/'geography/reference_coordinates.csv').set_index('area_id')
s=pd.read_csv(r/'stage1/split_outcomes.csv.gz'); learned=pd.read_csv(r/'stage1/learned_map.csv',keep_default_na=False)
assert len(a)==6227 and a.admin_code.nunique()==6227 and set(a.admin_code)==set(c.index)
counts=s.groupby(['admin_code','split_role']).size().unstack(fill_value=0)
assert set(d.admin_code)==set(learned.admin_code)==set(counts.query('fit>=1 and validation>=1').index)
assert np.array_equal(d.fit_outcomes,counts.loc[d.admin_code,'fit']) and np.array_equal(d.validation_outcomes,counts.loc[d.admin_code,'validation'])
assert np.allclose(d[['ref_lat','ref_lon']],c.loc[d.admin_code,['ref_lat','ref_lon']])
rec=a[a.assignment_source!='learned'].copy(); dc=np.radians(d[['ref_lat','ref_lon']].to_numpy());rc=np.radians(c.loc[rec.admin_code,['ref_lat','ref_lon']].to_numpy())
best=[];dist=[]
for x in rc:
 h=np.sin((dc[:,0]-x[0])/2)**2+np.cos(x[0])*np.cos(dc[:,0])*np.sin((dc[:,1]-x[1])/2)**2
 dd=6371.0*2*np.arctan2(np.sqrt(h),np.sqrt(1-h));i=np.argmin(dd);best.append(d.admin_code.iloc[i]);dist.append(dd[i])
assert np.array_equal(best,rec.nearest_eligible_admin_code)
err=np.max(np.abs(np.array(dist)-rec.nearest_eligible_distance_km.astype(float)))
assert err<1e-8,err
assert np.array_equal(np.array(dist)<=100,rec.assignment_source=='nearest_donor')
assert set(a[a.assignment_source=='nearest_donor'].donor_admin_code).issubset(set(d.admin_code))
b=r/'stage1/result_GeoRF/space_partitions'; sb=pd.read_pickle(b/'s_branch.pkl'); bids=np.load(b/'X_branch_id.npy');bt=np.load(b/'branch_table.npy')
assert list(sb.columns)==[''] and set(np.unique(bids))=={''};assert set(sb[''][sb['']>=0])==set(learned.admin_code)
sing=pd.read_csv(r/'stage1/singleton_scores.csv.gz',keep_default_na=False); assert len(sing)==5888 and set(sing.admin_code).isdisjoint(set(d.admin_code)); assert set(sing.branch_id)=={''}
print('MAP VERIFIED',a.assignment_source.value_counts().to_dict(),'donors',len(d),'distance max error',err,'s_branch',sb.shape,'X_branch_id',bids.shape,'branch_table',bt.tolist(),'singleton views',len(sing))
base=r/'baseline/GeoRFBaseline';manifest=json.loads((base/'MANIFEST.json').read_text());changes=[]
for rel,sha in manifest['files_sha256'].items():
 if hashlib.sha256((base/rel).read_bytes()).hexdigest()!=sha:changes.append(rel)
assert changes==['src/model/GeoRF.py'],changes
print('BASELINE verified payload',len(manifest['files_sha256']),'only changed',changes)
g=json.loads((r/'geography/geography_audit.json').read_text())
for key in ['geometry_source','local_copy']:
 path=pathlib.Path(g[key]['path'])
 for name,sha in g[key]['component_sha256'].items():assert hashlib.sha256((path.parent/name).read_bytes()).hexdigest()==sha
for key in ['reference_coordinates','country_lookup']: assert hashlib.sha256(pathlib.Path(g[key]['path']).read_bytes()).hexdigest()==g[key]['sha256']
print('GEOGRAPHY raw and local component hashes verified')

import geopandas as gpd,shapely
sys.path.insert(0,str(pathlib.Path(__file__).parent/'IPCCHGeoRFExperiment'))
import baseline_runtime as brt,prepare_data as pdx
raw=gpd.read_file(g['geometry_source']['path']);out=gpd.read_file(g['local_copy']['path'])
assert len(out)==6227 and out.geometry.is_valid.all() and not out.geometry.is_empty.any() and set(out.geom_type)<= {'Polygon','MultiPolygon'}
assert set(raw.admin_code)==set(out.admin_code)
audit=pd.read_csv(r/'geography/geometry_repair_audit.csv');assert (audit.discarded_area_total==0).all();assert (audit.outcome=='unchanged_valid').sum()==5974
# Shapefile serialization can reorient rings: compare geometric equality, not WKB.
out=out.set_index('admin_code'); bad=[]
for _,row in raw.iterrows():
 expected=row.geometry if row.geometry.is_valid else pdx.authorized_areal_component(shapely.make_valid(row.geometry))[0]
 if not expected.equals(out.loc[row.admin_code].geometry):bad.append(int(row.admin_code))
assert not bad,bad
for shape in [shapely.Point(0,0),shapely.LineString([(0,0),(1,1)])]:
 try:pdx.repair_geometry(shape)
 except pdx.GeometryRepairError:pass
 else:raise AssertionError('nonpolygon passed')
original=(base/'src/model/GeoRF.py').read_text();assert original.count(brt.PATCH_NEW)==1
reverted=original.replace(brt.PATCH_NEW,brt.PATCH_OLD,1)
# Baseline archive uses CRLF; try both byte serializations.
expected=manifest['files_sha256']['src/model/GeoRF.py']
import zipfile
zpath=r.parents[2]/'GeoRFBaseline/releases/georf-baseline-v0.1.0.zip'
assert hashlib.sha256(zpath.read_bytes()).hexdigest()==brt.RELEASE_SHA256
with zipfile.ZipFile(zpath) as z: pristine=z.read('GeoRFBaseline/src/model/GeoRF.py')
assert hashlib.sha256(pristine).hexdigest()==expected
assert reverted==pristine.decode('utf-8').replace('\r\n','\n').replace('\r','\n')
print('GEOMETRY all 6227 serialized outputs equal independently reconstructed authorized repair; valid nonpolygons refused; BASELINE GeoRF reverse patch matches pristine hash')
