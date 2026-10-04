"""Independent D42 schedule/pool check from saved memberships and spatial maps; no fits."""
import csv
import gzip
import hashlib
import json
import struct
from pathlib import Path

BASE = Path('/mnt/c/Users/swl00/geoxgb_runs')
STAGE = BASE/'geoxgb-d34-e1-brier-20261002/stage1_e1pair'
DATES = ('2018-06','2018-10','2019-02','2019-06','2019-10','2020-02','2020-06')
month = lambda s: int(s[:4])*12+int(s[5:])-1
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result = {'pairs':{},'expected_fits':0,'input_hashes':{}}
for h,g in ((4,'G1'),(8,'G4'),(12,'G2')):
    for target in DATES:
        prior = [d for d in DATES if month(d)<month(target)-h]
        if not prior: continue
        source = max(prior)
        root = f'h{h}_{target}_{g}_r80_s42_e1pair'
        path = STAGE/'roots'/root/'fold_membership.csv.gz'
        result['input_hashes'][str(path)] = sha(path)
        with gzip.open(path,'rt',encoding='utf-8') as f: members=list(csv.DictReader(f))
        fitting = [r for r in members if r['role']=='fitting']
        assert len({(r['area'],r['target_month']) for r in fitting})==len(fitting)
        pair = {'horizon':h,'target':target,'source_target':source,'maps':{}}
        for arm,date in (('old_map_refit',source),('current_map_refit',target)):
            path=STAGE/'candidates'/f'h{h}_{date}_{g}_L1_r80_s42_e1brier_gt0'/'assignment_evidence.csv'
            result['input_hashes'][str(path)]=sha(path)
            with path.open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
            amap={r['FEWSNET_admin_code']:r['spatial_partition_id'] for r in rows}
            assert len(amap)==len(rows)
            assert all((r['spatial_partition_id']=='s-1')==(int(r['search_rows'])==0) for r in rows)
            regions={}
            for region in sorted(set(amap.values())-{'s-1'}):
                pool=[r for r in fitting if amap.get(r['area'])==region]
                support={'rows':len(pool),'areas':len({r['area'] for r in pool}),
                         'dates':len({r['target_month'] for r in pool}),
                         'classes':len({r['class_code'] for r in pool}),
                         'class_counts':[sum(int(r['class_code'])==i for r in pool) for i in range(4)]}
                digest=hashlib.sha256(b''.join(struct.pack('<qq',int(r['area']),month(r['target_month'])) for r in pool)).hexdigest()
                eligible=all(support[k]>=n for k,n in dict(rows=500,areas=50,dates=6,classes=2).items())
                regions[region]={'support':support,'fit_keys_sha256':digest,'eligible':eligible}
                result['expected_fits']+=int(eligible)
            coverage={}
            for part,role in [('C','confirmation'),('E3','heldout_target')]:
                evaluation=[r for r in members if r['role']==role]
                counts={'n':len(evaluation),'missing_map':0,'s-1':0,'support_fallback':0,'local':0}
                for row in evaluation:
                    rid=amap.get(row['area'])
                    why='missing_map' if rid is None else 's-1' if rid=='s-1' else 'local' if regions[rid]['eligible'] else 'support_fallback'
                    counts[why]+=1
                assert counts['n']==sum(v for k,v in counts.items() if k!='n')
                coverage[part]=counts
            pair['maps'][arm]={'map_sha256':sha(path),'regions':regions,'coverage':coverage}
        result['pairs'][root]=pair
assert len(result['pairs'])==12 and result['expected_fits']==175
path=BASE/'d42_input_check.json'
assert not path.exists()
path.write_text(json.dumps(result,indent=2),encoding='utf-8')
print('D42 independent source schedule/pools:',len(result['pairs']),'pairs,',result['expected_fits'],'fits;',path)
