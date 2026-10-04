#!/usr/bin/env python3
"""Read v7 labels and saved fold schedules; count calendar support only."""
import argparse, collections, csv, json
from pathlib import Path

def mi(s):
    y,m=map(int,s.split('-')); assert 1<=m<=12
    return y*12+m-1

def ym(n):
    y,m=divmod(n,12); return f'{y:04d}-{m+1:02d}'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',default='/mnt/c/users/swl00/ifpri dropbox/weilun shi/google fund/analysis/2.source_code/step5_geo_rf_trial/food_crisis_cluster'); ap.add_argument('--out',default='/tmp/geoxgb_calendar_counts.json'); a=ap.parse_args()
    repo=Path(a.repo); run=repo/'FEWSNETFourClassBaseline/runs/fourclass-v7-20260928'
    obs=run/'prepared/ledgers/observations.csv'
    counts=collections.Counter(); bycountry=collections.defaultdict(collections.Counter); keys=set(); firstline={}
    with obs.open(newline='') as f:
        for ln,r in enumerate(csv.DictReader(f),2):
            m=mi(r['month_label']); assert int(r['month'])==m
            key=(int(r['area']),m); assert key not in keys; keys.add(key)
            assert int(r['class_code'])==int(r['merged_class'])-1
            raw=float(r['raw_phase']); assert raw in (1,2,3,4,5); assert int(r['merged_class'])==min(int(raw),4)
            counts[m]+=1; bycountry[r['country']][m]+=1; firstline.setdefault(m,ln)
    dates=sorted(counts); floor=mi('2014-01'); cutoff=mi('2020-12')
    def hist(v):
        start=v-59; ds=[d for d in dates if start<=d<v]; pre=[d for d in ds if d<floor]
        assert all(start<=d<v for d in ds); assert len(range(start,v))==59
        missing=[d for d in range(start,v) if d not in counts]
        return {'start_inclusive':ym(start),'end_exclusive':ym(v),'observed_dates':len(ds),'rows':sum(counts[d] for d in ds),'earliest_observed':ym(ds[0]) if ds else None,'latest_observed':ym(ds[-1]) if ds else None,'before_floor_dates':[ym(d) for d in pre],'before_floor_rows':sum(counts[d] for d in pre),'calendar_months_before_source_start':max(0,min(v,dates[0])-start),'unobserved_calendar_months':[ym(d) for d in missing]}
    def internal(o,h):
        us=[d for d in dates if d<o][-6:]; assert len(us)<=6
        out=[]
        for u in us:
            v=u-h; assert u<o and v+h==u
            out.append({'U':ym(u),'validation_rows':counts[u],'V':ym(v),'map_known_at_V':v>cutoff,'history':hist(v)})
        eligible=[d for d in dates if d<o and d-h>cutoff]
        assert all(d<o and d-h>cutoff for d in eligible)
        return {'last_six':out,'ignore_inner_map_timing_count':len(us),'eligible_inner_map_total':len(eligible),'eligible_inner_map_count_up_to_six':min(6,len(eligible)),'eligible_last_six_U':[ym(d) for d in eligible[-6:]]}
    stage1=[]
    for p in sorted((run/'stage1/folds').glob('*/candidate.json')):
        f=json.loads(p.read_text()); h=f['horizon']; t=mi(f['target_month']); o=mi(f['origin_month']); assert t-h==o
        stage1.append({'file':str(p.relative_to(repo)),'H':h,'T':ym(t),'O':ym(o),'saved_n_terminal':f['partition']['n_terminal'],'saved_accepted_splits':f['partition']['accepted_splits'],'outer_history':hist(o),'internal':internal(o,h)})
    assert len(stage1)==27
    stage3=[]
    for p in sorted((run/'stage3').glob('h*/folds/*/fold.json')):
        f=json.loads(p.read_text()); h=f['horizon']; t=mi(f['target_month']); o=mi(f['origin_month']); assert t-h==o
        status=f['status']; assert (t in counts)==(status=='fitted')
        stage3.append({'file':str(p.relative_to(repo)),'H':h,'T':ym(t),'O':ym(o),'status':status,'map_known_at_O':o>cutoff,'outer_history':hist(o),'internal':internal(o,h)})
    assert len(stage3)==120
    first={}
    for h in (4,8,12):
        folds=sorted([f for f in stage3 if f['H']==h],key=lambda f:f['T']); first[str(h)]={}
        for population in ('scheduled','fitted'):
            fs=[f for f in folds if population=='scheduled' or f['status']=='fitted']
            first[str(h)][population]={str(n):next(({'T':f['T'],'O':f['O'],'eligible':f['internal']['eligible_inner_map_count_up_to_six'],'U':f['internal']['eligible_last_six_U']} for f in fs if f['internal']['eligible_inner_map_count_up_to_six']>=n),None) for n in (1,3,6)}
    dev=[]
    for t in dates:
        if not mi('2019-01')<=t<mi('2021-01'):continue
        for h in (4,8,12):
            o=t-h; cs=[f for f in stage1 if f['H']==h and mi(f['T'])<o and mi(f['T'])>=mi('2018-01')]
            assert all(mi(f['T'])<o for f in cs)
            dev.append({'H':h,'T':ym(t),'O':ym(o),'stage1_candidate_count':len(cs),'candidate_T':[f['T'] for f in cs],'saved_root_only_candidate_count':sum(f['saved_n_terminal']==1 for f in cs),'saved_split_candidate_count':sum(f['saved_n_terminal']>1 for f in cs),'empty_candidate_set':not cs,'map_content_not_reconstructed':True,'null_map_due_to_no_candidates':not cs})
    support={c:{'first':ym(min(ds)),'last':ym(max(ds)),'rows':sum(ds.values()),'observed_dates':len(ds),'missing_global_observed_dates':[ym(d) for d in dates if d not in ds]} for c,ds in sorted(bycountry.items())}
    result={'reference':str(run.relative_to(repo)),'semantics':{'history':'[V-59,V)','candidate_U':'last six globally observed label months strictly before O','V':'U-H','map_cutoff':'2020-12','inner_eligibility':'V>2020-12 AND U<O','snapshot_floor':'2014-01','counts_are':'label-support only, global rows, no feature-availability claims'},'source':{'rows':sum(counts.values()),'unique_area_month_keys':len(keys),'first':ym(dates[0]),'last':ym(dates[-1]),'date_rows':{ym(d):counts[d] for d in dates},'first_line_per_month':{ym(d):firstline[d] for d in dates},'country_support':support,'before_floor_rows':sum(n for d,n in counts.items() if d<floor),'before_floor_dates':sum(d<floor for d in dates)},'stage1':stage1,'stage3':stage3,'first_eligible_stage3':first,'development_outer_observed_2019_2020':dev}
    histories={}
    for stage in ('stage1','stage3'):
        for fold in result[stage]:
            histories[fold['O']]=fold.pop('outer_history')
            for pair in fold['internal']['last_six']:
                history=pair.pop('history')
                assert pair['V'] not in histories or histories[pair['V']]==history
                histories[pair['V']]=history
    result['history_by_origin']=histories
    Path(a.out).write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({'out':a.out,'source':{k:result['source'][k] for k in ('rows','first','last','before_floor_rows','before_floor_dates')},'first_eligible_stage3':first,'development':dev},separators=(',',':')))
if __name__=='__main__':main()
