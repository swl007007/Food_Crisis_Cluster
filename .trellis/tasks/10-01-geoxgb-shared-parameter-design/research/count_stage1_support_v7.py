#!/usr/bin/env python3
"""Read-only stdlib count diagnostic for v7 Stage1 actual keyed observations.
Usage: python3 /tmp/count_stage1_support_v7.py [run-directory] [output-json]
Temporal tails are exploratory counts, not approved schedules. Release dates unknown.
"""
import collections,csv,gzip,json,sys
from pathlib import Path
RUN=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/c/users/swl00/ifpri dropbox/weilun shi/google fund/analysis/2.source_code/step5_geo_rf_trial/food_crisis_cluster/FEWSNETFourClassBaseline/runs/fourclass-v7-20260928')
OUT=Path(sys.argv[2]) if len(sys.argv)>2 else Path('/tmp/stage1_support_v7.json')
def mi(s):
 y,m=map(int,s.split('-'));return y*12+m-1
def ml(x):return f'{x//12:04d}-{x%12+1:02d}'
def summary(xs):
 s=sorted(xs); n=len(s)
 if not n:return {'n':0}
 def q(p):
  k=(n-1)*p;i=int(k);return round(s[i]+(s[min(i+1,n-1)]-s[i])*(k-i),3)
 return dict(n=n,min=s[0],p10=q(.1),median=q(.5),p90=q(.9),max=s[-1],zero=sum(x==0 for x in s),le1=sum(x<=1 for x in s),le4=sum(x<=4 for x in s),le9=sum(x<=9 for x in s))
def bucket():return {'counts':[0]*4,'dates':set()}
def add(b,m,c):b['counts'][c]+=1;b['dates'].add(m)
def support(bs):
 bs=list(bs);return {'rows':summary([sum(b['counts']) for b in bs]),'unique_observed_target_months':summary([len(b['dates']) for b in bs]),'class_support':{str(i+1):summary([b['counts'][i] for b in bs]) for i in range(4)}}
def foldtot(bs):
 bs=list(bs);return {'rows':sum(sum(b['counts']) for b in bs),'class_counts_1to4':[sum(b['counts'][i] for b in bs) for i in range(4)],'dates':sorted({ml(m) for b in bs for m in b['dates']})}
obs={}
with (RUN/'prepared/ledgers/observations.csv').open(newline='') as f:
 for row in csv.DictReader(f):
  k=(int(row['area']),mi(row['month_label']));c=int(row['class_code']);assert k not in obs and c==int(row['merged_class'])-1 and c in range(4);obs[k]=c
allareas={role:[] for role in ('history','fitting','validation')};allbranches={role:[] for role in ('history','fitting','validation')}
byh=collections.defaultdict(lambda:{role:[] for role in ('history','fitting','validation')}); folds=[]; temporal=collections.defaultdict(lambda:collections.defaultdict(list));decision_train=[];decision_val=[];fit_records=[];terminal_fit_records=[];recover=[]
for cp in sorted((RUN/'stage1/folds').glob('*/candidate.json')):
 cand=json.loads(cp.read_text());name=cp.parent.name;H=cand['horizon'];O=mi(cand['origin_month'])
 area=collections.defaultdict(lambda:{role:bucket() for role in ('history','fitting','validation')});branch=collections.defaultdict(lambda:{role:bucket() for role in ('history','fitting','validation')});history=[];targetareas=set();seen=set()
 with gzip.open(cp.parent/'fold_membership.csv.gz','rt',newline='') as f:
  for row in csv.DictReader(f):
   a=int(row['area']);m=mi(row['target_month']);role=row['role'];b=row['branch_id'];key=(a,m);assert key not in seen;seen.add(key);c=obs[key]
   if role=='heldout_target':targetareas.add(a);continue
   assert role in ('fitting','validation') and O-35<=m<O
   add(area[a][role],m,c);add(area[a]['history'],m,c);add(branch[b][role],m,c);add(branch[b]['history'],m,c);history.append((a,m,c))
 for a in targetareas:area[a]
 assert set(area)==targetareas
 for role in allareas:
  aa=[x[role] for x in area.values()];bb=[x[role] for x in branch.values()];allareas[role].extend(aa);allbranches[role].extend(bb);byh[H][role].extend(aa)
 for role in ('fitting','validation'):
  tot=foldtot(x[role] for x in area.values());assert tot['rows']==cand['rows'][role] and tot['class_counts_1to4']==cand['class_counts'][role]
 assert set(branch)==set(cand['partition']['terminal_partitions'])
 ft={'fold':name,'horizon':H,'origin_month':cand['origin_month'],'target_areas':len(targetareas),'training_areas':cand['areas']['training'],'terminal_branches':len(branch),'observed_history_dates':foldtot(x['history'] for x in area.values())['dates'],'roles':{role:foldtot(x[role] for x in area.values()) for role in allareas},'temporal':{}}
 for tail in (12,18):
  half=tail//2;bounds={'E1':(O-tail,O-half),'E2':(O-half,O),'earliest_prefix':(O-35,O-tail-H)};t={}
  for label,(lo,hi) in bounds.items():
   ab={a:bucket() for a in targetareas}
   for a,m,c in history:
    if lo<=m<hi:add(ab[a],m,c)
   temporal[(H,tail)][label].extend(ab.values());t[label]={'interval':f'[{ml(lo)},{ml(hi)})',**foldtot(ab.values()),'area_support':support(ab.values())}
  ft['temporal'][str(tail)]=t
 for d in cand['partition']['decisions']:
  decision_train.extend(d.get('rows_train',[]));decision_val.extend(d.get('rows_val',[]))
 fit_records.extend(cand['fits']['georf_fit_log']);terminal_fit_records.extend(x for x in cand['partition']['terminal_checkpoint_routes'].values() if x)
 retained=RUN/'stage1/retained'/name/'checkpoints';files=sorted(retained.glob('rf_*')) if retained.is_dir() else []
 if files:
  last={e['saved_as'] or 'root':e for e in cand['fits']['georf_saved_log']}; rec=[]
  for p in files:
   b=p.name[3:] or 'root';e=last.get(b);rec.append({'branch':b,'path':str(p.relative_to(RUN)),'saved_fitting_counts_1to4':e.get('real_class_counts') if e else None,'saved_fitting_rows':e.get('real_rows') if e else None,'terminal':b in branch})
  recover.append({'fold':name,'checkpoints':rec})
 folds.append(ft)
def fitsummary(es):return {'logged_fits':len(es),'real_rows':summary([e['real_rows'] for e in es]),'class_support':{str(i+1):summary([e['real_class_counts'][i] for e in es]) for i in range(4)}}
out={'source_run':str(RUN),'label_source':'prepared/ledgers/observations.csv','membership_sources':'stage1/folds/*/fold_membership.csv.gz','label_semantics':'merged_class 1..4, class_code 0..3; class4 merges raw4/5; counts exclude four pseudo rows','denominators':'fold-area uses every heldout target area, including zero-history areas; fold-branch uses observed terminal assignments; repeated areas across folds counted separately; quantiles linear interpolation','fold_count':len(folds),'observations_rows':len(obs),'fold_area':{r:support(x) for r,x in allareas.items()},'fold_area_by_horizon':{str(h):{r:support(x) for r,x in y.items()} for h,y in byh.items()},'fold_terminal_branch':{r:support(x) for r,x in allbranches.items()},'recorded_candidate_child_proposals':{'n':len(decision_train),'fitting_rows':summary(decision_train),'validation_rows':summary(decision_val),'limitation':'decision rows are aggregate support for two best child proposals per logged decision; no keyed child labels/validation class4 or exhaustive scan attempts'},'actual_logged_fit_support':fitsummary(fit_records),'terminal_checkpoint_actual_fit_support':fitsummary(terminal_fit_records),'recoverable_retained_checkpoint_folds':len(recover),'recoverable_retained_checkpoints':sum(len(x['checkpoints']) for x in recover),'recoverable_retained_terminal_checkpoints':sum(sum(p['terminal'] for p in x['checkpoints']) for x in recover),'retained':recover,'exploratory_temporal':{f'H{h}_tail{tail}':{r:support(x) for r,x in val.items()} for (h,tail),val in temporal.items()},'temporal_limitations':'Count-only diagnostic, not approved schedule. Target<origin necessary but insufficient because label release dates unknown. Existing [O-35,O) history and target-area restriction retained. E1/E2 label rows counted independent of current random fitting/validation assignment. Prefix target<earliest E1 forecast origin (O-tail-H).','folds':folds}
OUT.write_text(json.dumps(out,ensure_ascii=False,separators=(',',':'))+'\n');print(json.dumps({k:out[k] for k in ('fold_count','observations_rows','recoverable_retained_checkpoint_folds','recoverable_retained_checkpoints','recoverable_retained_terminal_checkpoints')},indent=2));print(str(OUT))
