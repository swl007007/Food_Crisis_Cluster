import csv, hashlib, json, sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

p = Path(sys.argv[1])
def read_csv(f):
    with f.open(newline='') as h:
        return list(csv.DictReader(h))
def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()
ledger = read_csv(p / 'stage2/candidate_ledger.csv')
weights = read_csv(p / 'stage2/plan_weights.csv')
positive = [r for r in weights if float(r['weight']) > 0]
s = sum(float(r['weight']) for r in positive)
w = sorted((float(r['weight']) for r in positive), reverse=True)
summary = {
    'run': str(p.resolve()),
    'scheduled': len(ledger), 'statuses': dict(Counter(r['status'] for r in ledger)),
    'completed_root_only': sum(float(r['n_terminal']) == 1 for r in weights),
    'completed_split': sum(float(r['n_terminal']) > 1 for r in weights),
    'positive_weight_count': len(positive), 'sum_weights': s,
    'normalized_top1_share': w[0] / s,
    'normalized_top3_share': sum(w[:3]) / s,
    'weight_concentration_equivalent_NOT_independent_sample_count': s*s / sum(x*x for x in w),
    'by_scope_target_year': [], 'positive_maps': [], 'coverage_pairs': [],
    'source_sha256': {},
}
groups = defaultdict(list)
for r in weights:
    groups[(int(float(r['scope'])), r['target_month'][:4])].append(r)
for (h,y), rs in sorted(groups.items()):
    summary['by_scope_target_year'].append(dict(scope=h,target_year=y,completed=len(rs),root_only=sum(float(r['n_terminal'])==1 for r in rs),split=sum(float(r['n_terminal'])>1 for r in rs),positive=sum(float(r['weight'])>0 for r in rs)))
for name in ['candidate_ledger.csv','plan_weights.csv','consensus.json']:
    f=p/'stage2'/name
    summary['source_sha256'][str(f.relative_to(p))]=sha(f)
canonical = {}
coverage = {}
for r in positive:
    scope = int(float(r['scope']))
    ym = r['target_month']; y = ym[:4]
    f = p/'stage1/GeoRFResults'/f'result_GeoRF_{y}_fs{scope}_{ym}_visual'/f'correspondence_table_{ym}.csv'
    rows = read_csv(f)
    pairs = sorted((int(x['FEWSNET_admin_code']), x['partition_id']) for x in rows)
    assert len(pairs)==len(set(a for a,b in pairs)), 'duplicate area ID'
    labels = {}; renamed=[]
    for area,label in pairs:
        if label not in labels: labels[label]=len(labels)
        renamed.append((area, labels[label]))
    canonical[r['candidate']]=tuple(renamed)
    coverage[r['candidate']] = set(area for area,label in pairs)
    summary['positive_maps'].append(dict(candidate=r['candidate'],weight=float(r['weight']),normalized_weight=float(r['weight'])/s,map=str(f.relative_to(p)),map_sha256=sha(f),areas=len(pairs),partition_labels=dict(Counter(label for area,label in pairs)),canonical_sha256=hashlib.sha256(json.dumps(renamed,separators=(',',':')).encode()).hexdigest()))
for a,b in combinations(coverage,2):
    A,B=coverage[a],coverage[b]
    summary['coverage_pairs'].append(dict(a=a,b=b,identical_coverage=A==B,intersection=len(A&B),a_only=len(A-B),b_only=len(B-A),a_only_sorted=sorted(A-B),b_only_sorted=sorted(B-A),exact_duplicate_modulo_label_renaming=(canonical[a]==canonical[b]) if A==B else None))
summary['exact_duplicate_pairs_on_identical_coverage']=sum(x['exact_duplicate_modulo_label_renaming'] is True for x in summary['coverage_pairs'])
summary['distinct_full_coverage_partition_signatures']=len(set(canonical.values()))
print(json.dumps(summary,indent=2))
