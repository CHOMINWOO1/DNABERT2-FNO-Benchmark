"""Distance-caliper audit and fixed chromosome folds, without model scores."""
import json
from collections import defaultdict
from pathlib import Path
from dnabert_fno.data import sha256_file
from dnabert_fno.study import atomic_json

FOLDS=[['chr19','chr13','chr14','chr15','chr16'],['chr11','chr20','chr9'],['chr7','chr8','chr12','chr10'],
       ['chrX','chr1','chr3','chr22'],['chr5','chr2','chr17','chr18','chr4','chr6','chr21']]


def matched_pairs(rows,ratio):
    groups=defaultdict(list)
    for r in rows:groups[r['promoter_id']].append(r)
    pairs=[]
    for key,rs in sorted(groups.items()):
        for p in sorted([r for r in rs if r['label']],key=lambda r:r['global_id']):
            for n in sorted([r for r in rs if not r['label']],key=lambda r:r['global_id']):
                value=max(p['distance'],n['distance'])/min(p['distance'],n['distance'])
                if value<=ratio:
                    pairs.append({'promoter_id':key,'gene':p['gene'],'fold':p['fold'],'positive':p['global_id'],
                                  'negative':n['global_id'],'distance_ratio':value})
    return pairs


def main():
    out=Path('data/crispr_matched_cv');out.mkdir(exist_ok=True)
    prior=Path('data/crispr_ranking');manifest=json.loads((prior/'source.json').read_text())
    rows=[]
    for split,digest in manifest['sha256'].items():
        assert sha256_file(prior/f'{split}.json')==digest
        rows.extend(json.loads((prior/f'{split}.json').read_text()))
    rows.sort(key=lambda r:r['source_row']);assignment={c:f for f,cs in enumerate(FOLDS) for c in cs}
    assert len(assignment)==23
    for i,r in enumerate(rows):r['global_id']=i;r['fold']=assignment[r['chrom']]
    primary=matched_pairs(rows,1.25);audit={};sens={}
    for ratio in [1.1,1.25,1.5,2.0]:
        p=matched_pairs(rows,ratio)
        audit[str(ratio)]={'comparisons':len(p),'promoter_loci':len({x['promoter_id'] for x in p}),
                          'genes':len({x['gene'] for x in p}),'positive_rows':len({x['positive'] for x in p}),
                          'negative_rows':len({x['negative'] for x in p})}
        if ratio in [1.1,1.5]:sens[str(ratio)]=p
    foldstats=[]
    for fold,cs in enumerate(FOLDS):
        r=[x for x in rows if x['fold']==fold];p=[x for x in primary if x['fold']==fold]
        g=defaultdict(list)
        for x in r:g[x['promoter_id']].append(x['label'])
        foldstats.append({'fold':fold,'chromosomes':cs,'rows':len(r),'positives':sum(x['label'] for x in r),
                          'mixed_promoters':sum(0<sum(y)<len(y) for y in g.values()),
                          'matched_promoters':len({x['promoter_id'] for x in p}),'matched_comparisons':len(p)})
    assert all(x['matched_promoters']==9 for x in foldstats)
    for name,value in [('rows',rows),('matched_pairs',primary),('sensitivity_pairs',sens)]:
        path=out/f'{name}.json'
        if path.exists():assert json.loads(path.read_text())==value
        else:atomic_json(path,value)
    result={'prior_data_manifest_sha256':sha256_file(prior/'source.json'),'folds':foldstats,'caliper_audit':audit,
            'primary_ratio':1.25,'files_sha256':{name:sha256_file(out/f'{name}.json') for name in ['rows','matched_pairs','sensitivity_pairs']},
            'fold_rule':'Five disjoint chromosome sets, balanced to nine primary-matched loci per fold using labels/distances only; before model scores. Outer test f, dev (f+1)%5, train remaining three folds.',
            'scope':'Exploratory chromosome-out-of-fold reuse of the previously accessed K562 corpus, not a new independent external test.'}
    path=out/'source.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:atomic_json(path,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
