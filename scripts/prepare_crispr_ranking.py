"""Public CRISPR benchmark + verified GRCh38 reference windows, chromosome holdout."""
import argparse
import csv
import gzip
import hashlib
import json
import time
import urllib.request
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from dnabert_fno.data import sha256_file
from dnabert_fno.study import atomic_json

ROOT=Path('data/crispr_ranking')
SOURCE=Path('data/crispr_source/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz')
DEV={'chr2','chr4','chr6'}
TEST={'chr1','chr3','chr5'}


def query(chrom,start,end):return f'{chrom.removeprefix("chr")}:{start+1}..{end}:1'


def candidates():
    source_manifest=json.loads(Path('data/crispr_source/download_manifest.json').read_text())
    expected=next(r['sha256'] for r in source_manifest['files'] if Path(r['path'])==SOURCE)
    assert sha256_file(SOURCE)==expected
    with gzip.open(SOURCE,'rt') as f:raw=list(csv.DictReader(f,delimiter='\t'))
    records=[];excluded=[]
    for i,r in enumerate(raw):
        assert r['CellType']=='K562' and r['ValidConnection']=='TRUE' and r['chrTSS']==r['chrom']
        assert int(r['endTSS'])-int(r['startTSS']) in [0,1]
        e=(int(r['chromStart'])+int(r['chromEnd']))//2;p=int(r['startTSS'])
        if min(e,p)<500:
            excluded.append({'source_row':i,'reason':'missing_or_invalid_TSS_coordinate'})
            continue
        chrom=r['chrom'];label=int(r['Regulated']=='TRUE')
        assert r['Regulated'] in ['TRUE','FALSE']
        assert bool(label)==(r['Significant']=='TRUE' and float(r['EffectSize'])<0)
        records.append({'source_row':i,'chrom':chrom,'enhancer_start':e-500,'enhancer_end':e+500,
                        'promoter_start':p-500,'promoter_end':p+500,'tss':p,
                        'tested_start':int(r['chromStart']),'tested_end':int(r['chromEnd']),
                        'gene':r['measuredGeneEnsemblId'],'symbol':r['measuredGeneSymbol'],
                        'promoter_id':f'{chrom}:{p}:{r["measuredGeneEnsemblId"]}',
                        'enhancer_query':query(chrom,e-500,e+500),'promoter_query':query(chrom,p-500,p+500),
                        'distance':abs((int(r['chromStart'])+int(r['chromEnd']))/2-p),
                        'source_distance':float(r['distanceToTSS']),'label':label,'dataset':r['Dataset'],
                        'split':'test' if chrom in TEST else 'dev' if chrom in DEV else 'train'})
    return records,excluded


def fetch_batch(index,queries):
    folder=ROOT/'sequence_responses';folder.mkdir(exist_ok=True)
    dest=folder/f'batch_{index:04d}.json'
    if dest.exists():
        result=json.loads(dest.read_text());assert result['queries']==queries
        return result
    url='https://rest.ensembl.org/sequence/region/human'
    body=json.dumps({'regions':queries}).encode()
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,data=body,headers={'Content-Type':'application/json','Accept':'application/json'})
            with urllib.request.urlopen(req,timeout=45) as response:raw=response.read()
            data=json.loads(raw);assert len(data)==len(queries)
            byquery={r['query']:r for r in data};assert set(byquery)==set(queries)
            for q in queries:
                r=byquery[q];chrom,span,strand=q.split(':');start,end=span.split('..')
                assert r['id']==f'chromosome:GRCh38:{chrom}:{start}:{end}:{strand}'
                assert len(r['seq'])==1000 and not set(r['seq'].upper())-set('ACGTN')
            result={'url':url,'queries':queries,'response':data,'raw_response_sha256':hashlib.sha256(raw).hexdigest(),
                    'download_unix':time.time(),'assembly':'GRCh38','coordinates':'Ensembl 1-based inclusive, positive reference orientation'}
            atomic_json(dest,result);return result
        except Exception:
            if attempt==3:raise
            time.sleep(2**attempt)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');args=parser.parse_args()
    records,excluded=candidates();queries=sorted({r[k] for r in records for k in ['enhancer_query','promoter_query']})
    ROOT.mkdir(exist_ok=True)
    if args.download:
        batches=[queries[i:i+50] for i in range(0,len(queries),50)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures=[executor.submit(fetch_batch,i,q) for i,q in enumerate(batches)]
            for i,future in enumerate(as_completed(futures),1):
                future.result()
                if i%10==0 or i==len(futures):print(f'GRCh38 batches {i}/{len(futures)}',flush=True)
    sequence={};response_hashes={}
    for i in range(0,len(queries),50):
        path=ROOT/'sequence_responses'/f'batch_{i//50:04d}.json'
        d=json.loads(path.read_text());assert d['queries']==queries[i:i+50]
        response_hashes[str(path)]=sha256_file(path)
        for r in d['response']:sequence[r['query']]=r['seq'].upper()
    assert set(sequence)==set(queries)
    cross_path=ROOT/'ucsc_crosscheck.json'
    if args.download and not cross_path.exists():
        chosen=[queries[i] for i in np.linspace(0,len(queries)-1,6,dtype=int)]
        checks=[]
        for q in chosen:
            chrom,span,_=q.split(':');a,b=map(int,span.split('..'))
            url=f'https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=chr{chrom};start={a-1};end={b}'
            with urllib.request.urlopen(url,timeout=45) as response:d=json.load(response)
            assert d['genome']=='hg38' and d['start']==a-1 and d['end']==b
            assert d['dna'].upper()==sequence[q]
            checks.append({'query':q,'url':url,'sequence_sha256':hashlib.sha256(sequence[q].encode()).hexdigest(),'exact_match':True})
        atomic_json(cross_path,checks)
    checks=json.loads(cross_path.read_text());assert len(checks)==6 and all(r['exact_match'] for r in checks)
    kept=[]
    for r in records:
        r={**r,'enhancer':sequence[r['enhancer_query']],'promoter':sequence[r['promoter_query']]}
        if abs((r['enhancer_start']+500)-r['tss'])<1000:excluded.append({'source_row':r['source_row'],'reason':'overlapping_enhancer_and_promoter_input_windows'})
        elif any(r[k].count('N')/1000>.1 for k in ['enhancer','promoter']):excluded.append({'source_row':r['source_row'],'reason':'more_than_10_percent_N'})
        else:kept.append(r)
    rc=lambda s:min(s,s.translate(str.maketrans('ACGTN','TGCAN'))[::-1])
    # Exact sequence aliases in another chromosome are rare but must not cross the split.
    seen=set();splits={};overlap_excluded=[]
    for split in ['train','dev','test']:
        accepted=[]
        for r in kept:
            if r['split']!=split:continue
            if any(rc(r[k]) in seen for k in ['enhancer','promoter']):
                overlap_excluded.append({'source_row':r['source_row'],'reason':'exact_RC_component_seen_in_earlier_split'})
            else:accepted.append(r)
        splits[split]=accepted
        seen.update(rc(r[k]) for r in accepted for k in ['enhancer','promoter'])
    audit={}
    for split,rs in splits.items():
        groups=defaultdict(list)
        for i,r in enumerate(rs):r['row_id']=i;groups[r['promoter_id']].append(r['label'])
        audit[split]={'rows':len(rs),'positives':sum(r['label'] for r in rs),'genes':len({r['gene'] for r in rs}),
                      'promoter_loci':len(groups),'mixed_promoter_loci':sum(0<sum(y)<len(y) for y in groups.values()),
                      'chromosomes':sorted({r['chrom'] for r in rs})}
        path=ROOT/f'{split}.json'
        if path.exists():assert json.loads(path.read_text())==rs
        else:atomic_json(path,rs)
    assert audit['dev']['mixed_promoter_loci']>=10 and audit['test']['mixed_promoter_loci']>=15
    manifest={'source_manifest_sha256':sha256_file('data/crispr_source/download_manifest.json'),'source_sha256':sha256_file(SOURCE),
              'window_bp':1000,'orientation':'positive genomic reference; no strand annotation available in source',
              'distance':'absolute GRCh38 interval midpoint to startTSS; recalculated instead of using source distance',
              'splits':audit,'sha256':{s:sha256_file(ROOT/f'{s}.json') for s in splits},'sequence_response_sha256':response_hashes,
              'ucsc_crosscheck_sha256':sha256_file(cross_path),'excluded':excluded+overlap_excluded,
              'limitation':'Curated K562 CRISPR assay labels; negative is not proof of no effect. Test only three chromosomes. Reference DNA is not K562-specific haplotypes. No intervening sequence or chromatin features.'}
    path=ROOT/'source.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:atomic_json(path,manifest)
    print(json.dumps({'splits':audit,'unique_sequence_windows':len(queries),'excluded_rows':len(manifest['excluded'])},indent=2),flush=True)


if __name__=='__main__':main()
