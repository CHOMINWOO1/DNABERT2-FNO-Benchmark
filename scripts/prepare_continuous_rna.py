"""Label-independent cohort selection and verified contiguous GRCh37 windows."""
import argparse
import csv
import hashlib
import json
import time
import urllib.request
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from dnabert_fno.data import sha256_file
from dnabert_fno.study import atomic_json,stable_hash

ROOT=Path('data/continuous_rna');SOURCE=Path('data/continuous_rna_source')
BP=65536;SEED=20260920


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def reverse_complement(s):return s.translate(str.maketrans('ACGTN','TGCAN'))[::-1]


def candidates():
    manifest=read(SOURCE/'download_manifest.json')
    for f in manifest['files']:assert sha256_file(f['path'])==f['sha256']
    rows=list(csv.DictReader((SOURCE/'gene_coordinates.csv').open()));sizes={k:int(v) for k,v in (line.split() for line in (SOURCE/'hg19.chrom.sizes').read_text().splitlines())}
    pools=defaultdict(list);excluded=[]
    for i,r in enumerate(rows):
        tss=int(r['CAGE_representative_TSS'])-1;start=tss-BP//2;end=tss+BP//2;chrom=r['chrom']
        assert r['strand'] in ['+','-'] and r['split']==('test' if chrom=='chr8' else 'train')
        if chrom not in sizes or start<0 or end>=sizes[chrom]:excluded.append({'source_row':i,'reason':'boundary'});continue
        split='test' if chrom=='chr8' else 'dev' if chrom in ['chr7','chr9'] else 'train'
        query=f'{chrom.removeprefix("chr")}:{start+1}..{end}:1'
        pools[split].append({'source_row':i,'gene':r['id'],'symbol':r['symbol'],'chrom':chrom,'strand':r['strand'],
                             'tss_0based':tss,'start':start,'end':end,'split':split,'query':query})
    selected=[]
    for split in ['train','dev','test']:
        pool=sorted(pools[split],key=lambda r:hashlib.sha256(f'{SEED}:{r["gene"]}:{r["query"]}'.encode()).hexdigest())
        selected.extend(pool[:{'train':2048,'dev':512,'test':len(pool)}[split]])
    return selected,{'seed':SEED,'available':{s:len(v) for s,v in pools.items()},'selected':dict(Counter(r['split'] for r in selected)),
                     'boundary_excluded':excluded,'source_manifest_sha256':sha256_file(SOURCE/'download_manifest.json')}


def fetch_batch(number,queries):
    folder=ROOT/'sequence_responses';folder.mkdir(exist_ok=True);dest=folder/f'batch_{number:04d}.json'
    if dest.exists():
        obj=read(dest);assert obj['queries']==queries;return obj
    url='https://grch37.rest.ensembl.org/sequence/region/human'
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,data=json.dumps({'regions':queries}).encode(),headers={'Content-Type':'application/json','Accept':'application/json'})
            with urllib.request.urlopen(req,timeout=60) as response:raw=response.read()
            result=json.loads(raw);assert len(result)==len(queries)
            mapped={r['query']:r for r in result};assert set(mapped)==set(queries)
            for query in queries:
                chrom,span,strand=query.split(':');start,end=span.split('..');r=mapped[query]
                assert r['id']==f'chromosome:GRCh37:{chrom}:{start}:{end}:{strand}' and len(r['seq'])==BP
                assert not set(r['seq'].upper())-set('ACGTN')
            obj={'url':url,'queries':queries,'response':result,'raw_sha256':hashlib.sha256(raw).hexdigest(),'download_unix':time.time()}
            atomic_json(dest,obj);return obj
        except Exception:
            if attempt==3:raise
            time.sleep(2**attempt)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);selected,audit=candidates();selection={'rows':selected,'audit':audit}
    if (ROOT/'selection.json').exists():assert read(ROOT/'selection.json')==selection
    else:atomic_json(ROOT/'selection.json',selection)
    queries=sorted({r['query'] for r in selected});batches=[queries[i:i+25] for i in range(0,len(queries),25)]
    if args.download:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures=[executor.submit(fetch_batch,i,q) for i,q in enumerate(batches)]
            for i,f in enumerate(as_completed(futures),1):
                f.result()
                if i%10==0 or i==len(futures):print(f'GRCh37 sequences {i}/{len(futures)} batches',flush=True)
    sequences={};hashes={}
    for i,qs in enumerate(batches):
        path=ROOT/'sequence_responses'/f'batch_{i:04d}.json';obj=read(path);assert obj['queries']==qs;hashes[str(path)]=sha256_file(path)
        for r in obj['response']:sequences[r['query']]=r['seq'].upper()
    rows=[];seqs=[];excluded=[]
    for r in selected:
        seq=sequences[r['query']];n=seq.count('N')/BP
        if n>.01:excluded.append({'gene':r['gene'],'reason':'N_fraction_over_1pct','n_fraction':n});continue
        if r['strand']=='-':seq=reverse_complement(seq)
        rows.append({**r,'global_id':len(rows),'n_fraction':n});seqs.append(seq)
    # Exact/RC whole-window aliases may share a label, but cannot cross splits.
    aliases=defaultdict(set)
    for r,s in zip(rows,seqs):aliases[min(s,reverse_complement(s))].add(r['split'])
    assert all(len(v)==1 for v in aliases.values())
    labels=np.loadtxt(SOURCE/'rna_expression_values.csv',delimiter=',',skiprows=1)
    assert labels.shape==(23817,218) and np.isfinite(labels).all()
    chosen=labels[[r['source_row'] for r in rows]].astype(np.float64);np.save(ROOT/'labels.npy',chosen)
    with (SOURCE/'rna_expression_values.csv').open() as f:tracks=next(csv.reader(f))
    assert len(tracks)==218
    with (ROOT/'sequences.fasta').open('w') as f:
        for r,s in zip(rows,seqs):f.write(f'>{r["global_id"]}|{r["gene"]}\n{s}\n')
    atomic_json(ROOT/'rows.json',rows);atomic_json(ROOT/'tracks.json',tracks)
    # Verify independent UCSC hg19 sequence extraction for each split and strand.
    cross=[]
    for split in ['train','dev','test']:
        for strand in ['+','-']:
            r=next(r for r in rows if r['split']==split and r['strand']==strand)
            dest=ROOT/f'ucsc_{split}_{"plus" if strand=="+" else "minus"}.json'
            url=f'https://api.genome.ucsc.edu/getData/sequence?genome=hg19;chrom={r["chrom"]};start={r["start"]};end={r["end"]}'
            if not dest.exists():
                assert args.download,'UCSC crosscheck missing'
                with urllib.request.urlopen(url,timeout=45) as f:raw=f.read()
                dest.write_bytes(raw)
            response=read(dest);assert response['genome']=='hg19' and response['chrom']==r['chrom']
            assert response['start']==r['start'] and response['end']==r['end']
            assert response['dna'].upper()==sequences[r['query']]
            cross.append({'global_id':r['global_id'],'url':url,'response_sha256':sha256_file(dest),'exact_match':True})
    # Connected overlapping 64kb intervals form bootstrap clusters within the test chromosome.
    test=sorted([r for r in rows if r['split']=='test'],key=lambda r:r['start']);clusters=[];end=-1
    for r in test:
        if r['start']>=end:clusters.append([])
        clusters[-1].append(r['global_id']);end=max(end,r['end'])
    atomic_json(ROOT/'test_clusters.json',clusters)
    files=['rows.json','labels.npy','sequences.fasta','tracks.json','selection.json','test_clusters.json']
    source={'source_revision':read(SOURCE/'download_manifest.json')['revision'],'assembly':'GRCh37/hg19','coordinate_convention':'TSS input 1-based; stored start/end 0-based half-open; strand oriented',
            'audit':audit,'N_excluded':excluded,'counts':dict(Counter(r['split'] for r in rows)),'tracks':218,'max_bp':BP,
            'sequence_response_sha256':hashes,'ucsc_crosschecks':cross,'files_sha256':{name:sha256_file(ROOT/name) for name in files},
            'test_clusters':len(clusters),'largest_test_cluster':max(map(len,clusters)),'selection_uses_labels':False,
            'note':'Published labels are already log1p/standardized; downstream normalization will be fit on the selected train rows only.'}
    atomic_json(ROOT/'source.json',source);print(json.dumps({k:source[k] for k in ['counts','test_clusters','largest_test_cluster']},indent=2))


if __name__=='__main__':main()
