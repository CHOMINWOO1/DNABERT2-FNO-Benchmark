"""Audit the cohort, sequence aliases and overlap-cluster construction."""
import hashlib
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from dnabert_fno.data import sha256_file
from dnabert_fno.study import atomic_json


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def digest(s):
    rc=s.translate(str.maketrans('ACGTN','TGCAN'))[::-1]
    return hashlib.sha256(min(s,rc).encode()).hexdigest()


def main():
    data=Path('data/continuous_rna');source=read(data/'source.json');rows=read(data/'rows.json')
    for name,value in source['files_sha256'].items():assert sha256_file(data/name)==value
    with (data/'sequences.fasta').open() as f:seq=[line.strip() for line in f if not line.startswith('>')]
    assert len(seq)==len(rows)==3514 and len({r['gene'] for r in rows})==len(rows)
    assert [r['global_id'] for r in rows]==list(range(len(rows)))
    for r,s in zip(rows,seq):
        assert len(s)==r['end']-r['start']==65536 and r['tss_0based']-r['start']==32768
        assert s.count('N')/len(s)==r['n_fraction'] and r['n_fraction']<=.01
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        for key in ['chrom','gene']:
            assert not {r[key] for r in rows if r['split']==a}&{r[key] for r in rows if r['split']==b}
    views={}
    for bp in [4096,16384,65536]:
        start=(65536-bp)//2;groups=defaultdict(set)
        for r,s in zip(rows,seq):groups[digest(s[start:start+bp])].add(r['split'])
        views[str(bp)]={'unique_exact_rc':len(groups),'cross_split_aliases':sum(len(v)>1 for v in groups.values())}
        assert views[str(bp)]['cross_split_aliases']==0
    chunks=defaultdict(set);occurrences=Counter()
    for r,s in zip(rows,seq):
        for start in range(0,65536,1024):
            key=digest(s[start:start+1024]);chunks[key].add(r['split']);occurrences[key]+=1
    shared={k:v for k,v in chunks.items() if len(v)>1}
    actual=read(data/'test_clusters.json');expected=[];end=-1
    for r in sorted([r for r in rows if r['split']=='test'],key=lambda r:r['start']):
        if r['start']>=end:expected.append([])
        expected[-1].append(r['global_id']);end=max(end,r['end'])
    assert actual==expected and len(actual)==533 and sum(map(len,actual))==982
    labels=np.load(data/'labels.npy');assert labels.shape==(len(rows),218) and np.isfinite(labels).all()
    original=Path('data/continuous_rna_source')
    for entry in read(original/'download_manifest.json')['files']:assert sha256_file(entry['path'])==entry['sha256']
    coords=list(csv.DictReader((original/'gene_coordinates.csv').open()))
    rawlabels=np.loadtxt(original/'rna_expression_values.csv',delimiter=',',skiprows=1)
    np.testing.assert_array_equal(labels,rawlabels[[r['source_row'] for r in rows]])
    for r in rows:
        old=coords[r['source_row']]
        assert r['gene']==old['id'] and r['chrom']==old['chrom'] and r['strand']==old['strand']
        assert r['tss_0based']==int(old['CAGE_representative_TSS'])-1
    result={'passed':True,'rows':len(rows),'split_counts':dict(Counter(r['split'] for r in rows)),
            'views':views,'chunk_unique_exact_rc':len(chunks),'chunk_cross_split_aliases':len(shared),
            'chunk_cross_split_occurrences':sum(occurrences[k] for k in shared),'test_clusters':len(actual),
            'largest_test_cluster':max(map(len,actual)),'source_sha256':sha256_file(data/'source.json'),
            'audit_script_sha256':sha256_file(__file__),'original_label_and_coordinate_mapping_verified':True,
            'note':'Exact/RC alias checks do not test near-homology.'}
    atomic_json('runs/continuous_rna/data_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
