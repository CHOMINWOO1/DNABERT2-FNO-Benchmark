"""Fixed train/dev and a fresh component-disjoint holdout for information ablation."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from dnabert_fno.data import sha256_file
from prepare_long_epi import canonical, component_groups


def prepare():
    out=Path('data/epi_information')
    if (out/'source.json').exists():
        manifest=json.loads((out/'source.json').read_text())
        for split,digest in manifest['sha256'].items():
            assert sha256_file(out/f'{split}.csv')==digest
        for path,digest in manifest['prior_manifest_sha256'].items():
            assert sha256_file(path)==digest
        return manifest
    assert not out.exists(), 'Incomplete preparation requires inspection'
    prior_paths=['data/long_epi_pilot/source.json','data/epi_diagnostic_holdout/source.json']
    old,fresh=[json.loads(Path(p).read_text()) for p in prior_paths]
    raw={}
    for split,record in old['source_files'].items():
        assert sha256_file(record['path'])==record['sha256']
        raw[split]=list(csv.DictReader(gzip.open(record['path'],'rt')))
    all_rows=[dict(r,source_split=s,source_row=i) for s,rs in raw.items() for i,r in enumerate(rs)]
    lookup={(r['source_split'],r['source_row']):r for r in all_rows}
    resolve=lambda records:[lookup[(r['source_split'],r['source_row'])] for r in records]
    excluded=resolve([r for rs in old['assignment'].values() for r in rs]+fresh['assignment'])
    maps={}
    for role,widths in [('enhancer',[600,1200,3000]),('promoter',[400,800,2000])]:
        seq=sorted({r[role] for r in all_rows})
        maps[role]=dict(zip(seq,component_groups(seq,widths)))
    forbidden={role:{m[r[role]] for r in excluded} for role,m in maps.items()}
    eligible=[r for r in all_rows if all(m[r[role]] not in forbidden[role] for role,m in maps.items())]
    unique={}
    for r in eligible:
        key=(canonical(r['enhancer']),canonical(r['promoter']))
        if key in unique: assert int(unique[key]['label'])==int(r['label'])
        else: unique[key]=r
    rng=np.random.default_rng(20260913); chosen=[]; counts={}
    for label in [0,1]:
        pool=[r for _,r in sorted(unique.items()) if int(r['label'])==label]
        counts[str(label)]=len(pool); assert len(pool)>=512
        chosen.extend(pool[i] for i in rng.choice(len(pool),512,replace=False))
    chosen.sort(key=lambda r:(r['source_split'],r['source_row']))
    splits={s:resolve(old['assignment'][s]) for s in ['train','dev']};splits['test']=chosen
    for s in ['train','dev']:
        with Path(f'data/long_epi_pilot/bp5000/{s}.csv').open(newline='') as f: previous=list(csv.DictReader(f))
        assert [(r['enhancer']+r['promoter'],int(r['label'])) for r in splits[s]]==[(r['sequence'],int(r['label'])) for r in previous]
    overlap={role:len({m[r[role]] for r in chosen}&forbidden[role]) for role,m in maps.items()}
    assert not any(overlap.values())
    out.mkdir()
    for s,records in splits.items():
        with (out/f'{s}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['enhancer','promoter','label']);w.writeheader()
            w.writerows({k:r[k] for k in w.fieldnames} for r in records)
    manifest={'seed':20260913,'rows':{s:len(rs) for s,rs in splits.items()},'eligible_labels':counts,
              'prior_manifest_sha256':{p:sha256_file(p) for p in prior_paths},'source_files':old['source_files'],
              'sha256':{s:sha256_file(out/f'{s}.csv') for s in splits},
              'assignment':{s:[{k:r[k] for k in ['source_split','source_row','label']} for r in rs] for s,rs in splits.items()},
              'test_component_group_overlap_with_all_previous_epi_rows':overlap,
              'limitation':'Same GM12878 corpus; not chromosome/external holdout. Near homology and cross-role genomic overlap uncontrolled. Rows within test may share components.'}
    (out/'source.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    m=prepare(); print(json.dumps({k:v for k,v in m.items() if k not in ['assignment','source_files']},indent=2))
