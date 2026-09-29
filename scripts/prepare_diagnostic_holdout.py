"""Seal unused component-disjoint EPI pairs before diagnostic model selection."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from dnabert_fno.data import sha256_file
from prepare_long_epi import canonical, component_groups


def main():
    out=Path('data/epi_diagnostic_holdout')
    assert not out.exists(), 'Already sealed; do not redraw this holdout'
    source_path=Path('data/long_epi_pilot/source.json')
    source=json.loads(source_path.read_text())
    raw={}
    for split,record in source['source_files'].items():
        assert sha256_file(record['path'])==record['sha256']
        raw[split]=list(csv.DictReader(gzip.open(record['path'],'rt')))
    all_rows=[dict(r,source_split=s,source_row=i) for s,rows in raw.items() for i,r in enumerate(rows)]
    maps={}
    for role,widths in [('enhancer',[600,1200,3000]),('promoter',[400,800,2000])]:
        seq=sorted({r[role] for r in all_rows})
        maps[role]=dict(zip(seq,component_groups(seq,widths)))
    old=[raw[r['source_split']][r['source_row']] for split in source['assignment'].values() for r in split]
    forbidden={role:{mapping[r[role]] for r in old} for role,mapping in maps.items()}
    eligible=[r for r in all_rows if all(maps[role][r[role]] not in forbidden[role] for role in maps)]
    unique={}
    for r in eligible:
        key=(canonical(r['enhancer']),canonical(r['promoter']))
        if key in unique:
            assert int(unique[key]['label'])==int(r['label'])
        else:
            unique[key]=r
    rng=np.random.default_rng(20260911); chosen=[]; counts={}
    for label in [0,1]:
        pool=[r for _,r in sorted(unique.items()) if int(r['label'])==label]
        counts[str(label)]=len(pool)
        assert len(pool)>=512
        chosen += [pool[i] for i in rng.choice(len(pool),512,replace=False)]
    chosen.sort(key=lambda r:(r['source_split'],r['source_row']))
    out.mkdir()
    with (out/'test.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['sequence','label'])
        w.writerows((r['enhancer']+r['promoter'],r['label']) for r in chosen)
    manifest={'seed':20260911,'rows':1024,'bp':5000,'eligible_labels':counts,
              'prior_source_manifest_sha256':sha256_file(source_path),
              'source_files':source['source_files'],'test_sha256':sha256_file(out/'test.csv'),
              'assignment':[{k:r[k] for k in ['source_split','source_row','label']} for r in chosen],
              'component_group_overlap_with_all_previous_train_dev_test':{role:len({mapping[r[role]] for r in chosen}&forbidden[role]) for role,mapping in maps.items()},
              'limitation':'New unused cohort within the same public GM12878 data; not an external dataset. Component aliases at all prior crop lengths excluded. Near-homology, cross-role genomic overlap and chromosome separation are not controlled.'}
    assert not any(manifest['component_group_overlap_with_all_previous_train_dev_test'].values())
    (out/'source.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['assignment','source_files']},indent=2))


if __name__=='__main__': main()
