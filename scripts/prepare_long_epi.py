"""Fixed paired-context EPI pilot, with disjoint exact/RC component groups."""
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


LENGTHS = [1000, 2000, 5000]
REVISION = 'f1290ccd49d4c7f8014cac47c93779aed06a294b'


def canonical(s):
    return min(s, s.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1])


def center(s, n):
    assert len(s) >= n
    start = (len(s) - n) // 2
    return s[start:start+n]


def component_groups(sequences, widths):
    # A full component and any identical/RC cropped view always share a split.
    parent = list(range(len(sequences)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i, sequence in enumerate(sequences):
        for width in widths:
            key = canonical(center(sequence, width))
            if key in seen:
                a, b = find(i), find(seen[key]); parent[max(a, b)] = min(a, b)
            else:
                seen[key] = i
    return [find(i) for i in range(len(sequences))]


def main():
    source = Path('data/GUE_v2_source'); output = Path('data/long_epi_pilot')
    assert not output.exists(), 'Use a new directory to change the data protocol'
    rows, files = [], {}
    for split, count in [('train', 10000), ('val', 2000), ('test', 2000)]:
        path = source / f'EPI_GM12878_{split}.csv.gz'
        records = list(csv.DictReader(gzip.open(path, 'rt')))
        assert len(records) == count
        files[split] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'url': f'https://huggingface.co/datasets/genomic-benchmarks/GUE_v2/resolve/{REVISION}/EPI_GM12878/{path.name}'}
        for index, row in enumerate(records):
            e, p = row['enhancer'].upper(), row['promoter'].upper()
            assert len(e) == 3000 and len(p) == 2000 and not (set(e+p)-set('ACGTN'))
            rows.append({'enhancer': e, 'promoter': p, 'label': int(row['label']),
                         'source_split': split, 'source_row': index})
    pair_groups = defaultdict(list)
    for row in rows:
        pair_groups[(canonical(row['enhancer']), canonical(row['promoter']))].append(row)
    conflicts = [v for v in pair_groups.values() if len({r['label'] for r in v}) > 1]
    unique = [v[0] for k,v in sorted(pair_groups.items()) if len({r['label'] for r in v}) == 1]
    enhancers = sorted({r['enhancer'] for r in unique}); promoters = sorted({r['promoter'] for r in unique})
    eg = dict(zip(enhancers, component_groups(enhancers, [n*3//5 for n in LENGTHS])))
    pg = dict(zip(promoters, component_groups(promoters, [n*2//5 for n in LENGTHS])))
    rng = np.random.default_rng(20260909)
    names = ['train', 'dev', 'test']
    e_split = dict(zip(sorted(set(eg.values())), rng.choice(names, len(set(eg.values())), p=[.6,.2,.2])))
    p_split = dict(zip(sorted(set(pg.values())), rng.choice(names, len(set(pg.values())), p=[.6,.2,.2])))
    pools = {s: [] for s in names}
    for row in unique:
        a, b = e_split[eg[row['enhancer']]], p_split[pg[row['promoter']]]
        if a == b:
            pools[a].append(row)
    print('Eligible label counts:', {s: dict(Counter(r['label'] for r in v)) for s,v in pools.items()}, flush=True)
    selected = {}
    for split, n in [('train', 1024), ('dev', 256), ('test', 256)]:
        chosen = []
        for label in [0, 1]:
            candidates = [r for r in pools[split] if r['label'] == label]
            assert len(candidates) >= n//2, (split, label, len(candidates))
            chosen.extend(candidates[i] for i in rng.permutation(len(candidates))[:n//2])
        selected[split] = sorted(chosen, key=lambda r: (r['source_split'], r['source_row']))
    output.mkdir(parents=True)
    manifest = {'dataset': 'GUE+ GM12878 EPI, public GUE_v2 mirror', 'revision': REVISION,
                'source_files': files, 'official_reference': 'https://arxiv.org/html/2306.15006v2',
                'official_archive_byte_identity_verified': False, 'original_rows': len(rows),
                'unique_consistent_pairs': len(unique), 'conflicting_pair_groups': len(conflicts),
                'split_seed': 20260909, 'lengths_bp': LENGTHS, 'subset_sizes': {s:len(v) for s,v in selected.items()},
                'eligible_label_counts': {s:dict(Counter(r['label'] for r in v)) for s,v in pools.items()},
                'method': 'Assign enhancer and promoter exact/RC crop-alias groups independently 60/20/20; retain edges only when both assignments agree; balanced fixed subsample.',
                'crop': 'Centered enhancer and promoter views at 3:2 length ratio, concatenated in enhancer-promoter order; no padding, repetition, or genomic intervening sequence.',
                'limitation': 'New small split, not official scores. Near-homology and chromosome overlap are not controlled; shared component exact/RC identities are controlled at every tested length.',
                'assignment': {s:[{k:r[k] for k in ['source_split','source_row','label']} for r in v] for s,v in selected.items()},
                'audit': {}}
    for length in LENGTHS:
        folder = output / f'bp{length}'; folder.mkdir()
        views = {}
        for split, records in selected.items():
            views[split] = [(center(r['enhancer'], length*3//5), center(r['promoter'], length*2//5), r['label']) for r in records]
            with (folder / f'{split}.csv').open('w', newline='') as f:
                writer = csv.writer(f); writer.writerow(['sequence', 'label'])
                writer.writerows((e+p,y) for e,p,y in views[split])
        overlap = {}
        for a,b in [('train','dev'),('train','test'),('dev','test')]:
            measures = {}
            for name, index in [('enhancer',0),('promoter',1)]:
                measures[name] = len({canonical(r[index]) for r in views[a]} & {canonical(r[index]) for r in views[b]})
            measures['concatenated'] = len({canonical(r[0]+r[1]) for r in views[a]} & {canonical(r[0]+r[1]) for r in views[b]})
            assert not any(measures.values()); overlap[a+'_'+b] = measures
        manifest['audit'][str(length)] = {'overlap': overlap, 'sha256': {s:hashlib.sha256((folder/f'{s}.csv').read_bytes()).hexdigest() for s in names}}
    (output / 'source.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Prepared', output, manifest['subset_sizes'])


if __name__ == '__main__':
    main()
