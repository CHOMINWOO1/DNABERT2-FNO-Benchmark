"""Verify paired EPI crops, provenance, and every final saved probability file."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from verify_study import verify_predictions
from prepare_long_epi import canonical, center


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('runs/long_epi_pilot'); status = read(root/'status.json')
    assert status['phase'] == 'complete'
    protocol = read(root/'protocol.json'); plan = protocol['plan']
    for path, sha in protocol['code_sha256'].items():
        assert digest(path) == sha, path
    data_root = Path(plan['data_root']); source = read(data_root/'source.json')
    assert digest(data_root/'source.json') == protocol['source_manifest_sha256']
    raw = {}
    for split, record in source['source_files'].items():
        assert digest(record['path']) == record['sha256']
        raw[split] = list(csv.DictReader(gzip.open(record['path'], 'rt')))
    summaries = read(root/'summary.json')['by_length']; tested = 0
    selected = read(root/'training_complete.json')['selected_runs']
    selected_by_key = {(r['bp'],r['settings']['variant'],r['settings']['seed']):r for r in selected}
    paired = {}; reference_labels = None; component_counts = {}
    all_records = []
    for length in plan['lengths_bp']:
        folder = root/f'bp{length}'; data = data_root/f'bp{length}'
        views, labels_by_split = {}, {}
        for split in ['train','dev','test']:
            assert digest(data/f'{split}.csv') == protocol['data_sha256'][str(length)][split]
            with (data/f'{split}.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            expected = []
            for assignment in source['assignment'][split]:
                row = raw[assignment['source_split']][assignment['source_row']]
                e, p = center(row['enhancer'].upper(), length*3//5), center(row['promoter'].upper(), length*2//5)
                expected.append((e,p,int(row['label'])))
            assert [(r['sequence'],int(r['label'])) for r in rows] == [(e+p,y) for e,p,y in expected]
            views[split] = expected; labels_by_split[split] = [y for _,_,y in expected]
        if reference_labels is None:
            reference_labels = labels_by_split
        assert labels_by_split == reference_labels
        component_counts[str(length)] = {s:{'pairs':len(rows),
            'unique_enhancers_exact_rc':len({canonical(r[0]) for r in rows}),
            'unique_promoters_exact_rc':len({canonical(r[1]) for r in rows})} for s,rows in views.items()}
        for a,b in [('train','dev'),('train','test'),('dev','test')]:
            for i in [0,1]:
                assert not ({canonical(r[i]) for r in views[a]} & {canonical(r[i]) for r in views[b]})
        labels = np.array(labels_by_split['test'])
        records = [read(p) for p in (folder/'results').glob('*/result.json')]
        assert {(r['variant'],r['seed']) for r in records} == {(v,s) for v in plan['variants'] for s in plan['seeds']}
        for r in records:
            path = folder/'results'/f"{r['variant']}_seed{r['seed']}"
            assert r['training']['frozen_weights_unchanged']
            assert not r['training']['test_evaluated']
            chosen = selected_by_key[(length,r['variant'],r['seed'])]
            assert r['training'] == {k:v for k,v in chosen.items() if k!='bp'}
            assert digest(Path(r['training']['path'])/'best.pt') == r['best_checkpoint_sha256']
            p = verify_predictions(path/'test_predictions.csv',labels,r['test'])
            if r['lora_merge']:
                merge = r['lora_merge']
                q = verify_predictions(path/'merged_test_predictions.csv',labels,merge['test'])
                assert np.isclose(np.max(np.abs(p-q)),merge['max_abs_probability_difference'],atol=1e-12)
                assert np.isclose(np.mean(np.abs(p-q)),merge['mean_abs_probability_difference'],atol=1e-12)
                assert int(((p>=.5)!=(q>=.5)).sum()) == merge['changed_class_predictions']
            tested += 1
        for variant, report in summaries[f'bp{length}']['summary'].items():
            subset = [r for r in records if r['variant']==variant]
            for metric, values in report['metrics'].items():
                observed = [r['test'][metric] for r in subset]
                assert np.isclose(np.mean(observed),values['mean'],atol=1e-12)
                assert np.isclose(np.std(observed,ddof=1),values['sd'],atol=1e-12)
            paired.setdefault(variant,{})[str(length)] = {str(r['seed']):r['test']['mcc'] for r in subset}
        all_records.extend(records)
    for seed in plan['seeds']:
        assert len({r['training']['head_initial_sha256'] for r in all_records if r['seed']==seed})==1
    expected_runs = len(plan['lengths_bp'])*len(plan['variants'])*len(plan['seeds'])
    assert tested == expected_runs
    assert len(selected) == len(selected_by_key) == expected_runs
    assert len(list(root.glob('bp*/trials/*/training.json'))) == expected_runs
    changes = {}
    a,b = str(min(plan['lengths_bp'])),str(max(plan['lengths_bp']))
    for variant, values in paired.items():
        delta = {s:values[b][s]-values[a][s] for s in values[a]}
        changes[variant] = {'per_seed':delta,'mean':float(np.mean(list(delta.values()))),
                            'sd':float(np.std(list(delta.values()),ddof=1)),'positive_seeds':sum(x>0 for x in delta.values())}
    differences = {}
    for length in plan['lengths_bp']:
        differences[str(length)] = {}
        for control in ['cnn','lora']:
            delta = {s:paired['fno'][str(length)][s]-paired[control][str(length)][s] for s in paired['fno'][str(length)]}
            differences[str(length)][f'fno_minus_{control}'] = {'per_seed':delta,'mean':float(np.mean(list(delta.values()))),
                'sd':float(np.std(list(delta.values()),ddof=1)),'positive_seeds':sum(x>0 for x in delta.values())}
    result = {'passed':True,'study_id':status['study_id'],'runs':tested,'test_rows_per_run':len(reference_labels['test']),
              'checks':['source/data/checkpoint hashes','exact reconstruction from original enhancer/promoter pairs',
                        'paired sample/label order across lengths','component exact/RC disjointness',
                        'all six metrics recomputed, including independent confusion-count MCC',
                        'same initial head per seed','frozen weights unchanged','summary mean/sample SD'],
              'component_counts':component_counts,
              'mcc_5000_minus_1000':changes,'paired_mcc_by_length':differences}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
