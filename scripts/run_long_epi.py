"""Matched EPI context lengths: all training precedes all formal test metrics."""
import gc
import json
import warnings
from pathlib import Path

import torch

from dnabert_fno.cache import cache_features
from dnabert_fno.data import read_splits, TokenDataset, TokenCollator, collate_features, sha256_file
from dnabert_fno.loading import load_backbone
from dnabert_fno.runtime import environment, seed_everything
from dnabert_fno.study import atomic_json, stable_hash, fit_trial, evaluate_trial, aggregate


def main():
    warnings.filterwarnings('ignore', message='.*clean_up_tokenization_spaces.*')
    warnings.filterwarnings('ignore', message='Unable to import Triton.*')
    warnings.filterwarnings('ignore', message='Increasing alibi size.*')
    plan = json.loads(Path('configs/long_epi_pilot.json').read_text())
    base = json.loads(Path(plan['base_config']).read_text())
    cfg = {**base, **{k:plan[k] for k in ['variants','seeds','batch_size','gradient_accumulation',
                                        'length_bucketing','epochs','patience','learning_rate',
                                        'lora_head_learning_rate','max_tokens','inference_repeats','inference_warmup']}}
    cfg.update(dataset='GUEplus_GM12878_context_pilot', data_dir=plan['data_root'], output_dir=plan['output_dir'])
    # The stage-1 runner intentionally caps tokens at 512. This separate runner
    # uses the upstream dynamic ALiBi path verified by the 5kb/10kb GPU probes.
    assert cfg['max_tokens'] >= max(plan['lengths_bp']) + 2
    assert cfg['batch_size'] * cfg['gradient_accumulation'] == 32
    assert not cfg['length_bucketing'], 'Match epoch sample order across lengths'
    seed_everything(42); torch.set_num_threads(4)
    root = Path(plan['output_dir']); root.mkdir(parents=True, exist_ok=True)
    sources = sorted(Path('dnabert_fno').glob('*.py')) + [Path(__file__), Path('scripts/prepare_long_epi.py')]
    identity = {'plan': plan, 'config': cfg, 'environment': environment(cfg),
                'code_sha256': {str(p.resolve()):sha256_file(p) for p in sources},
                'source_manifest_sha256': sha256_file(Path(plan['data_root'])/'source.json'),
                'data_sha256': {str(n):{s:sha256_file(Path(plan['data_root'])/f'bp{n}'/f'{s}.csv')
                                      for s in ['train','dev','test']} for n in plan['lengths_bp']}}
    study_id = stable_hash(identity)
    if (root/'protocol.json').exists():
        assert json.loads((root/'protocol.json').read_text()) == identity, 'Protocol changed; use a new output directory'
    else:
        atomic_json(root/'protocol.json', identity)
    contexts, all_selected = [], []
    labels_reference = None
    for n in plan['lengths_bp']:
        subroot = root/f'bp{n}'; subroot.mkdir(exist_ok=True)
        local = {**cfg, 'data_dir':str(Path(plan['data_root'])/f'bp{n}'), 'output_dir':str(subroot),
                 'dataset':f'GUEplus_GM12878_disjoint_pilot_bp{n}'}
        atomic_json(root/'status.json', {'phase':'prepare_features','bp':n,'study_id':study_id})
        splits, audit = read_splits(local['data_dir'])
        labels = {s:[r['label'] for r in rows] for s,rows in splits.items()}
        if labels_reference is None:
            labels_reference = labels
        assert labels == labels_reference
        assert all(len(r['sequence']) == n for rows in splits.values() for r in rows)
        assert not any(x['including_reverse_complement'] for x in audit['overlap'].values())
        tokenizer, frozen, info = load_backbone(local, offline=True)
        collator = TokenCollator(tokenizer.pad_token_id)
        tokens = {s:TokenDataset(rows, tokenizer, local['max_tokens']) for s,rows in splits.items()}
        for s in tokens:
            audit[s]['tokenization'] = tokens[s].stats
            assert tokens[s].stats['truncated_rows'] == 0
        atomic_json(subroot/'data_audit.json',audit); atomic_json(subroot/'weight_loading.json',info)
        print(f'CONTEXT {n}bp: token maxima '+str({s:tokens[s].stats['tokens_max_before_truncation'] for s in tokens}),flush=True)
        frozen.cuda()
        features, cache = cache_features(local,frozen,tokens,collator,audit,environment(local))
        atomic_json(subroot/'feature_cache.json',cache)
        del frozen; gc.collect(); torch.cuda.empty_cache()
        chosen, selection = [], {}
        for variant in local['variants']:
            selection[variant] = {'lr':local['learning_rate'], 'selection':plan['selection']}
            online = variant.startswith('lora')
            for seed in local['seeds']:
                atomic_json(root/'status.json',{'phase':'training','bp':n,'variant':variant,'seed':seed,
                                               'completed_trials':len(all_selected),'study_id':study_id})
                record = fit_trial(subroot, stable_hash({'study_id':study_id,'bp':n}), local,variant,seed,
                                   local['learning_rate'],tokens if online else features,
                                   collator if online else collate_features)
                chosen.append(record); all_selected.append({'bp':n,**record})
        contexts.append((subroot,local,tokens,features,collator,chosen,selection,cache))
    for seed in cfg['seeds']:
        assert len({r['head_initial_sha256'] for r in all_selected if r['settings']['seed']==seed})==1
        assert len({r['lora_initial_sha256'] for r in all_selected if r['settings']['seed']==seed and r['settings']['variant']=='lora'})==1
    assert len(all_selected)==len(plan['lengths_bp'])*len(cfg['variants'])*len(cfg['seeds'])
    atomic_json(root/'training_complete.json',{'study_id':study_id,'selected_runs':all_selected})
    output = {}
    for subroot,local,tokens,features,collator,chosen,selection,cache in contexts:
        atomic_json(subroot/'training_complete.json',{'selected_runs':chosen,'selection':selection})
        atomic_json(root/'status.json',{'phase':'test_evaluation','context':subroot.name,'study_id':study_id})
        results=[evaluate_trial(subroot,local,r,tokens,features,collator) for r in chosen]
        aggregate(subroot,local,results,selection,cache)
        # Remove generic seven-arm notes from the reused aggregation writer.
        text = (subroot/'summary.md').read_text(encoding='utf-8')
        text = '\n'.join(line for line in text.splitlines() if not line.startswith(('MLP/CNN parameter','LoRA+FNO has more')))
        (subroot/'summary.md').write_text(text+'\n\nFixed LR from previous promoter study; no LR search on this EPI pilot.\n',encoding='utf-8')
        output[subroot.name]=json.loads((subroot/'summary.json').read_text())
        atomic_json(subroot/'status.json',{'phase':'complete','selected_runs':len(chosen)})
    atomic_json(root/'summary.json',{'study_id':study_id,'plan':plan,'by_length':output})
    atomic_json(root/'status.json',{'phase':'complete','selected_runs':len(all_selected),'study_id':study_id})
    print('LONG EPI PILOT COMPLETE',flush=True)


if __name__ == '__main__':
    main()
