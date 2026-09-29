"""Equal-budget dev-only architecture search, matched pooling, sealed holdout."""
import csv
import gc
import json
import time
import warnings
from pathlib import Path
import numpy as np
import torch
from dnabert_fno import study
from dnabert_fno.adaptation import load_trainable
from dnabert_fno.cache import cache_features
from dnabert_fno.data import FeatureDataset,TokenDataset,TokenCollator,collate_features,sha256_file
from dnabert_fno.loading import load_backbone
from dnabert_fno.runtime import environment,seed_everything
from dnabert_fno.train import save_predictions
from diagnostic_models import make_model,matched_width,count_formula


def read(path): return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as f:
        return [dict(sequence=r['sequence'],label=int(r['label']),row_id=i) for i,r in enumerate(csv.DictReader(f))]


def cached_old(split,base):
    reports=read('runs/long_epi_pilot/bp5000/feature_cache.json')
    report=reports[split]; identity=report['identity']
    assert identity['environment']==environment(base), 'Cached encoder environment changed'
    key=study.stable_hash(identity)[:20]
    directory=Path(base['cache_dir'])/'features'/key/split
    assert read(directory/'manifest.json')['identity']==identity
    source=Path('data/long_epi_pilot/bp5000')/f'{split}.csv'
    assert sha256_file(source)==identity['splits'][split]['file_sha256']
    assert sha256_file(source)==read('runs/long_epi_pilot/protocol.json')['data_sha256']['5000'][split]
    items=[{'label':r['label'],'row_id':r['row_id']} for r in rows(source)]
    ds=FeatureDataset(directory,items)
    assert len(ds.offsets)==len(items)+1 and ds.offsets[-1]==report['shape'][0]
    return ds


def main():
    warnings.filterwarnings('ignore',message='.*clean_up_tokenization_spaces.*')
    warnings.filterwarnings('ignore',message='Unable to import Triton.*')
    warnings.filterwarnings('ignore',message='Increasing alibi size.*')
    torch.set_num_threads(4); seed_everything(42)
    plan=read('configs/epi_diagnostic.json'); prior=read(plan['source_protocol'])
    base={**prior['config'],**{k:plan[k] for k in ['epochs','patience','batch_size','gradient_accumulation']},
          'dataset':'GM12878_5kb_diagnostic','data_dir':'data/long_epi_pilot/bp5000','output_dir':plan['output_dir']}
    root=Path(plan['output_dir']);root.mkdir(parents=True,exist_ok=True)
    files=list(Path('dnabert_fno').glob('*.py'))+[Path(__file__),Path('scripts/diagnostic_models.py'),Path('scripts/prepare_diagnostic_holdout.py')]
    for path,digest in prior['code_sha256'].items():
        assert sha256_file(path)==digest, 'Prior experiment source changed'
    candidates=[]
    for variant,values in [('fno',plan['fno_modes']),('cnn',plan['cnn_kernels'])]:
        for value in values:
            w=matched_width(variant,value,plan['target_trainable_parameters'])
            for lr in plan['learning_rates']:
                cfg={**base,'width':w,'cnn_width':w,'modes':value if variant=='fno' else 16,
                     'diagnostic_kernel':value if variant=='cnn' else 3,'diagnostic_pooling':'mean'}
                candidates.append({'arm':f'{variant}_s{value}_lr{lr:g}','variant':variant,'structure':value,
                                   'lr':lr,'width':w,'params':count_formula(variant,w,value),'config':cfg})
    identity={'plan':plan,'base_config':base,'candidates':candidates,'environment':environment(base),
              'code_sha256':{str(p.resolve()):sha256_file(p) for p in files},
              'prior_protocol_sha256':sha256_file(plan['source_protocol']),
              'holdout_manifest_sha256':sha256_file('data/epi_diagnostic_holdout/source.json'),
              'holdout_data_sha256':sha256_file('data/epi_diagnostic_holdout/test.csv'),
              'factory':'Process-local study.make_model = scripts.diagnostic_models.make_model; original source files unchanged.'}
    study_id=study.stable_hash(identity)
    if (root/'protocol.json').exists(): assert read(root/'protocol.json')==identity, 'Protocol changed'
    else: study.atomic_json(root/'protocol.json',identity)
    if (root/'status.json').exists() and read(root/'status.json')['phase']=='complete':
        print('Already complete');return
    started=read(root/'started.json') if (root/'started.json').exists() else {'unix':time.time()}
    study.atomic_json(root/'started.json',started)
    datasets={s:cached_old(s,base) for s in ['train','dev']}
    original_factory=study.make_model; study.make_model=make_model
    trained=[]; search=[]
    def fit(candidate,seed,pooling='mean'):
        cfg={**candidate['config'],'diagnostic_pooling':pooling}
        arm=candidate['arm']+('' if pooling=='mean' else '_attention')
        study.atomic_json(root/'status.json',{'phase':'training','arm':arm,'seed':seed,'completed_trials':len(trained),'expected_trials':44})
        record=study.fit_trial(root/'arms'/arm,study.stable_hash({'study_id':study_id,'arm':arm}),cfg,
                               candidate['variant'],seed,candidate['lr'],datasets,collate_features)
        trained.append({'arm':arm,'pooling':pooling,'config':cfg,**record})
        return record
    for candidate in candidates:
        records=[fit(candidate,s) for s in plan['search_seeds']]
        search.append({**candidate,'records':records,'mean_best_dev_mcc':float(np.mean([r['best_dev_mcc'] for r in records]))})
    selection={}
    for variant in ['fno','cnn']:
        selection[variant]=sorted([r for r in search if r['variant']==variant],key=lambda r:(-r['mean_best_dev_mcc'],r['lr'],r['structure']))[0]
    study.atomic_json(root/'selection.json',{'search':search,'selected':selection,'test_used':False})
    final=[]
    for variant,candidate in selection.items():
        mean_records=candidate['records']+[fit(candidate,44)]
        for r in mean_records: final.append({'variant':variant,'pooling':'mean','config':candidate['config'],'training':r})
        for seed in plan['final_seeds']:
            r=fit(candidate,seed,'attention')
            final.append({'variant':variant,'pooling':'attention','config':{**candidate['config'],'diagnostic_pooling':'attention'},'training':r})
    assert len(trained)==44 and len(final)==12
    for seed in plan['final_seeds']:
        assert len({r['training']['head_initial_sha256'] for r in final if r['training']['settings']['seed']==seed})==1
    study.atomic_json(root/'training_complete.json',{'study_id':study_id,'all_training':trained,'final':final})
    study.atomic_json(root/'status.json',{'phase':'prepare_holdout_features','completed_trials':44})
    tokenizer,encoder,_=load_backbone(base,offline=True)
    tokens=TokenDataset(rows('data/epi_diagnostic_holdout/test.csv'),tokenizer,base['max_tokens'])
    assert tokens.stats['truncated_rows']==0
    collator=TokenCollator(tokenizer.pad_token_id)
    encoder.cuda()
    fresh,cache_report=cache_features(base,encoder,{'holdout':tokens},collator,
        {'holdout':{'sha256':identity['holdout_data_sha256']}},environment(base))
    study.atomic_json(root/'holdout_cache.json',{'cache':cache_report,'tokenization':tokens.stats})
    del encoder;gc.collect();torch.cuda.empty_cache()
    testsets={'previous_test':cached_old('test',base),'new_holdout':fresh['holdout']}
    results=[]
    def evaluate(label,variant,seed,cfg,training,factory,split,dataset,collate):
        dest=root/'results'/split/f'{label}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists():
            result=read(dest/'result.json')
            assert result['checkpoint_sha256']==sha256_file(Path(training['path'])/'best.pt')
            return result
        study.atomic_json(root/'status.json',{'phase':'evaluation','split':split,'model':label,'seed':seed,'completed_trials':44})
        model=factory(cfg,variant,seed)
        checkpoint=Path(training['path'])/'best.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        load_trainable(model,state['trainable'])
        metrics,predictions,seconds=study.evaluate_study(model,dataset,collate,cfg)
        save_predictions(dest/'predictions.csv',predictions)
        result={'model':label,'variant':variant,'seed':seed,'split':split,'metrics':metrics,'seconds':seconds,
                'training':training,'checkpoint_sha256':sha256_file(checkpoint),'config':cfg}
        study.atomic_json(dest/'result.json',result)
        print(f"EVAL {split} {label} seed={seed}: MCC={metrics['mcc']:.4f}",flush=True)
        del model,state;gc.collect();torch.cuda.empty_cache()
        return result
    for record in final:
        seed=record['training']['settings']['seed'];variant=record['variant'];label=variant+'_'+record['pooling']
        for split,dataset in testsets.items():
            results.append(evaluate(label,variant,seed,record['config'],record['training'],make_model,split,dataset,collate_features))
    # Historical untuned anchors, with no tuning or checkpoint reselection on the new cohort.
    for variant in ['cnn','fno','lora']:
        for seed in plan['final_seeds']:
            old=read(f'runs/long_epi_pilot/bp5000/results/{variant}_seed{seed}/result.json')
            cfg={**prior['config'],'data_dir':base['data_dir']}
            results.append(evaluate('prior_'+variant,variant,seed,cfg,old['training'],original_factory,'new_holdout',
                                    tokens if variant=='lora' else fresh['holdout'],collator if variant=='lora' else collate_features))
    summary={}
    for split in testsets:
        summary[split]={}
        for label in sorted({r['model'] for r in results if r['split']==split}):
            group=[r for r in results if r['split']==split and r['model']==label]
            summary[split][label]={metric:{'mean':float(np.mean([r['metrics'][metric] for r in group])),
                 'sd':float(np.std([r['metrics'][metric] for r in group],ddof=1)),
                 'per_seed':{str(r['seed']):r['metrics'][metric] for r in group}} for metric in group[0]['metrics']}
    study.atomic_json(root/'summary.json',{'study_id':study_id,'selection':{k:{n:v[n] for n in ['arm','structure','width','lr','mean_best_dev_mcc']} for k,v in selection.items()},'summary':summary,'results':results})
    study.atomic_json(root/'status.json',{'phase':'complete','training_trials':44,'evaluations':len(results),'elapsed_seconds':time.time()-started['unix'],'study_id':study_id})
    print('EPI DIAGNOSTIC COMPLETE',flush=True)


if __name__=='__main__': main()
