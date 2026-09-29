"""Five chromosome folds; dev-only equal-budget residual mixer comparison."""
import gc
import json
import sys
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dnabert_fno import study
from dnabert_fno.data import sha256_file
from dnabert_fno.adaptation import load_trainable
from dnabert_fno.runtime import seed_everything,environment
from dnabert_fno.train import save_predictions
from crispr_ranking_models import train_one_epoch,ranking_metrics
import matched_cv_models as models
from cache_matched_grid import build,gpu_load


def read(p):return json.loads(Path(p).read_text())


class LogNames:
    def __init__(self,stream):self.stream=stream
    def write(self,s):return self.stream.write(s.replace('dev_MCC=','dev_matched_concordance='))
    def flush(self):return self.stream.flush()


def main():
    warnings.filterwarnings('ignore',message='.*clean_up_tokenization_spaces.*')
    warnings.filterwarnings('ignore',message='Unable to import Triton.*')
    warnings.filterwarnings('ignore',message='Increasing alibi size.*')
    sys.stdout=LogNames(sys.stdout);torch.set_num_threads(4);seed_everything(42)
    plan=read('configs/matched_cv.json');data=Path(plan['data_dir']);source=read(data/'source.json')
    for name,digest in source['files_sha256'].items():assert sha256_file(data/f'{name}.json')==digest
    rows=read(data/'rows.json');pairs=read(data/'matched_pairs.json');prior=read(plan['base_protocol'])
    base={**prior['config'],**{k:plan[k] for k in ['data_dir','output_dir','epochs','patience','batch_size','gradient_accumulation','selection_metric']},
          'dataset':'K562_distance_matched_5fold_mixer_comparison'}
    root=Path(plan['output_dir']);root.mkdir(parents=True,exist_ok=True)
    files=list(Path('dnabert_fno').glob('*.py'))+[Path(p) for p in ['scripts/run_matched_cv.py','scripts/matched_cv_models.py','scripts/cache_matched_grid.py',
          'scripts/prepare_matched_cv.py','scripts/crispr_ranking_models.py','scripts/epi_information_models.py']]
    identity={'plan':plan,'base_config':base,'environment':environment(base),'data_manifest_sha256':sha256_file(data/'source.json'),
              'code_sha256':{str(p.resolve()):sha256_file(p) for p in files},'protocol_document_sha256':sha256_file('docs/MATCHED_CV_PROTOCOL.md'),
              'widths':{k:models.width_for(k) for k in plan['variants']},
              'legacy_field_note':'best_dev_mcc holds configured matched_concordance; no MCC-based selection.'}
    study_id=study.stable_hash(identity)
    if (root/'protocol.json').exists():assert read(root/'protocol.json')==identity
    else:study.atomic_json(root/'protocol.json',identity)
    if (root/'status.json').exists() and read(root/'status.json')['phase']=='complete':print('Already complete');return
    if not (root/'started.json').exists():study.atomic_json(root/'started.json',{'unix':time.time()})
    started=read(root/'started.json')['unix'];study.atomic_json(root/'status.json',{'phase':'grid_features'})
    seqs,cache=build(base,root/'grid_cache',rows);index={s:i for i,s in enumerate(seqs)}
    models.CACHE=gpu_load(root/'grid_cache/grid.npy',base['device'])
    study.make_model=models.make_model;study.train_one_epoch=train_one_epoch;study.evaluate_study=models.evaluate
    trained=[];final=[];selection={};foldconfigs={}
    for fold in plan['folds']:
        dev=(fold+1)%5
        splits={'train':[r for r in rows if r['fold'] not in [fold,dev]],'dev':[r for r in rows if r['fold']==dev]}
        n=len(splits['train']);pos=sum(r['label'] for r in splits['train']);dist=np.log10([r['distance'] for r in splits['train']])
        cfg={**base,'class_weights':[n/(2*(n-pos)),n/(2*pos)],'distance_mean':float(dist.mean()),'distance_sd':float(dist.std()),
             'outer_test_fold':fold,'inner_dev_fold':dev}
        foldconfigs[str(fold)]=cfg;datasets={s:models.Dataset(rs,index,cfg['distance_mean'],cfg['distance_sd'],pairs) for s,rs in splits.items()}
        selection[str(fold)]={}
        for variant in plan['variants']:
            candidates=[]
            for lr in plan['learning_rates']:
                records=[]
                for seed in plan['search_seeds']:
                    study.atomic_json(root/'status.json',{'phase':'training','fold':fold,'variant':variant,'lr':lr,'seed':seed,
                                                        'completed_trials':len(trained),'expected_trials':105})
                    r=study.fit_trial(root/f'fold{fold}',study_id,cfg,variant,seed,lr,datasets,models.collate)
                    records.append(r);trained.append(r)
                candidates.append({'lr':lr,'records':records,'mean_best_dev_matched_concordance':float(np.mean([r['best_dev_mcc'] for r in records]))})
            best=sorted(candidates,key=lambda c:(-c['mean_best_dev_matched_concordance'],c['lr']))[0]
            selection[str(fold)][variant]={'candidates':candidates,'selected':best}
            r=study.fit_trial(root/f'fold{fold}',study_id,cfg,variant,44,best['lr'],datasets,models.collate);trained.append(r)
            final.extend({'fold':fold,'config':cfg,'training':r} for r in best['records']+[r])
        x=np.log10([r['distance'] for r in splits['train']])[:,None];y=np.array([r['label'] for r in splits['train']])
        simple=make_pipeline(StandardScaler(),LogisticRegression(C=.001,class_weight='balanced',solver='lbfgs',max_iter=5000,tol=1e-6,random_state=42))
        simple.fit(x,y);path=root/f'fold{fold}'/'distance.joblib';joblib.dump(simple,path)
        selection[str(fold)]['distance']={'C':.001,'checkpoint_sha256':sha256_file(path),'test_used':False}
    assert len(trained)==105 and len(final)==45
    for fold in plan['folds']:
        for seed in plan['final_seeds']:
            assert len({r['training']['head_initial_sha256'] for r in final if r['fold']==fold and r['training']['settings']['seed']==seed})==1
    study.atomic_json(root/'selection.json',{'study_id':study_id,'folds':selection,'test_used':False})
    study.atomic_json(root/'training_complete.json',{'study_id':study_id,'all_training':trained,'final':final,'fold_configs':foldconfigs,'distance_fits':5})
    results=[]
    for entry in final:
        fold=entry['fold'];cfg=entry['config'];r=entry['training'];variant=r['settings']['variant'];seed=r['settings']['seed']
        test=[x for x in rows if x['fold']==fold];dest=root/'results'/f'fold{fold}'/f'{variant}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
        study.atomic_json(root/'status.json',{'phase':'evaluation','fold':fold,'variant':variant,'seed':seed,'completed_trials':105})
        model=models.make_model(cfg,variant,seed);checkpoint=Path(r['path'])/'best.pt'
        load_trainable(model,torch.load(checkpoint,map_location='cpu',weights_only=True)['trainable'])
        ds=models.Dataset(test,index,cfg['distance_mean'],cfg['distance_sd'],pairs)
        metrics,p,seconds=models.evaluate(model,ds,models.collate,cfg);p['global_id']=[x['global_id'] for x in test]
        save_predictions(dest/'predictions.csv',p)
        result={'fold':fold,'model':variant,'seed':seed,'metrics':metrics,'seconds':seconds,'training':r,
                'checkpoint_sha256':sha256_file(checkpoint),'prediction_sha256':sha256_file(dest/'predictions.csv')}
        study.atomic_json(dest/'result.json',result);results.append(result)
        print(f'TEST fold={fold} {variant} seed={seed}: matched={metrics["matched_concordance"]:.4f}',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    for fold in plan['folds']:
        test=[r for r in rows if r['fold']==fold];simple=joblib.load(root/f'fold{fold}'/'distance.joblib')
        p=simple.predict_proba(np.log10([r['distance'] for r in test])[:,None])[:,1]
        metrics,_=ranking_metrics(test,p);foldpairs=[p for p in pairs if p['fold']==fold]
        metrics['matched_concordance']=models.matched_concordance(foldpairs,{r['global_id']:p[i] for i,r in enumerate(test)})[0]
        dest=root/'results'/f'fold{fold}'/'distance';dest.mkdir(parents=True,exist_ok=True)
        save_predictions(dest/'predictions.csv',{'row_id':list(range(len(test))),'label':[r['label'] for r in test],'probability':p.tolist(),'global_id':[r['global_id'] for r in test]})
        result={'fold':fold,'model':'distance','seed':None,'metrics':metrics,'prediction_sha256':sha256_file(dest/'predictions.csv'),
                'checkpoint_sha256':sha256_file(root/f'fold{fold}'/'distance.joblib')}
        study.atomic_json(dest/'result.json',result);results.append(result)
    study.atomic_json(root/'summary.json',{'study_id':study_id,'results':results})
    study.atomic_json(root/'status.json',{'phase':'complete','study_id':study_id,'neural_trials':105,'distance_fits':5,
                                        'evaluations':len(results),'elapsed_seconds':time.time()-started})
    print('MATCHED CV COMPLETE',flush=True)


if __name__=='__main__':main()
