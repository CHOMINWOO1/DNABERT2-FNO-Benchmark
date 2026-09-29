"""Chromosome-held-out CRISPR candidate ranking with a frozen DNABERT-2."""
import gc
import json
import sys
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from dnabert_fno import study
from dnabert_fno.adaptation import load_trainable
from dnabert_fno.data import TokenDataset,TokenCollator,collate_features,sha256_file
from dnabert_fno.loading import load_backbone
from dnabert_fno.runtime import environment,seed_everything,amp_context,move_batch,timed_start,elapsed,reset_peak,peak_memory
from dnabert_fno.train import save_predictions
from crispr_ranking_models import VARIANTS,make_model,train_one_epoch,evaluate,VectorDataset,design,simple_design,ranking_metrics


def read(p):return json.loads(Path(p).read_text())


def features(root,cfg,study_id,splitrows,batch_size):
    folder=root/'features';folder.mkdir(exist_ok=True);answer={};pending=[]
    for split,rows in splitrows.items():
        dest=folder/f'{split}.npz';meta=folder/f'{split}.json'
        if meta.exists():
            m=read(meta);assert m['study_id']==study_id and m['sha256']==sha256_file(dest)
            with np.load(dest) as z:answer[split]={k:z[k].copy() for k in z.files}
        else:pending.append(split)
    if not pending:return answer
    tokenizer,encoder,_=load_backbone(cfg,offline=True);before=study.frozen_hash(encoder)
    assert all(not p.requires_grad for p in encoder.parameters());encoder.cuda().eval()
    collator=TokenCollator(tokenizer.pad_token_id)
    for split in pending:
        rows=splitrows[split];seqs=sorted({r[k] for r in rows for k in ['enhancer','promoter']})
        tokens=TokenDataset([{'sequence':s,'label':0,'row_id':i} for i,s in enumerate(seqs)],tokenizer,cfg['max_tokens'])
        assert tokens.stats['truncated_rows']==0
        loader=DataLoader(tokens,batch_size=batch_size,shuffle=False,collate_fn=collator,num_workers=0)
        study.atomic_json(root/'status.json',{'phase':'feature_extraction','split':split,'unique_sequences':len(seqs)})
        reset_peak(cfg);start=timed_start(cfg);chunks=[]
        for i,batch in enumerate(loader):
            batch=move_batch(batch,cfg)
            with torch.no_grad(),amp_context(cfg):h=encoder(batch['input_ids'],batch['attention_mask'])
            mask=batch['content_mask'].float();pooled=(h.float()*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)
            chunks.append(pooled.cpu().numpy())
            if i%100==0:print(f'FEATURE {split}: {min((i+1)*batch_size,len(seqs))}/{len(seqs)}',flush=True)
        vectors=np.concatenate(chunks);assert np.isfinite(vectors).all()
        index={s:i for i,s in enumerate(seqs)}
        data={role:vectors[[index[r[role]] for r in rows]] for role in ['enhancer','promoter']}
        dest=folder/f'{split}.npz';np.savez(dest,**data)
        after=study.frozen_hash(encoder);assert after==before
        study.atomic_json(folder/f'{split}.json',{'study_id':study_id,'sha256':sha256_file(dest),
            'data_sha256':sha256_file(Path(cfg['data_dir'])/f'{split}.json'),'unique_sequences':len(seqs),'rows':len(rows),
            'tokenization':tokens.stats,'encoder_sha256_before':before,'encoder_sha256_after':after,
            'seconds':elapsed(cfg,start),'peak_memory':peak_memory(cfg)})
        answer[split]=data
    del encoder,h,pooled,batch;gc.collect();torch.cuda.empty_cache()
    return answer


class LogNames:
    def __init__(self,stream):self.stream=stream
    def write(self,s):return self.stream.write(s.replace('dev_MCC=','dev_macro_AUROC='))
    def flush(self):return self.stream.flush()


def main():
    warnings.filterwarnings('ignore',message='.*clean_up_tokenization_spaces.*')
    warnings.filterwarnings('ignore',message='Unable to import Triton.*')
    warnings.filterwarnings('ignore',message='Increasing alibi size.*')
    warnings.filterwarnings('error',category=ConvergenceWarning)
    sys.stdout=LogNames(sys.stdout);torch.set_num_threads(4);seed_everything(42)
    plan=read('configs/crispr_ranking.json');prior=read(plan['base_protocol']);source=read(Path(plan['data_dir'])/'source.json')
    for s,digest in source['sha256'].items():assert sha256_file(Path(plan['data_dir'])/f'{s}.json')==digest
    splitrows={s:read(Path(plan['data_dir'])/f'{s}.json') for s in ['train','dev']}
    n=len(splitrows['train']);pos=sum(r['label'] for r in splitrows['train'])
    distances=np.log10([r['distance'] for r in splitrows['train']]);dm=float(distances.mean());ds=float(distances.std())
    cfg={**prior['config'],**{k:plan[k] for k in ['data_dir','output_dir','epochs','patience','batch_size','gradient_accumulation','selection_metric']},
         'dataset':'K562_CRISPR_promoter_ranking','length_bucketing':False,'class_weights':[n/(2*(n-pos)),n/(2*pos)],
         'distance_mean':dm,'distance_sd':ds}
    root=Path(plan['output_dir']);root.mkdir(parents=True,exist_ok=True)
    files=list(Path('dnabert_fno').glob('*.py'))+[Path(__file__),Path('scripts/crispr_ranking_models.py'),Path('scripts/prepare_crispr_ranking.py'),Path('scripts/epi_information_models.py')]
    for path,digest in prior['code_sha256'].items():assert sha256_file(path)==digest
    identity={'plan':plan,'config':cfg,'environment':environment(cfg),'code_sha256':{str(p.resolve()):sha256_file(p) for p in files},
              'data_manifest_sha256':sha256_file(Path(plan['data_dir'])/'source.json'),
              'protocol_document_sha256':sha256_file('docs/CRISPR_RANKING_PROTOCOL.md'),
              'legacy_field_note':'study.fit_trial training field best_dev_mcc holds configured selection_metric=macro_auroc in this study; no MCC-based selection.'}
    study_id=study.stable_hash(identity)
    if (root/'protocol.json').exists():assert read(root/'protocol.json')==identity
    else:study.atomic_json(root/'protocol.json',identity)
    if (root/'status.json').exists() and read(root/'status.json')['phase']=='complete':print('Already complete');return
    if not (root/'started.json').exists():study.atomic_json(root/'started.json',{'unix':time.time()})
    started=read(root/'started.json')['unix']
    study.make_model=make_model;study.train_one_epoch=train_one_epoch;study.evaluate_study=evaluate
    f=features(root,cfg,study_id,splitrows,plan['extraction_batch_size'])
    alltraining=[];final=[];search={};selected={}
    for variant in VARIANTS:
        datasets={s:VectorDataset(design(f[s],rs,variant,dm,ds),rs,variant) for s,rs in splitrows.items()}
        candidates=[]
        for lr in plan['learning_rates']:
            records=[]
            for seed in plan['search_seeds']:
                study.atomic_json(root/'status.json',{'phase':'training','variant':variant,'seed':seed,'lr':lr,'completed_trials':len(alltraining),'expected_trials':35})
                r=study.fit_trial(root/'neural',study_id,cfg,variant,seed,lr,datasets,collate_features)
                records.append(r);alltraining.append(r)
            candidates.append({'lr':lr,'records':records,'mean_best_dev_macro_auroc':float(np.mean([r['best_dev_mcc'] for r in records]))})
        best=sorted(candidates,key=lambda c:(-c['mean_best_dev_macro_auroc'],c['lr']))[0]
        search[variant]=candidates;selected[variant]=best
        r=study.fit_trial(root/'neural',study_id,cfg,variant,44,best['lr'],datasets,collate_features)
        final.extend(best['records']+[r]);alltraining.append(r)
    assert len(alltraining)==35 and len(final)==15
    simple={};labels=np.array([r['label'] for r in splitrows['train']])
    for variant in ['distance','gc_distance','kmer4_distance']:
        dest=root/'simple'/variant;dest.mkdir(parents=True,exist_ok=True);done=dest/'selection.json'
        if done.exists():
            record=read(done);assert record['study_id']==study_id and record['checkpoint_sha256']==sha256_file(dest/'model.joblib')
            simple[variant]=record;continue
        matrices={s:simple_design(rs,variant) for s,rs in splitrows.items()};trials=[];models=[]
        for c in plan['logistic_C']:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',solver='lbfgs',max_iter=5000,tol=1e-6,random_state=42))
            model.fit(matrices['train'],labels)
            metrics,_=ranking_metrics(splitrows['dev'],model.predict_proba(matrices['dev'])[:,1])
            trials.append({'C':c,'dev':metrics});models.append(model)
        best=sorted(range(len(trials)),key=lambda i:(-trials[i]['dev']['macro_auroc'],trials[i]['C']))[0]
        joblib.dump(models[best],dest/'model.joblib')
        record={'study_id':study_id,'variant':variant,'trials':trials,'selected':trials[best],
                'checkpoint_sha256':sha256_file(dest/'model.joblib'),'test_used':False}
        study.atomic_json(done,record);simple[variant]=record
        print(f'SIMPLE {variant}: C={trials[best]["C"]} dev macro AUROC={trials[best]["dev"]["macro_auroc"]:.4f}',flush=True)
    study.atomic_json(root/'selection.json',{'study_id':study_id,'search':search,'selected':selected,'simple':simple,'test_used':False})
    study.atomic_json(root/'training_complete.json',{'study_id':study_id,'all_training':alltraining,'final':final,'simple_fits':15})
    test=read(Path(plan['data_dir'])/'test.json');ft=features(root,cfg,study_id,{'test':test},plan['extraction_batch_size'])['test']
    results=[]
    for r in final:
        variant=r['settings']['variant'];seed=r['settings']['seed'];dest=root/'results'/f'{variant}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
        model=make_model(cfg,variant,seed);checkpoint=Path(r['path'])/'best.pt'
        load_trainable(model,torch.load(checkpoint,map_location='cpu',weights_only=True)['trainable'])
        dataset=VectorDataset(design(ft,test,variant,dm,ds),test,variant)
        metrics,p,seconds=evaluate(model,dataset,collate_features,cfg);_,groups=ranking_metrics(test,p['probability'])
        save_predictions(dest/'predictions.csv',p);study.atomic_json(dest/'per_promoter.json',groups)
        record={'model':variant,'seed':seed,'metrics':metrics,'seconds':seconds,'training':r,'checkpoint_sha256':sha256_file(checkpoint),
                'prediction_sha256':sha256_file(dest/'predictions.csv'),'per_promoter_sha256':sha256_file(dest/'per_promoter.json')}
        study.atomic_json(dest/'result.json',record);results.append(record)
        print(f'TEST {variant} seed={seed}: macro AUROC={metrics["macro_auroc"]:.4f}, macro AP={metrics["macro_ap"]:.4f}',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    for variant,r in simple.items():
        dest=root/'results'/variant;dest.mkdir(parents=True,exist_ok=True)
        model=joblib.load(root/'simple'/variant/'model.joblib');p=model.predict_proba(simple_design(test,variant))[:,1]
        metrics,groups=ranking_metrics(test,p)
        save_predictions(dest/'predictions.csv',{'row_id':list(range(len(test))),'label':[r['label'] for r in test],'probability':p.tolist()})
        study.atomic_json(dest/'per_promoter.json',groups)
        record={'model':variant,'seed':None,'metrics':metrics,'selection':r,'prediction_sha256':sha256_file(dest/'predictions.csv'),
                'per_promoter_sha256':sha256_file(dest/'per_promoter.json')}
        study.atomic_json(dest/'result.json',record);results.append(record)
        print(f'TEST {variant}: macro AUROC={metrics["macro_auroc"]:.4f}, macro AP={metrics["macro_ap"]:.4f}',flush=True)
    summary={}
    for name in VARIANTS+list(simple):
        rs=[r for r in results if r['model']==name]
        summary[name]={m:{'mean':float(np.mean([r['metrics'][m] for r in rs])),
                         'sd':float(np.std([r['metrics'][m] for r in rs],ddof=1)) if len(rs)>1 else None,
                         'values':[r['metrics'][m] for r in rs]} for m in rs[0]['metrics']}
    study.atomic_json(root/'summary.json',{'study_id':study_id,'summary':summary,'results':results})
    study.atomic_json(root/'status.json',{'phase':'complete','study_id':study_id,'neural_trials':35,'simple_fits':15,
                                        'evaluations':len(results),'elapsed_seconds':time.time()-started})
    print('CRISPR RANKING COMPLETE',flush=True)


if __name__=='__main__':main()
