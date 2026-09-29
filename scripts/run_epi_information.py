"""Dev-selected input information ablations; evaluate a fresh cohort only after fitting."""
import csv
import gc
import json
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
from dnabert_fno import study
from dnabert_fno.adaptation import load_trainable
from dnabert_fno.data import collate_features, sha256_file
from dnabert_fno.loading import load_backbone
from dnabert_fno.metrics import binary_metrics
from dnabert_fno.runtime import environment,seed_everything,amp_context,peak_memory,reset_peak,timed_start,elapsed
from dnabert_fno.train import save_predictions
from epi_information_models import NEURAL,make_model,VectorDataset,feature_view,region_weights,pooled_regions,composition_matrix
from prepare_epi_information import prepare


def read(path): return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as f:
        records=list(csv.DictReader(f))
    for i,r in enumerate(records):
        r['label']=int(r['label']);r['row_id']=i
        assert len(r['enhancer'])==3000 and len(r['promoter'])==2000
    return records


def extract(cfg,plan,root,study_id,splits):
    """Store pooled vectors only; all representations come from the same frozen encoder."""
    folder=root/'features';folder.mkdir(exist_ok=True)
    pending=[];answer={}
    for split,records in splits.items():
        dest=folder/f'{split}.npz';meta=folder/f'{split}.json'
        if meta.exists():
            record=read(meta)
            assert record['study_id']==study_id and record['sha256']==sha256_file(dest)
            assert record['data_sha256']==sha256_file(Path(plan['data_dir'])/f'{split}.csv')
            with np.load(dest) as z: answer[split]={k:z[k].copy() for k in z.files}
        else: pending.append(split)
    if not pending: return answer
    tokenizer,encoder,_=load_backbone(cfg,offline=True)
    assert tokenizer.is_fast and not any(p.requires_grad for p in encoder.parameters())
    before=study.frozen_hash(encoder)
    encoder.cuda().eval()
    for split in pending:
        features={};stats={};records=splits[split]
        study.atomic_json(root/'status.json',{'phase':'extract','split':split})
        reset_peak(cfg);start=timed_start(cfg)
        for role in ['enhancer','promoter','joint']:
            sequences=[r['enhancer']+r['promoter'] if role=='joint' else r[role] for r in records]
            encoded=tokenizer(sequences,return_offsets_mapping=True,return_special_tokens_mask=True,truncation=False)
            lengths=list(map(len,encoded['input_ids']))
            assert max(lengths)<=cfg['max_tokens']
            stats[role]={'min_tokens':min(lengths),'max_tokens':max(lengths),'mean_tokens':float(np.mean(lengths)),'truncated_rows':0}
            chunks=[];globals_=[]
            for first in range(0,len(records),plan['extraction_batch_size']):
                last=min(first+plan['extraction_batch_size'],len(records));n=max(lengths[first:last]);b=last-first
                ids=torch.full((b,n),tokenizer.pad_token_id,dtype=torch.long,device='cuda')
                mask=torch.zeros((b,n),dtype=torch.long,device='cuda')
                weights=np.zeros((b,2 if role=='joint' else 1,n),dtype=np.float32)
                for j,i in enumerate(range(first,last)):
                    length=lengths[i]
                    ids[j,:length]=torch.tensor(encoded['input_ids'][i],device='cuda');mask[j,:length]=1
                    offsets=encoded['offset_mapping'][i]
                    spans=[(a,z) for a,z in offsets if z>a]
                    assert spans[0][0]==0 and spans[-1][1]==len(sequences[i])
                    assert all(z==a2 for (_,z),(a2,_) in zip(spans,spans[1:]))
                    if role=='joint': weights[j,:,:length]=region_weights(offsets,3000)
                    else: weights[j,0,:length]=1-np.array(encoded['special_tokens_mask'][i],dtype=np.float32)
                w=torch.from_numpy(weights).to('cuda')
                with torch.no_grad(),amp_context(cfg): hidden=encoder(ids,mask)
                pooled=pooled_regions(hidden,w)
                chunks.append(pooled.cpu().numpy().reshape(b,-1))
                if role=='joint': globals_.append(pooled_regions(hidden,w.sum(1,keepdim=True)).cpu().numpy()[:,0])
                if first%200==0: print(f'FEATURE {split} {role}: {last}/{len(records)}',flush=True)
            features['joint_pair' if role=='joint' else role]=np.concatenate(chunks)
            if role=='joint': features['joint_global']=np.concatenate(globals_)
        for x in features.values(): assert np.isfinite(x).all() and len(x)==len(records)
        after=study.frozen_hash(encoder);assert after==before
        dest=folder/f'{split}.npz';np.savez(dest,**features)
        study.atomic_json(folder/f'{split}.json',{'study_id':study_id,'data_sha256':sha256_file(Path(plan['data_dir'])/f'{split}.csv'),
            'sha256':sha256_file(dest),'shapes':{k:list(v.shape) for k,v in features.items()},'tokenization':stats,
            'encoder_sha256_before':before,'encoder_sha256_after':after,'encoder_frozen':True,
            'generation_seconds':elapsed(cfg,start),'peak_memory':peak_memory(cfg)})
        answer[split]=features
    del encoder,hidden,pooled,ids,mask,w;gc.collect();torch.cuda.empty_cache()
    return answer


def main():
    warnings.filterwarnings('ignore',message='.*clean_up_tokenization_spaces.*')
    warnings.filterwarnings('ignore',message='Unable to import Triton.*')
    warnings.filterwarnings('ignore',message='Increasing alibi size.*')
    warnings.filterwarnings('error',category=ConvergenceWarning)
    torch.set_num_threads(4);seed_everything(42)
    plan=read('configs/epi_information.json');data=prepare()
    prior=read(plan['source_protocol'])
    cfg={**prior['config'],**{k:plan[k] for k in ['epochs','patience','batch_size','gradient_accumulation']},
         'data_dir':plan['data_dir'],'output_dir':plan['output_dir'],'length_bucketing':False,'dataset':'GM12878_information_ablation'}
    root=Path(plan['output_dir']);root.mkdir(parents=True,exist_ok=True)
    files=list(Path('dnabert_fno').glob('*.py'))+[Path(__file__),Path('scripts/epi_information_models.py'),Path('scripts/prepare_epi_information.py'),Path('scripts/prepare_long_epi.py')]
    for path,digest in prior['code_sha256'].items(): assert sha256_file(path)==digest
    identity={'plan':plan,'config':cfg,'data_manifest_sha256':sha256_file(Path(plan['data_dir'])/'source.json'),
              'environment':environment(cfg),'code_sha256':{str(p.resolve()):sha256_file(p) for p in files},
              'protocol_document_sha256':sha256_file('docs/EPI_INFORMATION_PROTOCOL.md'),
              'source_protocol_sha256':sha256_file(plan['source_protocol'])}
    study_id=study.stable_hash(identity)
    if (root/'protocol.json').exists(): assert read(root/'protocol.json')==identity
    else: study.atomic_json(root/'protocol.json',identity)
    if (root/'status.json').exists() and read(root/'status.json')['phase']=='complete': print('Already complete');return
    if not (root/'started.json').exists(): study.atomic_json(root/'started.json',{'unix':time.time()})
    started=read(root/'started.json')['unix']
    splitrows={s:rows(Path(plan['data_dir'])/f'{s}.csv') for s in ['train','dev']}
    features=extract(cfg,plan,root,study_id,splitrows)
    labels={s:np.array([r['label'] for r in rs]) for s,rs in splitrows.items()}
    study.make_model=make_model
    search={};selected={};final=[];alltraining=[]
    for variant in NEURAL:
        datasets={s:VectorDataset(feature_view(features[s],variant),labels[s]) for s in splitrows}
        candidates=[]
        for lr in plan['learning_rates']:
            records=[]
            for seed in plan['search_seeds']:
                study.atomic_json(root/'status.json',{'phase':'training','variant':variant,'seed':seed,'lr':lr,'completed_trials':len(alltraining),'expected_trials':35})
                r=study.fit_trial(root/'neural',study_id,cfg,variant,seed,lr,datasets,collate_features)
                records.append(r);alltraining.append(r)
            candidates.append({'lr':lr,'records':records,'mean_best_dev_mcc':float(np.mean([r['best_dev_mcc'] for r in records]))})
        best=sorted(candidates,key=lambda c:(-c['mean_best_dev_mcc'],c['lr']))[0]
        search[variant]=candidates;selected[variant]=best
        r=study.fit_trial(root/'neural',study_id,cfg,variant,44,best['lr'],datasets,collate_features)
        alltraining.append(r);final.extend(best['records']+[r])
    assert len(alltraining)==35 and len(final)==15
    for seed in plan['final_seeds']:
        hashes=[r['head_initial_sha256'] for r in final if r['settings']['seed']==seed and r['settings']['variant'] in ['separate_pair','joint_pair']]
        assert len(set(hashes))==1
    simple={}
    for kind in ['gc','kmer4']:
        for role in ['enhancer','promoter','pair']:
            name=f'{kind}_{role}';dest=root/'composition'/name;dest.mkdir(parents=True,exist_ok=True)
            done=dest/'selection.json'
            if done.exists():
                record=read(done);assert record['study_id']==study_id
                assert record['checkpoint_sha256']==sha256_file(dest/'model.joblib')
                simple[name]=record;continue
            x={s:composition_matrix(rs,kind,role) for s,rs in splitrows.items()};trials=[];models=[]
            for c in plan['logistic_C']:
                model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=10000,tol=1e-8,solver='lbfgs',random_state=42))
                start=time.perf_counter();model.fit(x['train'],labels['train']);seconds=time.perf_counter()-start
                metrics=binary_metrics(labels['dev'],model.predict_proba(x['dev'])[:,1])
                trials.append({'C':c,'dev':metrics,'fit_seconds':seconds});models.append(model)
            index=sorted(range(len(trials)),key=lambda i:(-trials[i]['dev']['mcc'],trials[i]['C']))[0]
            joblib.dump(models[index],dest/'model.joblib')
            record={'study_id':study_id,'kind':kind,'role':role,'selected':trials[index],'trials':trials,
                    'checkpoint_sha256':sha256_file(dest/'model.joblib'),'dimensions':int(x['train'].shape[1]),'test_used':False}
            study.atomic_json(done,record);simple[name]=record
            print(f'COMPOSITION {name}: C={trials[index]["C"]} dev MCC={trials[index]["dev"]["mcc"]:.4f}',flush=True)
    study.atomic_json(root/'selection.json',{'study_id':study_id,'search':search,'selected':selected,'composition':simple,'test_used':False})
    study.atomic_json(root/'training_complete.json',{'study_id':study_id,'all_training':alltraining,'final':final,'composition_models':6,'composition_fits':30})
    testrows=rows(Path(plan['data_dir'])/'test.csv')
    testfeatures=extract(cfg,plan,root,study_id,{'test':testrows})['test']
    testlabels=np.array([r['label'] for r in testrows]);results=[]
    for record in final:
        variant=record['settings']['variant'];seed=record['settings']['seed']
        dest=root/'results'/f'{variant}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
        model=make_model(cfg,variant,seed);checkpoint=Path(record['path'])/'best.pt'
        load_trainable(model,torch.load(checkpoint,map_location='cpu',weights_only=True)['trainable'])
        ds=VectorDataset(feature_view(testfeatures,variant),testlabels)
        metrics,predictions,seconds=study.evaluate_study(model,ds,collate_features,cfg)
        save_predictions(dest/'predictions.csv',predictions)
        result={'model':variant,'seed':seed,'metrics':metrics,'seconds':seconds,'training':record,
                'checkpoint_sha256':sha256_file(checkpoint),'predictions_sha256':sha256_file(dest/'predictions.csv')}
        study.atomic_json(dest/'result.json',result);results.append(result)
        print(f'TEST {variant} seed={seed}: MCC={metrics["mcc"]:.4f} AUC={metrics["roc_auc"]:.4f}',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    for name,record in simple.items():
        dest=root/'results'/name;dest.mkdir(parents=True,exist_ok=True)
        model=joblib.load(root/'composition'/name/'model.joblib')
        x=composition_matrix(testrows,record['kind'],record['role'])
        p=model.predict_proba(x)[:,1];metrics=binary_metrics(testlabels,p)
        save_predictions(dest/'predictions.csv',{'row_id':list(range(len(p))),'label':testlabels.tolist(),'probability':p.tolist()})
        result={'model':name,'seed':None,'metrics':metrics,'selection':record,'predictions_sha256':sha256_file(dest/'predictions.csv')}
        study.atomic_json(dest/'result.json',result);results.append(result)
        print(f'TEST {name}: MCC={metrics["mcc"]:.4f} AUC={metrics["roc_auc"]:.4f}',flush=True)
    summary={}
    for name in NEURAL+list(simple):
        rs=[r for r in results if r['model']==name]
        summary[name]={m:{'mean':float(np.mean([r['metrics'][m] for r in rs])),
                          'sd':float(np.std([r['metrics'][m] for r in rs],ddof=1)) if len(rs)>1 else None,
                          'values':[r['metrics'][m] for r in rs]} for m in rs[0]['metrics']}
    study.atomic_json(root/'summary.json',{'study_id':study_id,'summary':summary,'results':results})
    study.atomic_json(root/'status.json',{'phase':'complete','study_id':study_id,'neural_training_trials':35,
                                       'composition_fits':30,'evaluations':len(results),'elapsed_seconds':time.time()-started})
    print('EPI INFORMATION COMPLETE',flush=True)


if __name__=='__main__': main()
