"""Matched-length continuous-DNA mixer study with deferred test perturbations."""
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from dnabert_fno.study import atomic_json,stable_hash
from dnabert_fno.data import sha256_file
from dnabert_fno.runtime import seed_everything,amp_context,environment,timed_start,elapsed,reset_peak,peak_memory
from dnabert_fno.train import accumulation_denominator
import continuous_rna_models as models


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def install_cache(path,device):
    array=np.load(path,mmap_mode='r');models.CACHE=torch.empty(array.shape,device=device,dtype=torch.float32)
    for first in range(0,len(array),32):models.CACHE[first:first+32].copy_(torch.from_numpy(np.array(array[first:first+32],copy=True)))


def fit(root,sid,cfg,kind,bp,seed,lr,ids,y):
    dest=root/'trials'/f'{kind}_bp{bp}_seed{seed}_lr{lr:g}';dest.mkdir(parents=True,exist_ok=True)
    settings={'study_id':sid,'model':kind,'bp':bp,'seed':seed,'lr':lr}
    if (dest/'training.json').exists():
        record=read(dest/'training.json');assert record['settings']==settings and sha256_file(dest/'best.pt')==record['checkpoint_sha256'];return record
    model=models.make_model(kind,cfg,seed);nparams=sum(p.numel() for p in model.parameters());assert nparams==models.count_params(kind,models.width_for(kind))
    initial=models.head_hash(model);optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'])
    best=-float('inf');best_epoch=None;stale=0;history=[];reset_peak(cfg);started=timed_start(cfg);training_seconds=0.;dev_seconds=0.
    for epoch in range(1,cfg['epochs']+1):
        model.train();seed_everything(seed+1000*epoch);order=np.random.default_rng(seed+1000*epoch).permutation(ids['train'])
        batch=cfg['batch_size'];steps=(len(order)+batch-1)//batch;optimizer.zero_grad(set_to_none=True);total=0.;start=timed_start(cfg)
        for step,first in enumerate(range(0,len(order),batch)):
            selected=order[first:first+batch];x=models.crop(models.CACHE[selected],bp)
            target=torch.as_tensor(y[selected],dtype=torch.float32,device=cfg['device'])
            denominator=accumulation_denominator(step,len(order),batch,cfg['gradient_accumulation'])
            with amp_context(cfg):prediction=model(x)
            loss_sum=(prediction-target).square().mean(1).sum();loss=loss_sum/denominator
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite expression loss')
            loss.backward();total+=float(loss_sum.detach())
            if (step+1)%cfg['gradient_accumulation']==0 or step+1==steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
        train_time=elapsed(cfg,start);training_seconds+=train_time;scheduler.step();start=timed_start(cfg)
        p=models.predict(model,ids['dev'],bp,cfg);metric=models.metrics(y[ids['dev']],p);dev_time=elapsed(cfg,start);dev_seconds+=dev_time
        if metric['macro_pearson']>best+cfg['min_delta']:
            best=metric['macro_pearson'];best_epoch=epoch;stale=0
            torch.save({'settings':settings,'epoch':epoch,'state_dict':model.state_dict()},dest/'best.pt');np.save(dest/'dev_predictions.npy',p)
        else:stale+=1
        history.append({'epoch':epoch,'train_mse':total/len(order),'dev_macro_pearson':metric['macro_pearson'],'dev_mse':metric['mse'],
                        'train_seconds':train_time,'dev_seconds':dev_time})
        atomic_json(dest/'epochs.json',history)
        if stale>=cfg['patience']:break
    record={'settings':settings,'path':str(dest),'params':nparams,'head_initial_sha256':initial,'epochs_completed':len(history),
            'best_epoch':best_epoch,'best_dev_macro_pearson':best,'train_seconds':training_seconds,'dev_seconds':dev_seconds,
            'elapsed_seconds':elapsed(cfg,started),'peak_memory':peak_memory(cfg),'checkpoint_sha256':sha256_file(dest/'best.pt'),
            'dev_predictions_sha256':sha256_file(dest/'dev_predictions.npy'),'test_evaluated':False}
    atomic_json(dest/'training.json',record);del model,optimizer,scheduler;gc.collect();torch.cuda.empty_cache();return record


def main():
    cfg=read('configs/continuous_rna.json');root=Path(cfg['output_dir']);data=Path(cfg['data_dir']);torch.set_num_threads(4);seed_everything(42)
    root.mkdir(parents=True,exist_ok=True);source=read(data/'source.json');cache=read(root/'features/manifest.json')
    for name,digest in source['files_sha256'].items():assert sha256_file(data/name)==digest
    assert cache['identity']['source_sha256']==sha256_file(data/'source.json') and sha256_file(root/'features/grid.npy')==cache['sha256']
    for p,digest in cache['identity']['code_sha256'].items():assert sha256_file(p)==digest
    files=[Path('scripts')/p for p in ['continuous_rna_models.py','run_continuous_rna.py','cache_continuous_rna.py','prepare_continuous_rna.py','download_continuous_rna.py']]+list(Path('dnabert_fno').glob('*.py'))
    protocol={'config':cfg,'environment':environment(cfg),'source_sha256':sha256_file(data/'source.json'),'cache_manifest_sha256':sha256_file(root/'features/manifest.json'),
              'protocol_document_sha256':sha256_file('docs/CONTINUOUS_RNA_PROTOCOL.md'),'code_sha256':{str(p):sha256_file(p) for p in files}}
    sid=stable_hash(protocol)
    if (root/'protocol.json').exists():assert read(root/'protocol.json')==protocol
    else:atomic_json(root/'protocol.json',protocol)
    if (root/'status.json').exists() and read(root/'status.json')['phase']=='complete':print('Already complete');return
    started=time.time();rows=read(data/'rows.json');ids={s:np.array([r['global_id'] for r in rows if r['split']==s]) for s in ['train','dev','test']}
    labels=np.load(data/'labels.npy');mean=labels[ids['train']].mean(0);sd=labels[ids['train']].std(0);assert (sd>0).all()
    y=(labels-mean)/sd;install_cache(root/'features/grid.npy',cfg['device'])
    feature_mean=torch.zeros(768,device=cfg['device'],dtype=torch.float64)
    for first in range(0,len(ids['train']),16):feature_mean+=models.CACHE[ids['train'][first:first+16]].double().sum((0,1))
    models.MASK_MEAN=(feature_mean/(len(ids['train'])*256)).float()
    np.savez(root/'normalizers.npz',label_mean=mean,label_sd=sd,feature_mean=models.MASK_MEAN.cpu().numpy())
    trained=[];final=[];selection={}
    for bp in cfg['lengths_bp']:
        for kind in cfg['variants']:
            candidates=[]
            for lr in cfg['learning_rates']:
                atomic_json(root/'status.json',{'phase':'training','completed_trials':len(trained),'expected_trials':36,'bp':bp,'model':kind,'seed':42,'lr':lr})
                r=fit(root,sid,cfg,kind,bp,42,lr,ids,y);trained.append(r);candidates.append(r)
                print(f'TRAIN {len(trained)}/36 {kind} {bp} seed42 lr{lr}',flush=True)
            best=sorted(candidates,key=lambda r:(-r['best_dev_macro_pearson'],r['settings']['lr']))[0]
            selection[f'{kind}_{bp}']={'candidates':candidates,'selected_lr':best['settings']['lr']};final.append(best)
            for seed in [43,44]:
                atomic_json(root/'status.json',{'phase':'training','completed_trials':len(trained),'expected_trials':36,'bp':bp,'model':kind,'seed':seed,'lr':best['settings']['lr']})
                r=fit(root,sid,cfg,kind,bp,seed,best['settings']['lr'],ids,y);trained.append(r);final.append(r)
                print(f'TRAIN {len(trained)}/36 {kind} {bp} seed{seed}',flush=True)
    assert len(trained)==36 and len(final)==27
    for seed in cfg['seeds']:assert len({r['head_initial_sha256'] for r in final if r['settings']['seed']==seed})==1
    atomic_json(root/'training_complete.json',{'study_id':sid,'all_training':trained,'final':final,'selection':selection,'normalizers_sha256':sha256_file(root/'normalizers.npz'),'test_used':False})
    results=[]
    for record in final:
        s=record['settings'];bp=s['bp'];kind=s['model'];seed=s['seed'];model=models.make_model(kind,cfg,seed)
        model.load_state_dict(torch.load(Path(record['path'])/'best.pt',map_location=cfg['device'],weights_only=True)['state_dict'])
        conditions=[('intact',None)]
        if bp==65536:conditions+=[('mask',None)]+[(k,pseed) for k in ['shuffle','swap'] for pseed in cfg['perturbation_seeds']]
        for condition,pseed in conditions:
            atomic_json(root/'status.json',{'phase':'evaluation','completed_evaluations':len(results),'expected_evaluations':126,'model':kind,'bp':bp,'condition':condition})
            start=timed_start(cfg);p=models.predict(model,ids['test'],bp,cfg,condition,pseed);seconds=elapsed(cfg,start)
            name=f'{kind}_bp{bp}_seed{seed}_{condition}'+('' if pseed is None else f'_{pseed}')
            dest=root/'results'/name;dest.mkdir(parents=True,exist_ok=True);np.save(dest/'predictions.npy',p)
            result={'settings':s,'condition':condition,'perturbation_seed':pseed,'training':record,'metrics':models.metrics(y[ids['test']],p),
                    'prediction_path':str(dest/'predictions.npy'),'prediction_sha256':sha256_file(dest/'predictions.npy'),'inference_seconds_cached':seconds}
            atomic_json(dest/'result.json',result);results.append(result)
        del model;gc.collect();torch.cuda.empty_cache()
    assert len(results)==126
    atomic_json(root/'summary.json',{'study_id':sid,'results':results,'test_gene_ids':ids['test'].tolist(),'train_mean_baseline_mse':float(np.mean(y[ids['test']]**2))})
    atomic_json(root/'status.json',{'phase':'complete','study_id':sid,'training_trials':36,'final_checkpoints':27,'prediction_matrices':126,'elapsed_seconds':time.time()-started})
    print('CONTINUOUS RNA COMPLETE',flush=True)


if __name__=='__main__':main()
