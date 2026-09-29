"""Controlled pointwise/pairwise comparison; all selections precede test scoring."""
import csv
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from scipy.special import expit
from dnabert_fno.study import atomic_json,stable_hash
from dnabert_fno.data import sha256_file
from dnabert_fno.runtime import seed_everything,amp_context,environment,timed_start,elapsed,peak_memory,reset_peak
from dnabert_fno.train import accumulation_denominator
from cache_matched_grid import gpu_load
import matched_cv_models as grid
import residual_rank_models as models


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def train_trial(root,identity,cfg,fold,kind,objective,seed,rows,pairs,indices,base):
    dest=root/'trials'/f'fold{fold}_{kind}_{objective}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    settings={'study_id':identity,'fold':fold,'model':kind,'objective':objective,'seed':seed}
    if (dest/'training.json').exists():
        old=read(dest/'training.json');assert old['settings']==settings
        assert sha256_file(dest/'best.pt')==old['checkpoint_sha256'];return old
    trainpairs=[p for p in pairs if p['fold'] not in [fold,(fold+1)%5]]
    devpairs=[p for p in pairs if p['fold']==(fold+1)%5]
    dev_ids=sorted({p[k] for p in devpairs for k in ['positive','negative']})
    pair_ids=np.array([[p['positive'],p['negative']] for p in trainpairs])
    weights=models.pair_weights(trainpairs)
    model=models.make_model(kind,cfg,seed);reset_peak(cfg)
    nparams=sum(p.numel() for p in model.parameters());assert nparams==grid.count_params(kind,grid.width_for(kind))-129
    assert np.all(models.predict(model,indices[dev_ids],cfg)==0)
    head_hash=models.parameter_hash(model,True);initial_hash=models.parameter_hash(model)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs'])
    best=-float('inf');best_epoch=None;stale=0;history=[];start=timed_start(cfg)
    for epoch in range(1,cfg['epochs']+1):
        model.train();seed_everything(seed+1000*epoch)
        order=np.random.default_rng(seed+1000*epoch).permutation(len(trainpairs))
        total=0.;opt.zero_grad(set_to_none=True);batch=cfg['pair_batch_size']
        nsteps=(len(order)+batch-1)//batch
        for step,first in enumerate(range(0,len(order),batch)):
            selected=order[first:first+batch];ids=pair_ids[selected]
            inp=torch.as_tensor(indices[ids].reshape(-1,2),device=cfg['device'])
            baseline=torch.as_tensor(base[ids],dtype=torch.float32,device=cfg['device'])
            w=torch.as_tensor(weights[selected],device=cfg['device'])
            denominator=accumulation_denominator(step,len(order),batch,cfg['gradient_accumulation'])
            with amp_context(cfg):correction=model(inp).reshape(-1,2)
            loss_sum=(models.pair_loss(baseline+correction,objective)*w).sum()
            loss=loss_sum/denominator
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pair objective')
            loss.backward();total+=float(loss_sum.detach())
            if (step+1)%cfg['gradient_accumulation']==0 or step+1==nsteps:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
                opt.step();opt.zero_grad(set_to_none=True)
        scheduler.step()
        correction=models.predict(model,indices[dev_ids],cfg)
        metric=grid.matched_concordance(devpairs,dict(zip(dev_ids,base[dev_ids]+correction)))[0]
        if metric>best+cfg['min_delta']:
            best=metric;best_epoch=epoch;stale=0
            torch.save({'settings':settings,'epoch':epoch,'state_dict':model.state_dict()},dest/'best.pt')
            np.save(dest/'dev_correction.npy',correction)
        else:stale+=1
        history.append({'epoch':epoch,'train_loss':total/len(order),'dev_raw_concordance':metric})
        atomic_json(dest/'epochs.json',history)
        if stale>=cfg['patience']:break
    seconds=elapsed(cfg,start);correction=np.load(dest/'dev_correction.npy')
    selected,candidates=models.choose_gamma(base[dev_ids],correction,devpairs,dev_ids,cfg['gammas'])
    record={'settings':settings,'path':str(dest),'trainable_params':nparams,'initial_sha256':initial_hash,
            'head_initial_sha256':head_hash,'epochs_completed':len(history),'best_epoch':best_epoch,
            'best_dev_raw_concordance':best,'selected_gamma':selected['gamma'],'gamma_candidates':candidates,
            'selected_dev_concordance':selected['dev_concordance'],'train_pairs':len(trainpairs),
            'train_promoters':len({p['promoter_id'] for p in trainpairs}),'dev_global_ids':dev_ids,
            'checkpoint_sha256':sha256_file(dest/'best.pt'),'dev_correction_sha256':sha256_file(dest/'dev_correction.npy'),
            'fit_seconds_including_dev':seconds,'peak_memory':peak_memory(cfg),'test_evaluated':False}
    atomic_json(dest/'training.json',record);del model,opt,scheduler;gc.collect()
    return record


def main():
    cfg=read('configs/residual_rank.json');torch.set_num_threads(4);seed_everything(42)
    root=Path(cfg['output_dir']);prior=Path(cfg['prior_run']);data=Path(cfg['data_dir']);root.mkdir(parents=True,exist_ok=True)
    old=read(prior/'protocol.json');assert read(prior/'status.json')['phase']=='complete'
    for p,digest in old['code_sha256'].items():assert sha256_file(p)==digest
    source=read(data/'source.json')
    for name,digest in source['files_sha256'].items():assert sha256_file(data/f'{name}.json')==digest
    cache=read(prior/'grid_cache/manifest.json');assert sha256_file(prior/'grid_cache/grid.npy')==cache['sha256']
    assert cache['encoder_hash_before']==cache['encoder_hash_after']
    assets=[data/'source.json',prior/'protocol.json',prior/'grid_cache/manifest.json']+[prior/f'fold{f}/distance.joblib' for f in cfg['folds']]
    files=[Path('scripts')/p for p in ['residual_rank_models.py','run_residual_rank.py','matched_cv_models.py','cache_matched_grid.py']]+list(Path('dnabert_fno').glob('*.py'))
    protocol={'config':cfg,'environment':environment(cfg),'assets_sha256':{str(p):sha256_file(p) for p in assets},
              'code_sha256':{str(p):sha256_file(p) for p in files},'protocol_sha256':sha256_file('docs/RESIDUAL_RANK_PROTOCOL.md')}
    sid=stable_hash(protocol)
    if (root/'protocol.json').exists():assert read(root/'protocol.json')==protocol
    else:atomic_json(root/'protocol.json',protocol)
    if (root/'status.json').exists() and read(root/'status.json')['phase']=='complete':print('Already complete');return
    started=time.time();rows=read(data/'rows.json');pairs=read(data/'matched_pairs.json')
    assert [r['global_id'] for r in rows]==list(range(len(rows)))
    seqs=sorted({r[k] for r in rows for k in ['enhancer','promoter']});assert stable_hash(seqs)==cache['identity']['sequence_hash']
    index={s:i for i,s in enumerate(seqs)};indices=np.array([[index[r['enhancer']],index[r['promoter']]] for r in rows])
    grid.CACHE=gpu_load(prior/'grid_cache/grid.npy');trained=[];bases={}
    for fold in cfg['folds']:
        simple=joblib.load(prior/f'fold{fold}/distance.joblib');train=[r for r in rows if r['fold'] not in [fold,(fold+1)%5]]
        np.testing.assert_allclose(simple[0].mean_,np.log10([r['distance'] for r in train]).mean(),rtol=0,atol=1e-12)
        bases[fold]=simple.decision_function(np.log10([r['distance'] for r in rows])[:,None])
        for kind in cfg['variants']:
            for objective in cfg['objectives']:
                for seed in cfg['seeds']:
                    atomic_json(root/'status.json',{'phase':'training','completed_trials':len(trained),'expected_trials':90,
                        'fold':fold,'model':kind,'objective':objective,'seed':seed})
                    record=train_trial(root,sid,cfg,fold,kind,objective,seed,rows,pairs,indices,bases[fold]);trained.append(record)
                    print(f"TRAIN {len(trained)}/90 fold{fold} {kind} {objective} seed{seed} gamma={record['selected_gamma']}",flush=True)
    assert len(trained)==90
    for fold in cfg['folds']:
        for seed in cfg['seeds']:
            subset=[r for r in trained if r['settings']['fold']==fold and r['settings']['seed']==seed]
            assert len({r['head_initial_sha256'] for r in subset})==1
            for kind in cfg['variants']:assert len({r['initial_sha256'] for r in subset if r['settings']['model']==kind})==1
    atomic_json(root/'training_complete.json',{'study_id':sid,'trials':trained,'test_used':False})
    results=[]
    for record in trained:
        s=record['settings'];fold=s['fold'];test=[r['global_id'] for r in rows if r['fold']==fold]
        atomic_json(root/'status.json',{'phase':'evaluation','completed_evaluations':len(results),'expected_evaluations':95})
        model=models.make_model(s['model'],cfg,s['seed']);path=Path(record['path'])/'best.pt'
        model.load_state_dict(torch.load(path,map_location=cfg['device'],weights_only=True)['state_dict'])
        correction=models.predict(model,indices[test],cfg);baseline=bases[fold][test]
        dest=root/'results'/Path(record['path']).name;dest.mkdir(parents=True,exist_ok=True)
        with (dest/'predictions.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['global_id','label','distance_logit','correction','raw_score','selected_score'])
            w.writerows([i,rows[i]['label'],b,c,b+c,b+record['selected_gamma']*c] for i,b,c in zip(test,baseline,correction))
        fp=[p for p in pairs if p['fold']==fold]
        scores={name:grid.matched_concordance(fp,dict(zip(test,baseline+gamma*correction)))[0] for name,gamma in [('raw',1),('selected',record['selected_gamma'])]}
        result={'settings':s,'training':record,'primary':scores,'predictions':str(dest/'predictions.csv'),'prediction_sha256':sha256_file(dest/'predictions.csv')}
        atomic_json(dest/'result.json',result);results.append(result);del model
    for fold in cfg['folds']:
        test=[r['global_id'] for r in rows if r['fold']==fold];dest=root/'results'/f'fold{fold}_distance';dest.mkdir(parents=True,exist_ok=True)
        with (dest/'predictions.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['global_id','label','distance_logit']);w.writerows([i,rows[i]['label'],bases[fold][i]] for i in test)
        result={'settings':{'fold':fold,'model':'distance','objective':None,'seed':None},'predictions':str(dest/'predictions.csv'),'prediction_sha256':sha256_file(dest/'predictions.csv')}
        atomic_json(dest/'result.json',result);results.append(result)
    atomic_json(root/'summary.json',{'study_id':sid,'results':results})
    atomic_json(root/'status.json',{'phase':'complete','study_id':sid,'neural_trials':90,'distance_models_reused':5,
                                  'evaluations':len(results),'elapsed_seconds':time.time()-started})
    print('RESIDUAL RANK COMPLETE',flush=True)


if __name__=='__main__':main()
