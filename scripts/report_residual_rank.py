"""Independent OOF score audit and prespecified residual-ranking contrasts."""
import csv
import json
import os
import shutil
from collections import Counter,defaultdict
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from dnabert_fno.data import sha256_file
from dnabert_fno.study import atomic_json,stable_hash
from cache_matched_grid import gpu_load
import matched_cv_models as grid
import residual_rank_models as models


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def concordances(pairs,score):
    groups=defaultdict(list)
    for pair in pairs:
        positive=float(score[pair['positive']]);negative=float(score[pair['negative']])
        groups[pair['promoter_id']].append(1 if positive>negative else 0 if positive<negative else .5)
    return {name:float(np.mean(v)) for name,v in sorted(groups.items())}


def macro_auc(rows,score):
    groups=defaultdict(list)
    for r in rows:groups[r['promoter_id']].append(r)
    values=[]
    for group in groups.values():
        if len({r['label'] for r in group})==2:values.append(roc_auc_score([r['label'] for r in group],[score[r['global_id']] for r in group]))
    assert len(values)==123
    return float(np.mean(values))


def stat(values):
    return {'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1)) if len(values)>1 else None,'values':values}


def main():
    torch.set_num_threads(4)
    root=Path('runs/residual_rank');out=Path('docs/residual_rank_results');status=read(root/'status.json');assert status['phase']=='complete'
    protocol=read(root/'protocol.json');cfg=protocol['config'];sid=stable_hash(protocol);assert sid==status['study_id']
    for field in ['assets_sha256','code_sha256']:
        for p,digest in protocol[field].items():assert sha256_file(p)==digest
    assert sha256_file('docs/RESIDUAL_RANK_PROTOCOL.md')==protocol['protocol_sha256']
    prior=Path(cfg['prior_run']);data=Path(cfg['data_dir']);source=read(data/'source.json')
    for name,digest in source['files_sha256'].items():assert sha256_file(data/f'{name}.json')==digest
    rows=read(data/'rows.json');pairs=read(data/'matched_pairs.json');sensitivity=read(data/'sensitivity_pairs.json')
    by_promoter=defaultdict(list)
    for r in rows:by_promoter[r['promoter_id']].append(r)
    for ratio,actual in [('1.25',pairs),*sensitivity.items()]:
        expected={(p['global_id'],n['global_id']) for group in by_promoter.values() for p in group if p['label']==1 for n in group if n['label']==0 and max(p['distance'],n['distance'])<=float(ratio)*min(p['distance'],n['distance'])}
        assert expected=={(p['positive'],p['negative']) for p in actual} and len(expected)==len(actual)
    rc=lambda x:min(x,x.translate(str.maketrans('ACGTN','TGCAN'))[::-1])
    for a in range(5):
        for b in range(a):
            ra=[r for r in rows if r['fold']==a];rb=[r for r in rows if r['fold']==b]
            for key in ['chrom','gene','promoter_id']:assert not {r[key] for r in ra}&{r[key] for r in rb}
            assert not {rc(r[key]) for r in ra for key in ['enhancer','promoter']}&{rc(r[key]) for r in rb for key in ['enhancer','promoter']}
    cache=read(prior/'grid_cache/manifest.json');assert sha256_file(prior/'grid_cache/grid.npy')==cache['sha256']
    assert cache['encoder_hash_before']==cache['encoder_hash_after']
    seqs=sorted({r[k] for r in rows for k in ['enhancer','promoter']});assert stable_hash(seqs)==cache['identity']['sequence_hash']
    index={s:i for i,s in enumerate(seqs)};indices=np.array([[index[r['enhancer']],index[r['promoter']]] for r in rows])
    grid.CACHE=gpu_load(prior/'grid_cache/grid.npy');bases={}
    for fold in cfg['folds']:
        simple=joblib.load(prior/f'fold{fold}/distance.joblib');train=[r for r in rows if r['fold'] not in [fold,(fold+1)%5]]
        distances=np.log10([r['distance'] for r in train]);assert np.isclose(simple[0].mean_[0],distances.mean(),atol=1e-12,rtol=0)
        assert np.isclose(simple[0].scale_[0],distances.std(),atol=1e-12,rtol=0)
        bases[fold]=simple.decision_function(np.log10([r['distance'] for r in rows])[:,None])
    training=read(root/'training_complete.json');results=read(root/'summary.json')['results']
    assert training['study_id']==sid and not training['test_used'] and len(training['trials'])==90 and len(results)==95
    oof={};seen={};maxdifference=0.;expected_records={(fold,kind,objective,seed) for fold in range(5) for kind in cfg['variants'] for objective in cfg['objectives'] for seed in cfg['seeds']}
    assert {(r['settings']['fold'],r['settings']['model'],r['settings']['objective'],r['settings']['seed']) for r in training['trials']}==expected_records
    for number,result in enumerate(results,1):
        s=result['settings'];fold=s['fold'];name=s['model'];objective=s['objective'];seed=s['seed']
        test=[r['global_id'] for r in rows if r['fold']==fold];p=Path(result['predictions']);assert sha256_file(p)==result['prediction_sha256']
        with p.open(newline='') as f:stored=list(csv.DictReader(f))
        assert [int(r['global_id']) for r in stored]==test and [int(r['label']) for r in stored]==[rows[i]['label'] for i in test]
        baseline=np.array([float(r['distance_logit']) for r in stored]);np.testing.assert_array_equal(baseline,bases[fold][test])
        if name=='distance':
            key=('distance',None,'selected');values=baseline
            if key not in oof:oof[key]=np.zeros(len(rows));seen[key]=np.zeros(len(rows),int)
            oof[key][test]=values;seen[key][test]+=1
            atomic_json(root/'verification_progress.json',{'completed':number,'total':95})
            continue
        record=result['training'];assert record in training['trials'] and not record['test_evaluated']
        dest=Path(record['path']);assert sha256_file(dest/'best.pt')==record['checkpoint_sha256']
        assert sha256_file(dest/'dev_correction.npy')==record['dev_correction_sha256']
        history=read(dest/'epochs.json');best=-float('inf');best_epoch=None;stale=0
        for h in history:
            assert np.isfinite(h['train_loss'])
            if h['dev_raw_concordance']>best+cfg['min_delta']:best=h['dev_raw_concordance'];best_epoch=h['epoch'];stale=0
            else:stale+=1
        assert best_epoch==record['best_epoch'] and best==record['best_dev_raw_concordance']
        assert len(history)==record['epochs_completed'] and (len(history)==cfg['epochs'] or stale==cfg['patience'])
        trainpairs=[p for p in pairs if p['fold'] not in [fold,(fold+1)%5]]
        assert record['train_pairs']==len(trainpairs) and record['train_promoters']==len({p['promoter_id'] for p in trainpairs})==27
        devpairs=[p for p in pairs if p['fold']==(fold+1)%5];dev_ids=sorted({p[k] for p in devpairs for k in ['positive','negative']})
        assert record['dev_global_ids']==dev_ids
        model=models.make_model(name,cfg,seed)
        assert models.parameter_hash(model)==record['initial_sha256'] and models.parameter_hash(model,True)==record['head_initial_sha256']
        assert sum(p.numel() for p in model.parameters())==record['trainable_params']
        assert np.all(models.predict(model,indices[dev_ids],cfg)==0)
        checkpoint=torch.load(dest/'best.pt',map_location=cfg['device'],weights_only=True)
        assert checkpoint['settings']==s and checkpoint['epoch']==record['best_epoch'];model.load_state_dict(checkpoint['state_dict'])
        dev_correction=models.predict(model,indices[dev_ids],cfg);np.testing.assert_array_equal(dev_correction,np.load(dest/'dev_correction.npy'))
        selected,candidates=models.choose_gamma(bases[fold][dev_ids],dev_correction,devpairs,dev_ids,cfg['gammas'])
        assert candidates==record['gamma_candidates'] and selected['gamma']==record['selected_gamma'] and selected['dev_concordance']==record['selected_dev_concordance']
        assert selected['dev_concordance']>=candidates[0]['dev_concordance']
        assert abs(np.mean(list(concordances(devpairs,dict(zip(dev_ids,bases[fold][dev_ids]+dev_correction))).values()))-best)<1e-12
        correction=models.predict(model,indices[test],cfg);saved=np.array([float(r['correction']) for r in stored]);maxdifference=max(maxdifference,float(np.max(np.abs(correction-saved))))
        np.testing.assert_allclose(correction,saved,atol=1e-7,rtol=0);del model
        for mode,gamma in [('raw',1.),('selected',selected['gamma'])]:
            values=np.array([float(r[mode+'_score']) for r in stored]);np.testing.assert_array_equal(values,baseline+gamma*saved)
            assert np.isfinite(values).all()
            assert abs(np.mean(list(concordances([p for p in pairs if p['fold']==fold],dict(zip(test,values))).values()))-result['primary'][mode])<1e-12
            key=(objective+'_'+name,seed,mode)
            if key not in oof:oof[key]=np.zeros(len(rows));seen[key]=np.zeros(len(rows),int)
            oof[key][test]=values;seen[key][test]+=1
        atomic_json(root/'verification_progress.json',{'completed':number,'total':95})
        if number%15==0:print(f'Verified {number}/95',flush=True)
    assert all(np.all(x==1) for x in seen.values())
    for fold in range(5):
        for seed in cfg['seeds']:
            subset=[r for r in training['trials'] if r['settings']['fold']==fold and r['settings']['seed']==seed]
            assert len({r['head_initial_sha256'] for r in subset})==1
            for kind in cfg['variants']:assert len({r['initial_sha256'] for r in subset if r['settings']['model']==kind})==1
    names=[objective+'_'+kind for kind in cfg['variants'] for objective in cfg['objectives']]+['distance']
    summaries={};pergroups={};costs={}
    for name in names:
        summaries[name]={}
        seeds=cfg['seeds'] if name!='distance' else [None]
        for mode in ['raw','selected']:
            ps=[oof[(name,seed,mode if name!='distance' else 'selected')] for seed in seeds]
            gs=[list(concordances(pairs,p).values()) for p in ps];pergroups[(name,mode)]=np.mean(gs,axis=0)
            summaries[name][mode]={'primary':stat([float(np.mean(g)) for g in gs]),
                'macro_auroc':stat([macro_auc(rows,p) for p in ps]),
                'sensitivity':{ratio:stat([float(np.mean(list(concordances(pairs_,p).values()))) for p in ps]) for ratio,pairs_ in sensitivity.items()}}
        if name!='distance':
            records=[r for r in training['trials'] if r['settings']['objective']+'_'+r['settings']['model']==name]
            summaries[name]['gamma_counts']={str(k):sum(r['selected_gamma']==k for r in records) for k in cfg['gammas']}
            costs[name]={'trials':len(records),'params':sorted({r['trainable_params'] for r in records}),
                'epochs':sum(r['epochs_completed'] for r in records),'fit_seconds_including_dev':sum(r['fit_seconds_including_dev'] for r in records),
                'peak_allocated_mib':max(r['peak_memory']['allocated_mib'] for r in records)}
    genes={p['promoter_id']:p['gene'] for p in pairs};assert len(genes)==len(set(genes.values()))==45
    idx=np.random.default_rng(cfg['bootstrap_seed']).integers(0,45,size=(cfg['bootstrap_repeats'],45))
    contrasts=[('pairwise_fno',b) for b in ['pairwise_cnn','pairwise_mlp','distance']]+[(f'pairwise_{k}',f'pointwise_{k}') for k in cfg['variants']]+[(f'pairwise_{k}','distance') for k in ['mlp','cnn']]
    bootstrap={}
    for mode in ['raw','selected']:
        bootstrap[mode]={}
        for a,b in contrasts:
            delta=pergroups[(a,mode)]-pergroups[(b,mode)];boots=delta[idx].mean(1)
            bootstrap[mode][a+'_minus_'+b]={'difference':float(delta.mean()),'lower_95':float(np.quantile(boots,.025)),'upper_95':float(np.quantile(boots,.975))}
    summary={'study_id':sid,'models':summaries,'bootstrap':bootstrap,'primary_promoters':45,'primary_pairs':397,
             'uncertainty':'45-gene paired bootstrap of seed-average scores, conditional on fitted models; no multiple-comparison adjustment or retraining/fold uncertainty'}
    atomic_json(root/'oof_summary.json',summary);atomic_json(root/'cost_summary.json',{'models':costs,'runner_seconds':status['elapsed_seconds']})
    verification={'passed':True,'study_id':sid,'analysis_script_sha256':sha256_file(__file__),'trials':90,'evaluations':95,
        'max_abs_regenerated_correction_difference':maxdifference,'oof_rows_per_arm_seed':10273,
        'checks':['Data/code/encoder cache/baseline/checkpoint hashes','Independent pair reconstruction and chromosome/gene/exact-RC isolation',
                  'Frozen distance train-only normalizer','Zero initialization and matched head/model initial hashes',
                  'Every best epoch and dev-only gamma choice','All 90 neural corrections and five distance predictions regenerated',
                  'Independent score concordance, finite outputs, score composition and exact OOF coverage']}
    atomic_json(root/'verification.json',verification)
    out.mkdir(parents=True,exist_ok=True);manifest=[]
    for p in list(root.glob('*.json'))+list(root.glob('trials/*/*.json'))+list(root.glob('results/*/*')):
        if p.suffix not in ['.json','.csv']:continue
        dest=out/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        manifest.append({'source':str(p),'export':str(dest.relative_to(out)),'sha256':sha256_file(dest)})
    atomic_json(out/'export_manifest.json',manifest)
    with (out/'oof_scores.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','seed','mode','global_id','fold','label','score'])
        for (name,seed,mode),scores in oof.items():w.writerows([name,seed,mode,r['global_id'],r['fold'],r['label'],scores[r['global_id']]] for r in rows)
    with (out/'per_promoter.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','mode','promoter_id','gene','concordance'])
        for (name,mode),values in pergroups.items():w.writerows([name,mode,prom,genes[prom],v] for prom,v in zip(sorted(genes),values))
    fmt=lambda s:f"{s['mean']:.4f}"+(f" ± {s['sd']:.4f}" if s['sd'] is not None else '')
    table=['# Frozen-distance residual objective diagnostic','','| Arm | Raw gamma=1 | Dev-selected gamma | Gamma=0 folds/seeds |','|---|---:|---:|---:|']
    for name in names:
        m=summaries[name];table.append(f"| {name} | {fmt(m['raw']['primary'])} | {fmt(m['selected']['primary'])} | {m.get('gamma_counts',{}).get('0.0','—')} |")
    for mode in ['raw','selected']:
        table+=['',f'## {mode} contrasts','','| Contrast | Difference | Lower 95% | Upper 95% |','|---|---:|---:|---:|']
        for name,b in bootstrap[mode].items():table.append(f"| {name} | {b['difference']:+.4f} | {b['lower_95']:+.4f} | {b['upper_95']:+.4f} |")
    (out/'TABLES.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8));colors=['#2475b5','#ce7623','#31935c']
    for ax,mode in zip(axes,['raw','selected']):
        for i,kind in enumerate(cfg['variants']):
            for offset,objective,marker in [(-.12,'pointwise','o'),(.12,'pairwise','s')]:
                m=summaries[objective+'_'+kind][mode]['primary']
                ax.errorbar(i+offset,m['mean'],yerr=m['sd'],color=colors[i],marker=marker,linestyle='none',capsize=4,label=objective if i==0 else None)
        ax.axhline(summaries['distance'][mode]['primary']['mean'],color='#333',ls='--',label='Distance only')
        ax.set_xticks(range(3),labels=['MLP','CNN','FNO']);ax.set_ylabel('Macro matched concordance');ax.grid(axis='y',alpha=.2)
        ax.set_title('Unshrunk correction (gamma = 1)' if mode=='raw' else 'Gamma chosen on dev (includes 0)');ax.legend(fontsize=8)
    limits=(min(ax.get_ylim()[0] for ax in axes),max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:ax.set_ylim(limits)
    fig.text(.5,.02,'45 genes / 397 comparisons. Mean ± seed SD (3 seeds). Gene bootstrap intervals reported separately.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'objective_comparison.png');fig.savefig(out/'objective_comparison.svg');plt.close(fig)
    print(json.dumps({'verification':verification,'models':summaries,'bootstrap':bootstrap},indent=2))


if __name__=='__main__':main()
