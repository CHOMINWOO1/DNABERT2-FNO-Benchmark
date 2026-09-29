"""Verify OOF isolation and saved scores; summarize distance-controlled mixer gains."""
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
import torch
from dnabert_fno import study
from dnabert_fno.data import sha256_file
from dnabert_fno.adaptation import load_trainable
from crispr_ranking_models import ranking_metrics
from prepare_matched_cv import matched_pairs
from verify_study import verify_predictions
from cache_matched_grid import gpu_load
import matched_cv_models as models


def read(p):return json.loads(Path(p).read_text())


def independent_matched(pairs,p):
    grouped=defaultdict(list)
    for r in pairs:
        x,y=float(p[r['positive']]),float(p[r['negative']])
        grouped[r['promoter_id']].append(1. if x>y else 0. if x<y else .5)
    return [{'promoter_id':k,'comparisons':len(v),'concordance':sum(v)/len(v)} for k,v in sorted(grouped.items())]


def main():
    torch.set_num_threads(4)
    root=Path('runs/matched_cv');out=Path('docs/matched_cv_results');data=Path('data/crispr_matched_cv')
    status=read(root/'status.json');assert status['phase']=='complete'
    protocol=read(root/'protocol.json');result=read(root/'summary.json');source=read(data/'source.json');plan=protocol['plan']
    assert study.stable_hash(protocol)==status['study_id']==result['study_id']
    for path,digest in protocol['code_sha256'].items():assert sha256_file(path)==digest
    assert sha256_file('docs/MATCHED_CV_PROTOCOL.md')==protocol['protocol_document_sha256']
    assert sha256_file(data/'source.json')==protocol['data_manifest_sha256']
    for name,digest in source['files_sha256'].items():assert sha256_file(data/f'{name}.json')==digest
    rows=read(data/'rows.json');pairs=read(data/'matched_pairs.json');sensitivity=read(data/'sensitivity_pairs.json')
    assert matched_pairs(rows,1.25)==pairs
    for ratio,ps in sensitivity.items():assert matched_pairs(rows,float(ratio))==ps
    by_promoter=defaultdict(list)
    for row in rows:by_promoter[row['promoter_id']].append(row)
    for ratio,actual in [('1.25',pairs),*sensitivity.items()]:
        expected=set()
        for group in by_promoter.values():
            for positive in [x for x in group if x['label']==1]:
                for negative in [x for x in group if x['label']==0]:
                    if max(positive['distance'],negative['distance'])<=float(ratio)*min(positive['distance'],negative['distance']):
                        expected.add((positive['global_id'],negative['global_id']))
        observed={(p['positive'],p['negative']) for p in actual}
        assert len(observed)==len(actual) and observed==expected
    assert len(rows)==10273 and len(pairs)==397
    prior=read('data/crispr_ranking/source.json');assert sha256_file('data/crispr_ranking/source.json')==source['prior_data_manifest_sha256']
    old=[]
    for s,digest in prior['sha256'].items():
        assert sha256_file(f'data/crispr_ranking/{s}.json')==digest
        old.extend(read(f'data/crispr_ranking/{s}.json'))
    old.sort(key=lambda r:r['source_row'])
    for a,b in zip(rows,old):assert {k:v for k,v in a.items() if k not in ['global_id','fold']}==b
    rc=lambda s:min(s,s.translate(str.maketrans('ACGTN','TGCAN'))[::-1])
    for a in range(5):
        for b in range(a):
            ra=[r for r in rows if r['fold']==a];rb=[r for r in rows if r['fold']==b]
            for k in ['chrom','gene','promoter_id']:assert not {r[k] for r in ra}&{r[k] for r in rb}
            assert not {rc(r[k]) for r in ra for k in ['enhancer','promoter']}&{rc(r[k]) for r in rb for k in ['enhancer','promoter']}
    cache=read(root/'grid_cache/manifest.json');assert cache['encoder_hash_before']==cache['encoder_hash_after']
    assert sha256_file(root/'grid_cache/grid.npy')==cache['sha256']
    seqs=sorted({r[k] for r in rows for k in ['enhancer','promoter']});assert study.stable_hash(seqs)==cache['identity']['sequence_hash']
    index={s:i for i,s in enumerate(seqs)};models.CACHE=gpu_load(root/'grid_cache/grid.npy')
    selection=read(root/'selection.json');training=read(root/'training_complete.json')
    assert len(training['all_training'])==105 and len(training['final'])==45 and not selection['test_used']
    for r in training['all_training']:
        history=read(Path(r['path'])/'epochs.json');best=-float('inf');epoch=None
        for h in history:
            score=h['dev']['matched_concordance']
            if score>best+protocol['base_config']['min_delta']:best=score;epoch=h['epoch']
        assert r['best_dev_mcc']==best and r['best_epoch']==epoch and not r['test_evaluated']
        checkpoint=torch.load(Path(r['path'])/'best.pt',map_location='cpu',weights_only=True)
        assert checkpoint['epoch']==epoch and checkpoint['settings']==r['settings']
    for fold in range(5):
        cfg=training['fold_configs'][str(fold)];train=[r for r in rows if r['fold'] not in [fold,(fold+1)%5]]
        distances=np.log10([r['distance'] for r in train]);n=len(train);pos=sum(r['label'] for r in train)
        assert cfg['outer_test_fold']==fold and cfg['inner_dev_fold']==(fold+1)%5
        assert np.isclose(cfg['distance_mean'],distances.mean(),atol=1e-12,rtol=0) and np.isclose(cfg['distance_sd'],distances.std(),atol=1e-12,rtol=0)
        np.testing.assert_allclose(cfg['class_weights'],[n/(2*(n-pos)),n/(2*pos)],atol=1e-12,rtol=0)
        for variant in plan['variants']:
            s=selection['folds'][str(fold)][variant];assert len(s['candidates'])==3
            for c in s['candidates']:
                assert {r['settings']['seed'] for r in c['records']}=={42,43}
                assert np.isclose(c['mean_best_dev_matched_concordance'],np.mean([r['best_dev_mcc'] for r in c['records']]),atol=1e-12,rtol=0)
            assert sorted(s['candidates'],key=lambda c:(-c['mean_best_dev_matched_concordance'],c['lr']))[0]==s['selected']
        for seed in plan['final_seeds']:
            assert len({r['training']['head_initial_sha256'] for r in training['final'] if r['fold']==fold and r['training']['settings']['seed']==seed})==1
    oof={};seen={};assert len(result['results'])==50
    for number,r in enumerate(result['results'],1):
        fold=r['fold'];name=r['model'];seed=r['seed'];test=[x for x in rows if x['fold']==fold]
        dest=root/'results'/f'fold{fold}'/(name if seed is None else f'{name}_seed{seed}')
        assert sha256_file(dest/'predictions.csv')==r['prediction_sha256']
        p=verify_predictions(dest/'predictions.csv',[x['label'] for x in test],r['metrics'])
        with (dest/'predictions.csv').open(newline='') as f:predrows=list(csv.DictReader(f))
        assert [int(x['global_id']) for x in predrows]==[x['global_id'] for x in test]
        globalp={x['global_id']:p[i] for i,x in enumerate(test)}
        matched=independent_matched([x for x in pairs if x['fold']==fold],globalp)
        assert abs(np.mean([g['concordance'] for g in matched])-r['metrics']['matched_concordance'])<1e-12
        more,_=ranking_metrics(test,p)
        for k,v in more.items():assert abs(v-r['metrics'][k])<1e-12
        if seed is not None:
            cfg=training['fold_configs'][str(fold)];checkpoint=Path(r['training']['path'])/'best.pt'
            assert sha256_file(checkpoint)==r['checkpoint_sha256']
            assert r['training']['settings']['lr']==selection['folds'][str(fold)][name]['selected']['lr']
            model=models.make_model(cfg,name,seed);load_trainable(model,torch.load(checkpoint,map_location='cpu',weights_only=True)['trainable'])
            assert sum(p.numel() for p in model.parameters())==models.count_params(name,models.width_for(name))
            ds=models.Dataset(test,index,cfg['distance_mean'],cfg['distance_sd'],pairs)
            _,again,_=models.evaluate(model,ds,models.collate,cfg)
            np.testing.assert_allclose(again['probability'],p,atol=1e-7,rtol=0);del model
        else:
            path=root/f'fold{fold}'/'distance.joblib';assert sha256_file(path)==r['checkpoint_sha256']
            simple=joblib.load(path);train=[x for x in rows if x['fold'] not in [fold,(fold+1)%5]]
            np.testing.assert_allclose(simple[0].mean_,np.log10([x['distance'] for x in train]).mean(),atol=1e-12,rtol=0)
            np.testing.assert_allclose(simple.predict_proba(np.log10([x['distance'] for x in test])[:,None])[:,1],p,atol=1e-12,rtol=0)
        key=(name,seed)
        if key not in oof:oof[key]=np.zeros(len(rows));seen[key]=np.zeros(len(rows),dtype=int)
        indices=[x['global_id'] for x in test];oof[key][indices]=p;seen[key][indices]+=1
        study.atomic_json(root/'verification_progress.json',{'completed_evaluations':number,'total':50})
        if number%10==0:print(f'Verified {number}/50',flush=True)
    assert all(np.all(x==1) for x in seen.values())
    costs={}
    for name in plan['variants']:
        trials=[r for r in training['all_training'] if r['settings']['variant']==name]
        chosen=[r['training'] for r in training['final'] if r['training']['settings']['variant']==name]
        epochs=sum(r['epochs_completed'] for r in trials)
        evaluations=[r for r in result['results'] if r['model']==name]
        assert len(trials)==35 and len(chosen)==15 and len(evaluations)==15
        costs[name]={'trainable_params':sorted({r['trainable_params'] for r in trials}),
                     'width':protocol['widths'][name],'trials':len(trials),'selected_checkpoints':len(chosen),
                     'total_epochs':epochs,'train_seconds':sum(r['train_seconds'] for r in trials),
                     'dev_seconds':sum(r['dev_seconds'] for r in trials),
                     'train_seconds_per_epoch':sum(r['train_seconds'] for r in trials)/epochs,
                     'peak_allocated_mib':max(r['peak_memory']['allocated_mib'] for r in trials),
                     'peak_reserved_mib':max(r['peak_memory']['reserved_mib'] for r in trials),
                     'evaluation_seconds':sum(r['seconds'] for r in evaluations),
                     'evaluation_rows_per_second':3*len(rows)/sum(r['seconds'] for r in evaluations)}
    study.atomic_json(root/'cost_summary.json',{'models':costs,'feature_cache':cache,
        'run_elapsed_seconds':status['elapsed_seconds'],
        'scope':'Training and evaluation use the fixed GPU grid cache. Evaluation includes ranking metrics and CPU work; it is not an end-to-end DNABERT throughput benchmark.'})
    summaries={};pergroups={};groups_reference=None
    for name in plan['variants']+['distance']:
        seeds=plan['final_seeds'] if name!='distance' else [None]
        values=[];full=[];senses={ratio:[] for ratio in sensitivity};group_arrays=[]
        for seed in seeds:
            p=oof[(name,seed)];g=independent_matched(pairs,p);group_arrays.append([x['concordance'] for x in g])
            if groups_reference is None:groups_reference=g
            assert [x['promoter_id'] for x in g]==[x['promoter_id'] for x in groups_reference]
            values.append(float(np.mean([x['concordance'] for x in g])))
            full.append(ranking_metrics(rows,p)[0]['macro_auroc'])
            for ratio,ps in sensitivity.items():senses[ratio].append(float(np.mean([x['concordance'] for x in independent_matched(ps,p)])))
        stat=lambda v:{'mean':float(np.mean(v)),'sd':float(np.std(v,ddof=1)) if len(v)>1 else None,'values':v}
        summaries[name]={'primary':stat(values),'unmatched_macro_auroc':stat(full),'sensitivity':{k:stat(v) for k,v in senses.items()}}
        pergroups[name]=np.array(group_arrays).mean(0)
    rng=np.random.default_rng(plan['bootstrap_seed']);genes={p['promoter_id']:p['gene'] for p in pairs}
    assert len(set(genes.values()))==len(groups_reference)==45
    bootidx=rng.integers(0,45,size=(plan['bootstrap_repeats'],45));bootstrap={}
    for a,b in [('fno','cnn'),('fno','mlp'),('fno','distance'),('cnn','mlp'),('mlp','distance')]:
        delta=pergroups[a]-pergroups[b];boots=delta[bootidx].mean(1)
        bootstrap[f'{a}_minus_{b}']={'difference':float(delta.mean()),'lower_95':float(np.quantile(boots,.025)),'upper_95':float(np.quantile(boots,.975))}
    study.atomic_json(root/'oof_summary.json',{'study_id':status['study_id'],'models':summaries,'bootstrap':bootstrap,
        'bootstrap_unit':'45 genes / promoter loci, equal weight; conditional on fitted models; does not include shared-training/fold/chromosome uncertainty',
        'primary_comparisons':397,'primary_promoters':45})
    verification={'passed':True,'study_id':status['study_id'],'analysis_script_sha256':sha256_file(__file__),'neural_trials':105,'evaluations':50,
        'oof_rows_per_model_seed':10273,'primary_promoters':45,'primary_comparisons':397,
        'checks':['source/code/cache/checkpoint hashes','caliper pairs independently reconstructed','chromosome/gene/exact-RC fold isolation',
                  'all dev-only selections and training-split normalizers','same initial classifier heads and parameter budgets',
                  'independent concordance and probability-metric recomputation','all neural/logistic predictions regenerated',
                  'every OOF row predicted exactly once by its held-out-fold model']}
    study.atomic_json(root/'verification.json',verification)
    out.mkdir(parents=True,exist_ok=True);manifest=[]
    paths=list(root.glob('*.json'))+list(root.glob('fold*/trials/*/*.json'))+list(root.glob('results/*/*/*'))+[root/'grid_cache/manifest.json']
    for p in paths:
        if p.suffix not in ['.json','.csv']:continue
        dest=out/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        manifest.append({'source':str(p),'export':str(dest.relative_to(out)),'sha256':sha256_file(dest)})
    for name in ['source','matched_pairs','sensitivity_pairs']:
        p=data/f'{name}.json';dest=out/f'data_{name}.json';shutil.copyfile(p,dest)
        manifest.append({'source':str(p),'export':dest.name,'sha256':sha256_file(dest)})
    study.atomic_json(out/'export_manifest.json',manifest)
    with (out/'oof_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','seed','global_id','source_row','fold','promoter_id','label','probability'])
        for (name,seed),p in oof.items():w.writerows([name,seed,r['global_id'],r['source_row'],r['fold'],r['promoter_id'],r['label'],p[r['global_id']]] for r in rows)
    with (out/'per_promoter_concordance.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['promoter_id','gene','comparisons','mlp','cnn','fno','distance'])
        for i,g in enumerate(groups_reference):w.writerow([g['promoter_id'],genes[g['promoter_id']],g['comparisons']]+[pergroups[k][i] for k in ['mlp','cnn','fno','distance']])
    fmt=lambda v:f"{v['mean']:.4f}"+(f" ± {v['sd']:.4f}" if v['sd'] is not None else '')
    table=['# Distance-matched OOF mixer comparison','','| Model | Primary ratio 1.25 | Strict ratio 1.1 | Ratio 1.5 | Unmatched macro AUROC |','|---|---:|---:|---:|---:|']
    for name,m in summaries.items():table.append(f"| {name} | {fmt(m['primary'])} | {fmt(m['sensitivity']['1.1'])} | {fmt(m['sensitivity']['1.5'])} | {fmt(m['unmatched_macro_auroc'])} |")
    table+=['','## Gene bootstrap','','| Contrast | Difference | 95% lower | 95% upper |','|---|---:|---:|---:|']
    for name,b in bootstrap.items():table.append(f"| {name} | {b['difference']:+.4f} | {b['lower_95']:+.4f} | {b['upper_95']:+.4f} |")
    (out/'TABLES.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(11,4.6));names=['mlp','cnn','fno','distance'];labels=['Token MLP','CNN','FNO','Distance']
    for i,name in enumerate(names):
        m=summaries[name]['primary'];axes[0].errorbar(i,m['mean'],yerr=m['sd'] or 0,fmt='o',capsize=5)
        if m['sd'] is not None:axes[0].scatter(np.array([i-.07,i,i+.07]),m['values'],alpha=.35,s=16)
    axes[0].set_xticks(range(4),labels=labels);axes[0].axhline(.5,color='#999',ls='--');axes[0].set_ylabel('Macro matched concordance');axes[0].set_title('Distance ratio ≤ 1.25: 45 promoters')
    for name,label in zip(names,labels):axes[1].plot([1.1,1.25,1.5],[summaries[name]['sensitivity']['1.1']['mean'],summaries[name]['primary']['mean'],summaries[name]['sensitivity']['1.5']['mean']],marker='o',label=label)
    axes[1].set_xticks([1.1,1.25,1.5],labels=['1.10\n32 loci','1.25\n45 loci','1.50\n52 loci']);axes[1].set_xlabel('Maximum distance ratio');axes[1].set_ylabel('Macro matched concordance');axes[1].legend(fontsize=8);axes[1].set_title('Different eligibility sets; fixed predictions')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.text(.5,.015,'Five chromosome folds. Neural: mean ± sample SD (3 seeds). Gene uncertainty is reported separately.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,1));fig.savefig(out/'matched_comparison.png');fig.savefig(out/'matched_comparison.svg');plt.close(fig)
    print(json.dumps({'verification':verification,'models':summaries,'bootstrap':bootstrap},indent=2))


if __name__=='__main__':main()
