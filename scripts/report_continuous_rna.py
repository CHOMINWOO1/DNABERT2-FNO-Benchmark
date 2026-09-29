"""Prespecified overlap-cluster bootstrap, length/perturbation tables and exports."""
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from dnabert_fno.study import atomic_json
from dnabert_fno.data import sha256_file


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def stat(values):return {'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1)),'values':[float(v) for v in values]}


def cluster_sums(values,groups):
    return np.array([values[g].sum(axis=0) for g in groups],dtype=np.float64)


def bootstrap_correlations(y,p,groups,weights,common=None):
    if common is None:
        n=weights@np.array([len(g) for g in groups],float)
        sy=weights@cluster_sums(y,groups);syy=weights@cluster_sums(y*y,groups)
    else:n,sy,syy=common
    matrix=np.concatenate([cluster_sums(p,groups),cluster_sums(p*p,groups),cluster_sums(y*p,groups)],axis=1)
    sums=weights@matrix;d=y.shape[1];sp,spp,syp=np.split(sums,[d,2*d],axis=1)
    numerator=syp-sy*sp/n[:,None]
    denominator=np.sqrt(np.maximum(syy-sy*sy/n[:,None],0)*np.maximum(spp-sp*sp/n[:,None],0))
    corr=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-15)
    return corr.mean(1)


def main():
    root=Path('runs/continuous_rna');out=Path('docs/continuous_rna_results');verification=read(root/'verification.json');assert verification['passed']
    assert sha256_file('scripts/verify_continuous_rna.py')==verification['verification_script_sha256']
    protocol=read(root/'protocol.json');cfg=protocol['config'];summary=read(root/'summary.json');training=read(root/'training_complete.json')
    data=Path(cfg['data_dir']);labels=np.load(data/'labels.npy');norm=np.load(root/'normalizers.npz');test_ids=summary['test_gene_ids'];rows=read(data/'rows.json')
    y=((labels-norm['label_mean'])/norm['label_sd'])[test_ids];mapping={v:i for i,v in enumerate(test_ids)}
    groups=[np.array([mapping[i] for i in ids]) for ids in read(data/'test_clusters.json')]
    assert len(groups)==533 and sum(map(len,groups))==982
    rng=np.random.default_rng(cfg['bootstrap_seed']);draws=rng.integers(0,len(groups),size=(cfg['bootstrap_repeats'],len(groups)))
    weights=np.array([np.bincount(x,minlength=len(groups)) for x in draws],dtype=np.float64)
    bygroup=defaultdict(list)
    for r in summary['results']:bygroup[(r['settings']['model'],r['settings']['bp'],r['condition'])].append(r)
    reports={};bootstrap={}
    with threadpool_limits(limits=4):
        common=(weights@np.array([len(g) for g in groups],float),weights@cluster_sums(y,groups),weights@cluster_sums(y*y,groups))
        number=0
        for (kind,bp,condition),records in bygroup.items():
            key=f'{kind}_{bp}_{condition}';per_seed={seed:[r for r in records if r['settings']['seed']==seed] for seed in cfg['seeds']}
            assert all(len(v)==(5 if condition in ['shuffle','swap'] else 1) for v in per_seed.values())
            metrics={metric:stat([np.mean([r['metrics'][metric] for r in per_seed[seed]]) for seed in cfg['seeds']]) for metric in ['macro_pearson','mse']}
            reports[key]={'model':kind,'bp':bp,'condition':condition,**metrics,'per_track_pearson':np.mean([r['metrics']['per_track_pearson'] for r in records],axis=0).tolist()}
            values=np.zeros(cfg['bootstrap_repeats'])
            for r in records:
                path=Path(r['prediction_path']);assert sha256_file(path)==r['prediction_sha256']
                values+=bootstrap_correlations(y,np.load(path),groups,weights,common)/len(records);number+=1
                atomic_json(root/'bootstrap_progress.json',{'prediction_matrices':number,'total':126})
            bootstrap[key]=values
            print(f'Bootstrapped {key} ({number}/126)',flush=True)
    definitions=[('fno_65536_intact',f'{k}_65536_intact') for k in ['cnn','attention']]
    definitions += [(f'{k}_65536_intact',f'{k}_{bp}_intact') for k in cfg['variants'] for bp in [4096,16384]]
    definitions += [(f'{k}_65536_intact',f'{k}_65536_{condition}') for k in cfg['variants'] for condition in ['mask','shuffle','swap']]
    assert len(definitions)==17
    contrasts={}
    for a,b in definitions:
        delta=bootstrap[a]-bootstrap[b]
        contrasts[f'{a}_minus_{b}']={'difference':reports[a]['macro_pearson']['mean']-reports[b]['macro_pearson']['mean'],
                                   'lower_95':float(np.quantile(delta,.025)),'upper_95':float(np.quantile(delta,.975))}
    result={'study_id':summary['study_id'],'groups':reports,'contrasts':contrasts,'bootstrap_clusters':533,'test_genes':982,'tracks':218,
            'bootstrap_repeats':cfg['bootstrap_repeats'],'bootstrap_seed':cfg['bootstrap_seed'],
            'analysis_script_sha256':sha256_file(__file__),'note':'Mean of training-seed correlations; shuffle/swap additionally average perturbation performance, not ensemble predictions. Intervals conditional on fitted models and chr8.'}
    atomic_json(root/'analysis_summary.json',result);np.savez_compressed(root/'bootstrap_samples.npz',**bootstrap)
    costs={}
    for kind in cfg['variants']:
        for bp in cfg['lengths_bp']:
            records=[r for r in training['all_training'] if r['settings']['model']==kind and r['settings']['bp']==bp]
            evals=[r for r in summary['results'] if r['settings']['model']==kind and r['settings']['bp']==bp and r['condition']=='intact']
            costs[f'{kind}_{bp}']={'params':sorted({r['params'] for r in records}),'epochs':sum(r['epochs_completed'] for r in records),
                'train_seconds':sum(r['train_seconds'] for r in records),'dev_seconds':sum(r['dev_seconds'] for r in records),
                'peak_allocated_mib':max(r['peak_memory']['allocated_mib'] for r in records),
                'cached_inference_seconds':float(np.mean([r['inference_seconds_cached'] for r in evals])),
                'cached_rows_per_second':len(test_ids)/np.mean([r['inference_seconds_cached'] for r in evals])}
    atomic_json(root/'cost_summary.json',{'models':costs,'encoder':read(root/'features/manifest.json'),'runner_seconds':read(root/'status.json')['elapsed_seconds']})
    out.mkdir(parents=True,exist_ok=True);manifest=[]
    paths=list(root.glob('*.json'))+list(root.glob('*.npz'))+list(root.glob('trials/*/*.json'))+list(root.glob('results/*/*'))+[root/'features/manifest.json']
    for p in paths:
        if p.suffix not in ['.json','.npy','.npz']:continue
        dest=out/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        manifest.append({'source':str(p),'export':str(dest.relative_to(out)),'sha256':sha256_file(dest)})
    for name in ['source.json','rows.json','tracks.json','test_clusters.json']:
        p=data/name;dest=out/f'data_{name}';shutil.copyfile(p,dest);manifest.append({'source':str(p),'export':dest.name,'sha256':sha256_file(dest)})
    atomic_json(out/'export_manifest.json',manifest)
    with (out/'per_track_pearson.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','bp','condition','track','pearson_mean']);tracks=read(data/'tracks.json')
        for r in reports.values():w.writerows([r['model'],r['bp'],r['condition'],track,value] for track,value in zip(tracks,r['per_track_pearson']))
    fmt=lambda x:f"{x['mean']:.4f} ± {x['sd']:.4f}"
    table=['# Continuous RNA results','','| Model | 4kb | 16kb | 64kb |','|---|---:|---:|---:|']
    for kind in cfg['variants']:table.append('| '+kind+' | '+' | '.join(fmt(reports[f'{kind}_{bp}_intact']['macro_pearson']) for bp in cfg['lengths_bp'])+' |')
    table+=['','## 64kb distal perturbations','','| Model | Intact | Mask | Shuffle | Swap |','|---|---:|---:|---:|---:|']
    for kind in cfg['variants']:table.append('| '+kind+' | '+' | '.join(fmt(reports[f'{kind}_65536_{c}']['macro_pearson']) for c in ['intact','mask','shuffle','swap'])+' |')
    table+=['','## Cluster bootstrap','','| Contrast | Difference | Lower 95% | Upper 95% |','|---|---:|---:|---:|']
    for name,b in contrasts.items():table.append(f"| {name} | {b['difference']:+.4f} | {b['lower_95']:+.4f} | {b['upper_95']:+.4f} |")
    (out/'TABLES.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8));colors=['#2475b5','#31935c','#ce7623']
    for kind,color in zip(cfg['variants'],colors):
        values=[reports[f'{kind}_{bp}_intact']['macro_pearson'] for bp in cfg['lengths_bp']]
        axes[0].errorbar(range(3),[v['mean'] for v in values],yerr=[v['sd'] for v in values],marker='o',color=color,label=kind,capsize=4)
        values=[reports[f'{kind}_65536_{c}']['macro_pearson'] for c in ['intact','mask','shuffle','swap']]
        axes[1].errorbar(range(4),[v['mean'] for v in values],yerr=[v['sd'] for v in values],marker='o',color=color,label=kind,capsize=4)
    axes[0].set_xticks(range(3),labels=['4kb','16kb','64kb']);axes[0].set_title('Same genes, different contiguous context')
    axes[1].set_xticks(range(4),labels=['Intact','Mask','Shuffle','Swap']);axes[1].set_title('64kb models: central 4kb preserved')
    limits=(min(ax.get_ylim()[0] for ax in axes),max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:ax.set_ylim(limits);ax.set_ylabel('Mean Pearson across 218 expression tracks');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.text(.5,.02,'982 test genes on chr8. Error bars: training-seed SD. Shuffle/swap: mean performance of 5 fixed perturbations.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(out/'continuous_comparison.png');fig.savefig(out/'continuous_comparison.svg');plt.close(fig)
    print(json.dumps({'groups':{k:v['macro_pearson'] for k,v in reports.items()},'contrasts':contrasts},indent=2))


if __name__=='__main__':main()
