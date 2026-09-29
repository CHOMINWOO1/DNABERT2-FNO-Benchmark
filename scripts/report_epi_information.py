"""Independent saved-prediction checks, paired bootstrap, portable evidence export."""
import csv
import gzip
import json
import os
import shutil
from pathlib import Path
import joblib
import numpy as np
import torch
from dnabert_fno.data import sha256_file,collate_features
from dnabert_fno.adaptation import load_trainable
from dnabert_fno.metrics import binary_metrics
from dnabert_fno import study
from epi_information_models import NEURAL,composition_matrix,make_model,VectorDataset,feature_view
from prepare_long_epi import component_groups
from verify_study import verify_predictions


def read(p):return json.loads(Path(p).read_text())
def csvrows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))


def main():
    torch.set_num_threads(4)
    root=Path('runs/epi_information');out=Path('docs/epi_information_results')
    status=read(root/'status.json');assert status['phase']=='complete'
    protocol=read(root/'protocol.json');summary=read(root/'summary.json');select=read(root/'selection.json')
    assert study.stable_hash(protocol)==status['study_id']==summary['study_id']
    for p,digest in protocol['code_sha256'].items():assert sha256_file(p)==digest
    assert sha256_file('docs/EPI_INFORMATION_PROTOCOL.md')==protocol['protocol_document_sha256']
    assert sha256_file('data/epi_information/source.json')==protocol['data_manifest_sha256']
    source=read('data/epi_information/source.json');raw={}
    for p,digest in source['prior_manifest_sha256'].items():assert sha256_file(p)==digest
    for s,r in source['source_files'].items():
        assert sha256_file(r['path'])==r['sha256']
        raw[s]=list(csv.DictReader(gzip.open(r['path'],'rt')))
    rows={s:csvrows(f'data/epi_information/{s}.csv') for s in ['train','dev','test']}
    for s,rs in rows.items():
        assert sha256_file(f'data/epi_information/{s}.csv')==source['sha256'][s]
        expected=[raw[r['source_split']][r['source_row']] for r in source['assignment'][s]]
        assert [(r['enhancer'],r['promoter'],int(r['label'])) for r in rs]==[(r['enhancer'],r['promoter'],int(r['label'])) for r in expected]
        f=read(root/f'features/{s}.json')
        assert f['sha256']==sha256_file(root/f'features/{s}.npz')
        assert f['data_sha256']==source['sha256'][s] and f['study_id']==status['study_id']
        assert f['encoder_sha256_before']==f['encoder_sha256_after'] and f['encoder_frozen']
        with np.load(root/f'features/{s}.npz') as z:
            for key in z.files: assert np.isfinite(z[key]).all() and list(z[key].shape)==f['shapes'][key]
    prior=read('data/long_epi_pilot/source.json');oldfresh=read('data/epi_diagnostic_holdout/source.json')
    oldassign=[r for rs in prior['assignment'].values() for r in rs]+oldfresh['assignment']
    old=[raw[r['source_split']][r['source_row']] for r in oldassign]
    allrows=[r for rs in raw.values() for r in rs];group_overlap={};component_counts={}
    for role,widths in [('enhancer',[600,1200,3000]),('promoter',[400,800,2000])]:
        sequences=sorted({r[role] for r in allrows});mapping=dict(zip(sequences,component_groups(sequences,widths)))
        a={mapping[r[role]] for r in old};b={mapping[r[role]] for r in rows['test']}
        group_overlap[role]=len(a&b);assert not a&b
        component_counts[role]=len(b)
    y=np.array([int(r['label']) for r in rows['test']]);assert len(y)==1024 and sum(y)==512
    training=read(root/'training_complete.json');assert len(training['all_training'])==35 and len(training['final'])==15
    assert len(list(root.glob('neural/trials/*/training.json')))==35
    for r in training['all_training']:
        assert not r['test_evaluated']
        history=read(Path(r['path'])/'epochs.json')
        best=-float('inf');epoch=None
        for h in history:
            if h['dev']['mcc']>best+protocol['config']['min_delta']:best=h['dev']['mcc'];epoch=h['epoch']
        assert best==r['best_dev_mcc'] and epoch==r['best_epoch']
        state=torch.load(Path(r['path'])/'best.pt',map_location='cpu',weights_only=True)
        assert state['epoch']==epoch and state['settings']==r['settings']
    for model in NEURAL:
        candidates=select['search'][model];assert len(candidates)==3
        for c in candidates:
            assert {r['settings']['seed'] for r in c['records']}=={42,43}
            assert np.isclose(c['mean_best_dev_mcc'],np.mean([r['best_dev_mcc'] for r in c['records']]),atol=1e-12)
        best=sorted(candidates,key=lambda c:(-c['mean_best_dev_mcc'],c['lr']))[0]
        assert best==select['selected'][model]
        assert all(r['settings']['lr']==best['lr'] for r in training['final'] if r['settings']['variant']==model)
    probabilities={}
    with np.load(root/'features/test.npz') as z:testfeatures={k:z[k].copy() for k in z.files}
    for r in summary['results']:
        name=r['model'];dest=root/'results'/(name if r['seed'] is None else f'{name}_seed{r["seed"]}')
        assert sha256_file(dest/'predictions.csv')==r['predictions_sha256']
        p=verify_predictions(dest/'predictions.csv',y,r['metrics'])
        probabilities.setdefault(name,[]).append(p)
        if r['seed'] is not None:
            checkpoint=Path(r['training']['path'])/'best.pt'
            assert sha256_file(checkpoint)==r['checkpoint_sha256']
            model=make_model(protocol['config'],name,r['seed'])
            load_trainable(model,torch.load(checkpoint,map_location='cpu',weights_only=True)['trainable'])
            ds=VectorDataset(feature_view(testfeatures,name),y)
            _,again,_=study.evaluate_study(model,ds,collate_features,protocol['config'])
            np.testing.assert_allclose(again['probability'],p,rtol=0,atol=1e-7)
            del model
    for name,record in select['composition'].items():
        dest=root/'composition'/name
        assert sha256_file(dest/'model.joblib')==record['checkpoint_sha256']
        assert sorted(record['trials'],key=lambda t:(-t['dev']['mcc'],t['C']))[0]==record['selected']
        model=joblib.load(dest/'model.joblib')
        xtrain=composition_matrix(rows['train'],record['kind'],record['role'])
        np.testing.assert_allclose(model[0].mean_,xtrain.mean(0),rtol=0,atol=1e-12)
        x=composition_matrix(rows['test'],record['kind'],record['role'])
        np.testing.assert_allclose(model.predict_proba(x)[:,1],probabilities[name][0],rtol=0,atol=1e-12)
        xdev=composition_matrix(rows['dev'],record['kind'],record['role'])
        dev=binary_metrics([int(r['label']) for r in rows['dev']],model.predict_proba(xdev)[:,1])
        for key,value in dev.items(): assert abs(value-record['selected']['dev'][key])<1e-12
    for name,metrics in summary['summary'].items():
        rs=[r for r in summary['results'] if r['model']==name]
        assert len(rs)==(3 if name in NEURAL else 1)
        for key,stat in metrics.items():
            values=[r['metrics'][key] for r in rs]
            assert abs(float(np.mean(values))-stat['mean'])<1e-12
            if len(rs)>1: assert abs(float(np.std(values,ddof=1))-stat['sd'])<1e-12
            else: assert stat['sd'] is None
    verification={'passed':True,'study_id':status['study_id'],'analysis_script_sha256':sha256_file(__file__),
                  'neural_training_trials':35,'composition_fits':30,'evaluations':21,'test_rows':1024,
                  'test_prior_component_group_overlap':group_overlap,'test_unique_component_groups':component_counts,
                  'checks':['code/data/feature/checkpoint/prediction hashes','raw sequence and label reconstruction',
                            'transitive crop/RC groups disjoint from every prior EPI split','frozen encoder extraction hashes unchanged',
                            'best epoch and dev-only hyperparameter selection','saved probabilities independently recomputed metrics',
                            'all neural and logistic probabilities regenerated; scaler fitted on train only','summary mean and sample SD']}
    study.atomic_json(root/'verification.json',verification)
    plan=protocol['plan'];rng=np.random.default_rng(plan['bootstrap_seed']);truth=y.astype(bool)
    idx=np.concatenate([rng.choice(np.flatnonzero(y==v),size=(plan['bootstrap_repeats'],int((y==v).sum())),replace=True) for v in [0,1]],axis=1)
    t=truth[idx][None,:,:];boot={}
    for name,ps in probabilities.items():
        p=(np.stack(ps)>=.5)[:,idx]
        tp=(p&t).sum(-1);tn=(~p&~t).sum(-1);fp=(p&~t).sum(-1);fn=(~p&t).sum(-1)
        denom=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        boot[name]=np.divide(tp*tn-fp*fn,denom,out=np.zeros_like(denom),where=denom>0).mean(0)
    contrasts=[('separate_pair','enhancer'),('separate_pair','promoter'),('joint_pair','separate_pair'),
               ('joint_pair','enhancer'),('joint_pair','promoter'),('joint_pair','joint_global'),
               ('kmer4_pair','kmer4_enhancer'),('kmer4_pair','kmer4_promoter'),('joint_pair','kmer4_pair'),('joint_pair','gc_pair')]
    intervals={}
    for a,b in contrasts:
        delta=boot[a]-boot[b]
        intervals[f'{a}_minus_{b}']={'difference':summary['summary'][a]['mcc']['mean']-summary['summary'][b]['mcc']['mean'],
                                   'lower_95':float(np.quantile(delta,.025)),'upper_95':float(np.quantile(delta,.975))}
    study.atomic_json(root/'bootstrap.json',{'seed':plan['bootstrap_seed'],'replicates':plan['bootstrap_repeats'],
        'scope':'Paired label-stratified row percentile bootstrap; mean MCC of fixed neural seed models; single deterministic logistic model.',
        'limitations':'No retraining uncertainty, within-test component dependence or multiple-comparison adjustment.', 'contrasts':intervals})
    out.mkdir(parents=True,exist_ok=True);manifest=[]
    paths=list(root.glob('*.json'))+list(root.glob('features/*.json'))+list(root.glob('neural/trials/*/*.json'))+list(root.glob('composition/*/*.json'))+list(root.glob('results/*/*'))
    for p in paths:
        if p.suffix not in ['.json','.csv']:continue
        dest=out/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        manifest.append({'source':str(p),'export':str(dest.relative_to(out)),'sha256':sha256_file(dest)})
    p=Path('data/epi_information/source.json');dest=out/'data_source.json';shutil.copyfile(p,dest)
    manifest.append({'source':str(p),'export':dest.name,'sha256':sha256_file(dest)})
    study.atomic_json(out/'export_manifest.json',manifest)
    numeric=[{'model':r['model'],'seed':r['seed'],**r['metrics']} for r in summary['results']]
    with (out/'all_evaluations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(numeric[0]));w.writeheader();w.writerows(numeric)
    order=NEURAL+['gc_enhancer','gc_promoter','gc_pair','kmer4_enhancer','kmer4_promoter','kmer4_pair']
    table=['# Information ablation numeric results','','| Model | MCC | ROC-AUC |','|---|---:|---:|']
    fmt=lambda x:f"{x['mean']:.4f}"+(f" ± {x['sd']:.4f}" if x['sd'] is not None else '')
    for name in order:
        m=summary['summary'][name];table.append(f"| {name} | {fmt(m['mcc'])} | {fmt(m['roc_auc'])} |")
    table+=['','## Conditional paired bootstrap','','| Contrast | Mean difference | 95% lower | 95% upper |','|---|---:|---:|---:|']
    for name,r in intervals.items():table.append(f"| {name} | {r['difference']:+.4f} | {r['lower_95']:+.4f} | {r['upper_95']:+.4f} |")
    (out/'TABLES.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    names=['Enhancer only','Promoter only','Independent encoders + pair head','Joint encoder + region pools','Joint encoder + global pool',
           'GC: enhancer','GC: promoter','GC: pair','4-mer: enhancer','4-mer: promoter','4-mer: pair']
    fig,axes=plt.subplots(1,2,figsize=(12,6.5),sharey=True)
    for ax,metric in zip(axes,['mcc','roc_auc']):
        for i,name in enumerate(order):
            m=summary['summary'][name][metric];color='#2677ad' if name in NEURAL else '#c28535'
            ax.errorbar(m['mean'],i,xerr=m['sd'] if m['sd'] is not None else 0,fmt='o',color=color,capsize=4)
            if name in NEURAL: ax.scatter(m['values'],np.array([i-.10,i,i+.10]),color=color,alpha=.35,s=16)
        ax.set_xlabel('MCC' if metric=='mcc' else 'ROC-AUC');ax.grid(axis='x',alpha=.2)
        ax.axvline(0 if metric=='mcc' else .5,color='#999',ls='--',lw=.8)
    axes[0].set_yticks(range(len(order)),labels=names);axes[0].invert_yaxis()
    fig.suptitle('What information supports EPI classification? Fresh holdout: 1,024 pairs')
    fig.text(.5,.015,'Blue: frozen DNABERT-2 + MLP, mean ± sample SD (3 seeds). Orange: deterministic logistic regression.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(out/'information_comparison.png');fig.savefig(out/'information_comparison.svg');plt.close(fig)
    print(json.dumps({'verification':verification,'summary':summary['summary'],'bootstrap':intervals},indent=2))


if __name__=='__main__':main()
