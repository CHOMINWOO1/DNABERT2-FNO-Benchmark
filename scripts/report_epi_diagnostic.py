"""Validate every diagnostic prediction, bootstrap paired contrasts, export evidence."""
import csv
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np
from verify_study import verify_predictions
from prepare_long_epi import canonical,center


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvrows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))


def main():
    root=Path('runs/epi_diagnostic'); out=Path('docs/epi_diagnostic_results')
    status=read(root/'status.json');assert status['phase']=='complete'
    protocol=read(root/'protocol.json');result=read(root/'summary.json');selection=read(root/'selection.json')
    for p,digest in protocol['code_sha256'].items():assert sha(p)==digest
    assert sha('data/epi_diagnostic_holdout/source.json')==protocol['holdout_manifest_sha256']
    assert sha('data/epi_diagnostic_holdout/test.csv')==protocol['holdout_data_sha256']
    source=read('data/epi_diagnostic_holdout/source.json'); prior=read('data/long_epi_pilot/source.json')
    assert sha('data/long_epi_pilot/source.json')==source['prior_source_manifest_sha256']
    raw={}
    for split,record in source['source_files'].items():
        assert sha(record['path'])==record['sha256']
        raw[split]=list(csv.DictReader(gzip.open(record['path'],'rt')))
    fresh=csvrows('data/epi_diagnostic_holdout/test.csv')
    expected=[raw[r['source_split']][r['source_row']] for r in source['assignment']]
    assert [(r['sequence'],int(r['label'])) for r in fresh]==[(r['enhancer']+r['promoter'],int(r['label'])) for r in expected]
    old=[raw[r['source_split']][r['source_row']] for group in prior['assignment'].values() for r in group]
    for role,widths in [('enhancer',[600,1200,3000]),('promoter',[400,800,2000])]:
        for width in widths:
            assert not ({canonical(center(r[role],width)) for r in old}&{canonical(center(r[role],width)) for r in expected})
    assert len(fresh)==1024 and sum(int(r['label']) for r in fresh)==512
    trained=read(root/'training_complete.json')
    assert len(trained['all_training'])==44 and len(trained['final'])==12
    assert len(list(root.glob('arms/*/trials/*/training.json')))==44
    for t in trained['all_training']:
        assert t['frozen_weights_unchanged'] and not t['test_evaluated']
    for variant in ['fno','cnn']:
        candidates=[c for c in selection['search'] if c['variant']==variant]
        assert len(candidates)==9
        for c in candidates:
            assert {r['settings']['seed'] for r in c['records']}=={42,43}
            assert np.isclose(np.mean([r['best_dev_mcc'] for r in c['records']]),c['mean_best_dev_mcc'])
        chosen=sorted(candidates,key=lambda c:(-c['mean_best_dev_mcc'],c['lr'],c['structure']))[0]
        assert chosen['arm']==selection['selected'][variant]['arm']
    labels={'new_holdout':np.array([int(r['label']) for r in fresh]),
            'previous_test':np.array([int(r['label']) for r in csvrows('data/long_epi_pilot/bp5000/test.csv')])}
    assert len(result['results'])==33
    predicted={}
    for r in result['results']:
        path=root/'results'/r['split']/f"{r['model']}_seed{r['seed']}"
        assert sha(Path(r['training']['path'])/'best.pt')==r['checkpoint_sha256']
        p=verify_predictions(path/'predictions.csv',labels[r['split']],r['metrics'])
        if r['split']=='new_holdout':predicted.setdefault(r['model'],{})[r['seed']]=p>=.5
    for split,groups in result['summary'].items():
        for model,metrics in groups.items():
            group=[r for r in result['results'] if r['split']==split and r['model']==model]
            assert len(group)==3
            assert {r['seed'] for r in group}=={42,43,44}
            for metric,record in metrics.items():
                values=[r['metrics'][metric] for r in group]
                assert np.isclose(np.mean(values),record['mean'],atol=1e-12)
                assert np.isclose(np.std(values,ddof=1),record['sd'],atol=1e-12)
    baseline_parity={}
    for variant,structure in [('fno',16),('cnn',3)]:
        for seed in [42,43]:
            folder=f'{variant}_seed{seed}_lr0.0003'
            current=read(root/f'arms/{variant}_s{structure}_lr0.0003/trials/{folder}/epochs.json')
            previous=read(Path('runs/long_epi_pilot/bp5000/trials')/folder/'epochs.json')
            assert len(current)==len(previous)
            loss_delta=max(abs(a['train_loss']-b['train_loss']) for a,b in zip(current,previous))
            dev_delta=max(abs(a['dev']['mcc']-b['dev']['mcc']) for a,b in zip(current,previous))
            assert loss_delta<1e-6 and dev_delta<1e-6
            baseline_parity[f'{variant}_seed{seed}']={'max_train_loss_delta':loss_delta,'max_dev_mcc_delta':dev_delta}
    verification={'passed':True,'training_trials':44,'evaluations':33,'new_holdout_rows':1024,'prior_test_rows':256,
                  'analysis_script_sha256':sha(__file__),
                  'prior_default_training_parity':baseline_parity,
                  'checks':['code/data/checkpoint hashes','raw sequence and label reconstruction','all prior EPI component views exact/RC disjoint',
                            'dev-only search selection recomputed','frozen weights unchanged','all saved probability metrics independently recomputed','summary mean/sample SD']}
    (root/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    y=labels['new_holdout'].astype(bool);rng=np.random.default_rng(20260912)
    idx=np.concatenate([rng.choice(np.flatnonzero(y==label),size=(5000,int((y==label).sum())),replace=True) for label in [0,1]],axis=1)
    truth=y[idx][None,:,:]; boot={}
    for model,seeds in predicted.items():
        p=np.stack([seeds[s] for s in [42,43,44]])[:,idx]
        tp=(p&truth).sum(-1);tn=(~p&~truth).sum(-1);fp=(p&~truth).sum(-1);fn=(~p&truth).sum(-1)
        denom=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        boot[model]=np.divide(tp*tn-fp*fn,denom,out=np.zeros_like(denom),where=denom>0).mean(0)
    comparisons=[('fno_mean','cnn_mean'),('fno_attention','cnn_attention'),('fno_attention','fno_mean'),('cnn_attention','cnn_mean')]
    contrasts={f'{a}_minus_{b}':boot[a]-boot[b] for a,b in comparisons}
    contrasts['pooling_gain_fno_minus_cnn']=(boot['fno_attention']-boot['fno_mean'])-(boot['cnn_attention']-boot['cnn_mean'])
    bootstrap={'seed':20260912,'replicates':5000,'scope':'New holdout; paired stratified row percentile bootstrap; average MCC of three fixed trained seeds.',
               'limitation':'No retraining uncertainty, shared-component dependence adjustment, chromosome uncertainty or multiplicity correction.',
               'contrasts':{k:{'lower_95':float(np.quantile(v,.025)),'upper_95':float(np.quantile(v,.975))} for k,v in contrasts.items()}}
    (root/'bootstrap.json').write_text(json.dumps(bootstrap,indent=2)+'\n')
    out.mkdir(parents=True,exist_ok=True);manifest=[]
    files=list(root.glob('*.json'))+list(root.glob('arms/*/trials/*/*.json'))+list(root.glob('results/*/*/*'))
    for path in files:
        if path.suffix not in ['.json','.csv']:continue
        dest=out/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        manifest.append({'source':str(path),'export':str(dest.relative_to(out)),'sha256':sha(dest)})
    shutil.copyfile('data/epi_diagnostic_holdout/source.json',out/'holdout_source.json')
    manifest.append({'source':'data/epi_diagnostic_holdout/source.json','export':'holdout_source.json','sha256':sha(out/'holdout_source.json')})
    (out/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    numeric=[{'model':r['model'],'split':r['split'],'seed':r['seed'],**r['metrics'],
              'trainable_parameters':r['training']['trainable_params'],'training_seconds':r['training']['train_seconds'],
              'best_epoch':r['training']['best_epoch'],'completed_epochs':r['training']['epochs_completed']} for r in result['results']]
    with (out/'all_evaluations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(numeric[0]));w.writeheader();w.writerows(numeric)
    table=['# EPI diagnostic numeric results','', '| Model | New holdout MCC | Prior test MCC | New holdout ROC-AUC | Train seconds/run | Parameters |',
           '|---|---:|---:|---:|---:|---:|']
    order=['prior_cnn','cnn_mean','cnn_attention','prior_fno','fno_mean','fno_attention','prior_lora']
    for model in order:
        m=result['summary']['new_holdout'][model];group=[r for r in result['results'] if r['model']==model and r['split']=='new_holdout']
        oldm=result['summary']['previous_test'].get(model)
        prior_text=f"{oldm['mcc']['mean']:.4f} ± {oldm['mcc']['sd']:.4f}" if oldm else 'historical anchor'
        table.append(f"| {model} | {m['mcc']['mean']:.4f} ± {m['mcc']['sd']:.4f} | {prior_text} | {m['roc_auc']['mean']:.4f} ± {m['roc_auc']['sd']:.4f} | "
                     f"{np.mean([r['training']['train_seconds'] for r in group]):.1f} | {group[0]['training']['trainable_params']} |")
    table+=['','## Search (mean pooling; validation only)','', '| Model | Structure | Width | LR | Mean best dev MCC | Selected |','|---|---:|---:|---:|---:|---|']
    for c in selection['search']:
        table.append(f"| {c['variant']} | {c['structure']} | {c['width']} | {c['lr']} | {c['mean_best_dev_mcc']:.4f} | {c['arm']==selection['selected'][c['variant']]['arm']} |")
    table+=['','## Paired bootstrap on new holdout','', '| Contrast | 95% lower | 95% upper |','|---|---:|---:|']
    for name,interval in bootstrap['contrasts'].items():table.append(f"| {name} | {interval['lower_95']:+.4f} | {interval['upper_95']:+.4f} |")
    (out/'TABLES.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,ax=plt.subplots(figsize=(10,5))
    names=['Prior CNN','Tuned CNN\nmean','Tuned CNN\nattention','Prior FNO','Tuned FNO\nmean','Tuned FNO\nattention','Prior LoRA']
    colors=['#aaa','#429c8a','#429c8a','#aaa','#dc8a2d','#dc8a2d','#2677ad']
    for i,model in enumerate(order):
        m=result['summary']['new_holdout'][model]['mcc']
        ax.errorbar(i,m['mean'],yerr=m['sd'],fmt='o',color=colors[i],capsize=5)
        ax.scatter(np.array([i-.09,i,i+.09]),[m['per_seed'][str(s)] for s in [42,43,44]],s=15,color=colors[i],alpha=.45)
    ax.set_xticks(range(len(order)),labels=names);ax.set_ylabel('MCC');ax.grid(axis='y',alpha=.2)
    ax.set_title('5 kb EPI: sealed unused holdout (1,024 pairs)')
    fig.text(.5,.01,'Mean ± sample SD across 3 seeds; no test-based selection. Prior LoRA was not retuned.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(out/'holdout_comparison.png');fig.savefig(out/'holdout_comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    vmin=min(c['mean_best_dev_mcc'] for c in selection['search']);vmax=max(c['mean_best_dev_mcc'] for c in selection['search'])
    for ax,variant in zip(axes,['fno','cnn']):
        cs=[c for c in selection['search'] if c['variant']==variant]
        structures=sorted({c['structure'] for c in cs});lrs=sorted({c['lr'] for c in cs})
        matrix=np.array([[next(c['mean_best_dev_mcc'] for c in cs if c['structure']==s and c['lr']==lr) for lr in lrs] for s in structures])
        ax.imshow(matrix,vmin=vmin,vmax=vmax,cmap='viridis',aspect='auto')
        for i,s in enumerate(structures):
            for j,lr in enumerate(lrs):
                c=next(c for c in cs if c['structure']==s and c['lr']==lr)
                mark='*' if c['arm']==selection['selected'][variant]['arm'] else ''
                ax.text(j,i,f'{matrix[i,j]:.3f}{mark}',ha='center',va='center',color='white' if matrix[i,j]<(vmin+vmax)/2 else 'black')
        ax.set_xticks(range(3),labels=[f'{lr:g}' for lr in lrs]);ax.set_yticks(range(3),labels=structures)
        ax.set_xlabel('Learning rate');ax.set_ylabel('Modes' if variant=='fno' else 'Kernel');ax.set_title(variant.upper())
    fig.suptitle('Dev-only search: mean best MCC over 2 seeds (* selected)')
    fig.tight_layout();fig.savefig(out/'dev_search.png');plt.close(fig)
    print(json.dumps({'verification':verification,'selected':result['selection'],'bootstrap':bootstrap},indent=2))
    print((out/'TABLES.md').read_text(encoding='utf-8'))


if __name__=='__main__':main()
