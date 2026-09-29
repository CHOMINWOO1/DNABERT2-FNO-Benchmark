"""Verify the CRISPR ranking experiment, gene-cluster bootstrap, and evidence export."""
import csv
import gzip
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from dnabert_fno import study
from dnabert_fno.adaptation import load_trainable
from dnabert_fno.data import sha256_file,collate_features
from crispr_ranking_models import VARIANTS,make_model,design,simple_design,VectorDataset,evaluate,ranking_metrics
from verify_study import verify_predictions


def read(p):return json.loads(Path(p).read_text())


def independent_groups(rows,p):
    indices=defaultdict(list)
    for i,r in enumerate(rows):indices[r['promoter_id']].append(i)
    answer=[]
    for key,idx in sorted(indices.items()):
        y=np.array([rows[i]['label'] for i in idx]);v=p[idx]
        if len(np.unique(y))<2:continue
        answer.append({'promoter_id':key,'gene':rows[idx[0]]['gene'],'chrom':rows[idx[0]]['chrom'],
                       'n':len(idx),'positives':int(y.sum()),'auroc':float(roc_auc_score(y,v)),
                       'ap':float(average_precision_score(y,v)),'top1':float(y[v==v.max()].mean()),'positive_fraction':float(y.mean())})
    return answer


def main():
    torch.set_num_threads(4)
    root=Path('runs/crispr_ranking');out=Path('docs/crispr_ranking_results');status=read(root/'status.json')
    assert status['phase']=='complete';protocol=read(root/'protocol.json');cfg=protocol['config'];result=read(root/'summary.json')
    assert study.stable_hash(protocol)==status['study_id']==result['study_id']
    for p,digest in protocol['code_sha256'].items():assert sha256_file(p)==digest
    assert sha256_file('docs/CRISPR_RANKING_PROTOCOL.md')==protocol['protocol_document_sha256']
    source=read('data/crispr_ranking/source.json');assert sha256_file('data/crispr_ranking/source.json')==protocol['data_manifest_sha256']
    assert sha256_file('data/crispr_source/download_manifest.json')==source['source_manifest_sha256']
    rawpath=Path('data/crispr_source/EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz')
    assert sha256_file(rawpath)==source['source_sha256']
    with gzip.open(rawpath,'rt') as f:raw=list(csv.DictReader(f,delimiter='\t'))
    sequence={}
    for p,digest in source['sequence_response_sha256'].items():
        assert sha256_file(p)==digest
        response=read(p)
        for r in response['response']:
            chrom,span,strand=r['query'].split(':');a,b=span.split('..')
            assert r['id']==f'chromosome:GRCh38:{chrom}:{a}:{b}:{strand}'
            assert int(b)-int(a)+1==len(r['seq'])==1000
            sequence[r['query']]=r['seq'].upper()
    cross='data/crispr_ranking/ucsc_crosscheck.json';assert sha256_file(cross)==source['ucsc_crosscheck_sha256']
    assert all(r['exact_match'] for r in read(cross))
    rows={s:read(f'data/crispr_ranking/{s}.json') for s in ['train','dev','test']};allids=[]
    for s,rs in rows.items():
        assert sha256_file(f'data/crispr_ranking/{s}.json')==source['sha256'][s]
        for r in rs:
            original=raw[r['source_row']];allids.append(r['source_row'])
            assert r['label']==int(original['Regulated']=='TRUE') and r['gene']==original['measuredGeneEnsemblId']
            assert r['tss']==int(original['startTSS']) and r['chrom']==original['chrom']==original['chrTSS']
            assert r['distance']==abs((int(original['chromStart'])+int(original['chromEnd']))/2-r['tss'])
            for role in ['enhancer','promoter']:
                assert r[role]==sequence[r[f'{role}_query']]
                assert len(r[role])==1000 and r[role].count('N')<=100
            assert abs(r['enhancer_start']+500-r['tss'])>=1000
        f=read(root/f'features/{s}.json')
        assert f['study_id']==status['study_id'] and sha256_file(root/f'features/{s}.npz')==f['sha256']
        assert f['data_sha256']==source['sha256'][s] and f['encoder_sha256_before']==f['encoder_sha256_after']
    assert len(set(allids))==len(allids)
    assert set(allids)|{r['source_row'] for r in source['excluded']}==set(range(len(raw)))
    rc=lambda s:min(s,s.translate(str.maketrans('ACGTN','TGCAN'))[::-1])
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        for key in ['gene','promoter_id','enhancer_query','chrom']:assert not {r[key] for r in rows[a]}&{r[key] for r in rows[b]}
        assert not {rc(r[k]) for r in rows[a] for k in ['enhancer','promoter']}&{rc(r[k]) for r in rows[b] for k in ['enhancer','promoter']}
    train=rows['train'];d=np.log10([r['distance'] for r in train]);ytrain=np.array([r['label'] for r in train])
    assert abs(d.mean()-cfg['distance_mean'])<1e-12 and abs(d.std()-cfg['distance_sd'])<1e-12
    np.testing.assert_allclose(cfg['class_weights'],[len(train)/(2*(len(train)-ytrain.sum())),len(train)/(2*ytrain.sum())],rtol=0,atol=1e-12)
    selected=read(root/'selection.json');training=read(root/'training_complete.json')
    assert len(training['all_training'])==35 and len(training['final'])==15 and not selected['test_used']
    for r in training['all_training']:
        history=read(Path(r['path'])/'epochs.json');best=-float('inf');epoch=None
        for h in history:
            score=h['dev']['macro_auroc']
            if score>best+cfg['min_delta']:best=score;epoch=h['epoch']
        assert r['best_dev_mcc']==best and r['best_epoch']==epoch and not r['test_evaluated']
        checkpoint=torch.load(Path(r['path'])/'best.pt',map_location='cpu',weights_only=True)
        assert checkpoint['epoch']==epoch and checkpoint['settings']==r['settings']
    for variant,candidates in selected['search'].items():
        assert len(candidates)==3
        for c in candidates:
            assert {r['settings']['seed'] for r in c['records']}=={42,43}
            assert abs(np.mean([r['best_dev_mcc'] for r in c['records']])-c['mean_best_dev_macro_auroc'])<1e-12
        best=sorted(candidates,key=lambda c:(-c['mean_best_dev_macro_auroc'],c['lr']))[0]
        assert best==selected['selected'][variant]
        assert all(r['settings']['lr']==best['lr'] for r in training['final'] if r['settings']['variant']==variant)
    test=rows['test'];y=np.array([r['label'] for r in test]);groups_by_model={};probabilities={}
    with np.load(root/'features/test.npz') as z:features={k:z[k].copy() for k in z.files}
    for r in result['results']:
        name=r['model'];dest=root/'results'/(name if r['seed'] is None else f'{name}_seed{r["seed"]}')
        assert sha256_file(dest/'predictions.csv')==r['prediction_sha256']
        assert sha256_file(dest/'per_promoter.json')==r['per_promoter_sha256']
        p=verify_predictions(dest/'predictions.csv',y,r['metrics']);g=independent_groups(test,p)
        assert g==read(dest/'per_promoter.json')
        for metric,key in [('macro_auroc','auroc'),('macro_ap','ap'),('macro_top1','top1')]:
            assert abs(np.mean([v[key] for v in g])-r['metrics'][metric])<1e-12
        assert abs(np.mean([v['ap']-v['positive_fraction'] for v in g])-r['metrics']['macro_ap_lift'])<1e-12
        groups_by_model.setdefault(name,[]).append(g);probabilities.setdefault(name,[]).append(p)
        if r['seed'] is not None:
            checkpoint=Path(r['training']['path'])/'best.pt';assert sha256_file(checkpoint)==r['checkpoint_sha256']
            model=make_model(cfg,name,r['seed']);load_trainable(model,torch.load(checkpoint,map_location='cpu',weights_only=True)['trainable'])
            x=design(features,test,name,cfg['distance_mean'],cfg['distance_sd'])
            _,again,_=evaluate(model,VectorDataset(x,test,name),collate_features,cfg)
            np.testing.assert_allclose(again['probability'],p,rtol=0,atol=1e-7);del model
    for variant,r in selected['simple'].items():
        path=root/'simple'/variant/'model.joblib';assert sha256_file(path)==r['checkpoint_sha256']
        assert sorted(r['trials'],key=lambda t:(-t['dev']['macro_auroc'],t['C']))[0]==r['selected']
        model=joblib.load(path);np.testing.assert_allclose(model[0].mean_,simple_design(train,variant).mean(0),rtol=0,atol=1e-12)
        np.testing.assert_allclose(model.predict_proba(simple_design(test,variant))[:,1],probabilities[variant][0],rtol=0,atol=1e-12)
        m,_=ranking_metrics(rows['dev'],model.predict_proba(simple_design(rows['dev'],variant))[:,1])
        for k,v in m.items():assert abs(v-r['selected']['dev'][k])<1e-12
    for name,metrics in result['summary'].items():
        rs=[r for r in result['results'] if r['model']==name];assert len(rs)==(3 if name in VARIANTS else 1)
        for k,v in metrics.items():
            values=[r['metrics'][k] for r in rs];assert abs(np.mean(values)-v['mean'])<1e-12
            if len(rs)>1:assert abs(np.std(values,ddof=1)-v['sd'])<1e-12
            else:assert v['sd'] is None
    assert all(g['auroc']==.5 for seed in groups_by_model['promoter'] for g in seed)
    reference=groups_by_model['promoter'][0];genes=sorted({g['gene'] for g in reference})
    locusgenes=np.array([g['gene'] for g in reference]);plan=protocol['plan'];rng=np.random.default_rng(plan['bootstrap_seed'])
    counts=np.stack([(locusgenes==gene).astype(float) for gene in genes])
    indices=rng.integers(0,len(genes),size=(plan['bootstrap_repeats'],len(genes)))
    weights=counts[indices].sum(1);weights/=weights.sum(1,keepdims=True)
    estimates={};per_locus={}
    for name,seeds in groups_by_model.items():
        assert all([g['promoter_id'] for g in groups]==[g['promoter_id'] for g in reference] for groups in seeds)
        values=np.array([[g['auroc'] for g in groups] for groups in seeds]).mean(0)
        per_locus[name]=values;estimates[name]=weights@values
    contrasts=[('pair','promoter'),('pair','enhancer'),('enhancer_distance','enhancer'),('pair_distance','pair'),
               ('pair_distance','enhancer_distance'),('pair_distance','distance'),('enhancer_distance','distance'),('pair_distance','kmer4_distance')]
    intervals={}
    for a,b in contrasts:
        delta=estimates[a]-estimates[b]
        intervals[f'{a}_minus_{b}']={'difference':float((per_locus[a]-per_locus[b]).mean()),
            'lower_95':float(np.quantile(delta,.025)),'upper_95':float(np.quantile(delta,.975))}
    study.atomic_json(root/'bootstrap.json',{'seed':plan['bootstrap_seed'],'replicates':plan['bootstrap_repeats'],
        'scope':'Gene-cluster paired percentile bootstrap of mean within-promoter AUROC, conditional on trained models.',
        'genes':len(genes),'promoter_loci':len(reference),'contrasts':intervals,
        'limitation':'Does not include retraining or between-chromosome uncertainty; no multiple-comparison correction.'})
    verification={'passed':True,'study_id':status['study_id'],'analysis_script_sha256':sha256_file(__file__),
        'neural_trials':35,'simple_fits':15,'evaluations':18,'test_rows':len(test),'ranking_loci':len(reference),'ranking_genes':len(genes),
        'ranking_rows':sum(g['n'] for g in reference),'ranking_positives':sum(g['positives'] for g in reference),
        'checks':['source/sequence/data/code/feature/checkpoint hashes','raw labels and GRCh38 coordinates reconstructed',
                  'chromosome/gene/locus/exact-RC sequence split disjointness','encoder hashes unchanged',
                  'train-only distance scaling and class weights','dev-only best epoch and model selection',
                  'independently recomputed per-promoter and global metrics','all neural and logistic probabilities regenerated',
                  'promoter-only ranks exactly tied','summary mean and sample SD']}
    study.atomic_json(root/'verification.json',verification)
    out.mkdir(parents=True,exist_ok=True);manifest=[]
    paths=list(root.glob('*.json'))+list(root.glob('features/*.json'))+list(root.glob('neural/trials/*/*.json'))+list(root.glob('simple/*/*.json'))+list(root.glob('results/*/*'))
    for p in paths:
        if p.suffix not in ['.json','.csv']:continue
        dest=out/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        manifest.append({'source':str(p),'export':str(dest.relative_to(out)),'sha256':sha256_file(dest)})
    for p,name in [(Path('data/crispr_ranking/source.json'),'data_source.json'),(Path('data/crispr_source/download_manifest.json'),'download_manifest.json'),(Path('data/crispr_ranking/ucsc_crosscheck.json'),'ucsc_crosscheck.json')]:
        dest=out/name;shutil.copyfile(p,dest);manifest.append({'source':str(p),'export':name,'sha256':sha256_file(dest)})
    study.atomic_json(out/'export_manifest.json',manifest)
    numeric=[{'model':r['model'],'seed':r['seed'],**r['metrics']} for r in result['results']]
    with (out/'all_evaluations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(numeric[0]));w.writeheader();w.writerows(numeric)
    locus_records=[{'model':name,'promoter_id':reference[i]['promoter_id'],'gene':reference[i]['gene'],'chrom':reference[i]['chrom'],
                    'candidates':reference[i]['n'],'positives':reference[i]['positives'],'mean_seed_auroc':float(v)} for name,values in per_locus.items() for i,v in enumerate(values)]
    with (out/'per_promoter_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(locus_records[0]));w.writeheader();w.writerows(locus_records)
    order=VARIANTS+['distance','gc_distance','kmer4_distance']
    fmt=lambda v:f"{v['mean']:.4f}"+(f" ± {v['sd']:.4f}" if v['sd'] is not None else '')
    table=['# CRISPR candidate ranking results','','| Model | Within-promoter AUROC | Within-promoter AP | Global AP | Global MCC |','|---|---:|---:|---:|---:|']
    for name in order:
        m=result['summary'][name];table.append(f"| {name} | {fmt(m['macro_auroc'])} | {fmt(m['macro_ap'])} | {fmt(m['average_precision'])} | {fmt(m['mcc'])} |")
    table+=['','## Gene-cluster bootstrap','','| Contrast | Difference | 95% lower | 95% upper |','|---|---:|---:|---:|']
    for name,r in intervals.items():table.append(f"| {name} | {r['difference']:+.4f} | {r['lower_95']:+.4f} | {r['upper_95']:+.4f} |")
    (out/'TABLES.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    labels=['Promoter only','Enhancer only','Enhancer + promoter','Enhancer + distance','Enhancer + promoter + distance','Distance only','GC + distance','4-mer + distance']
    fig,ax=plt.subplots(figsize=(10,5.6))
    for i,name in enumerate(order):
        m=result['summary'][name]['macro_auroc'];color='#2677ad' if name in VARIANTS else '#c28535'
        ax.errorbar(m['mean'],i,xerr=m['sd'] or 0,fmt='o',color=color,capsize=4)
        if name in VARIANTS:ax.scatter(m['values'],[i-.08,i,i+.08],alpha=.3,s=15,color=color)
    ax.set_yticks(range(len(order)),labels=labels);ax.invert_yaxis();ax.axvline(.5,color='#999',ls='--');ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('Mean AUROC within the same promoter (0.5 = tied/random ranking)')
    ax.set_title(f'CRISPR enhancer ranking: {len(reference)} promoters on held-out chr1/3/5')
    fig.text(.5,.012,'Neural: mean ± sample SD across 3 seeds. Logistic: one deterministic fit. Uncertainty between genes is separate.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'ranking_comparison.png');fig.savefig(out/'ranking_comparison.svg');plt.close(fig)
    names=['enhancer','pair','enhancer_distance','pair_distance','distance']
    matrix=np.stack([per_locus[name] for name in names],axis=1)
    fig,ax=plt.subplots(figsize=(8,8));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    symbols={r['gene']:r['symbol'] for r in test}
    ax.set_yticks(range(len(reference)),labels=[symbols[g['gene']]+f" ({g['positives']}/{g['n']})" for g in reference],fontsize=8)
    ax.set_xticks(range(len(names)),labels=['Enhancer','E + P','E + distance','E + P + distance','Distance'],rotation=20,ha='right')
    ax.set_title('Per-promoter AUROC; labels show positive / tested candidates')
    fig.colorbar(im,ax=ax,label='Mean seed AUROC');fig.tight_layout();fig.savefig(out/'per_promoter_auroc.png');plt.close(fig)
    print(json.dumps({'verification':verification,'bootstrap':intervals,'summary':result['summary']},indent=2))


if __name__=='__main__':main()
