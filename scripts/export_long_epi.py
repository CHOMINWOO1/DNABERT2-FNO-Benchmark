"""Export verified long-context pilot records and standalone scientific figures."""
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    root=Path('runs/long_epi_pilot'); output=Path('docs/long_epi_results')
    assert read(root/'status.json')['phase']=='complete'
    assert read(root/'verification.json')['passed']
    output.mkdir(parents=True,exist_ok=True)
    manifest=[]
    def copy(path,relative):
        dest=output/relative; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        manifest.append({'source':str(path),'export':str(relative),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    for name in ['protocol.json','status.json','summary.json','training_complete.json','verification.json']:
        copy(root/name,Path(name))
    if (root/'bootstrap.json').exists():
        copy(root/'bootstrap.json',Path('bootstrap.json'))
    if (root/'pytest.txt').exists():
        copy(root/'pytest.txt',Path('pytest.txt'))
    for path in root.glob('bp*/*.json'):
        copy(path,path.relative_to(root))
    for path in root.glob('bp*/trials/*/*.json'):
        copy(path,path.relative_to(root))
    for path in root.glob('bp*/results/*/*'):
        if path.suffix in {'.csv','.json'}:
            copy(path,path.relative_to(root))
    copy(Path('data/long_epi_pilot/source.json'),Path('data_source.json'))
    copy(Path('data/GUE_v2_source/lfs_verification.json'),Path('lfs_verification.json'))
    for path in Path('runs/long_context_probe').glob('*.json'):
        copy(path,Path('capacity_probe')/path.name)
    result=read(root/'summary.json'); plan=result['plan']; lengths=plan['lengths_bp']
    variants=plan['variants']; colors={'cnn':'#429c8a','fno':'#dc8a2d','lora':'#2677ad'}
    names={'cnn':'Frozen + CNN','fno':'Frozen + FNO','lora':'LoRA'}
    rows=[]; costs={}
    for length in lengths:
        cache=read(root/f'bp{length}/feature_cache.json')
        cache_peak=max(r['peak_memory']['allocated_mib'] for r in cache.values())
        costs[str(length)]={'shared_cache_seconds':sum(r['generation_seconds'] for r in cache.values()),
                            'shared_cache_peak_mib':cache_peak,
                            'cache_float32_payload_bytes':sum(int(np.prod(r['shape']))*4 for r in cache.values()),'variants':{}}
        for variant in variants:
            records=[read(p) for p in (root/f'bp{length}/results').glob(f'{variant}_seed*/result.json')]
            costs[str(length)]['variants'][variant]={
                'train_seconds_mean':float(np.mean([r['training']['train_seconds'] for r in records])),
                'train_seconds_sd':float(np.std([r['training']['train_seconds'] for r in records],ddof=1)),
                'mean_train_epoch_seconds':float(np.mean([r['training']['mean_train_epoch_seconds'] for r in records])),
                'pipeline_peak_mib':max([r['training']['peak_memory']['allocated_mib'] for r in records]+
                                         ([cache_peak] if variant!='lora' else [])),
                'inference_samples_per_second_mean':float(np.mean([r['inference']['samples_per_second'] for r in records])),
                'best_epochs':{str(r['seed']):r['training']['best_epoch'] for r in records},
                'completed_epochs':{str(r['seed']):r['training']['epochs_completed'] for r in records},
                'merged_mcc_mean':float(np.mean([r['lora_merge']['test']['mcc'] for r in records])) if variant=='lora' else None,
                'merged_class_flips':{str(r['seed']):r['lora_merge']['changed_class_predictions'] for r in records} if variant=='lora' else None}
            for r in records:
                rows.append({'bp':length,'variant':variant,'seed':r['seed'],**r['test'],
                             'best_epoch':r['training']['best_epoch'],'epochs':r['training']['epochs_completed'],
                             'train_seconds':r['training']['train_seconds'],
                             'train_peak_mib':r['training']['peak_memory']['allocated_mib'],
                             'inference_samples_per_second':r['inference']['samples_per_second']})
    with (output/'all_runs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'costs.json').write_text(json.dumps(costs,indent=2)+'\n')
    (output/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    table=['# 긴 context 실험 수치표','',
           '동일한 EPI pair와 split을 사용한 3개 학습 seed의 평균 ± sample SD. SD는 신뢰구간이 아니다.','',
           '| bp | 방법 | MCC | Accuracy | F1 | ROC-AUC | PR-AUC | AP |',
           '|---:|---|---:|---:|---:|---:|---:|---:|']
    for n in lengths:
        for variant in variants:
            metrics=result['by_length'][f'bp{n}']['summary'][variant]['metrics']
            formatted=[f"{metrics[m]['mean']:.4f} ± {metrics[m]['sd']:.4f}"
                       for m in ['mcc','accuracy','f1','roc_auc','pr_auc','average_precision']]
            table.append(f"| {n} | {names[variant]} | "+' | '.join(formatted)+' |')
    table += ['', '## 실행 비용', '',
              '학습 시간은 feature 추출·dev 평가·checkpoint 저장을 제외한 training loop 실측이다. '
              'Peak는 PyTorch allocated 기준이며 frozen 방법은 표현 추출과 학습 중 큰 값을 사용한다. '
              '추론은 encoder와 분류기 및 GPU 전송을 포함하고 tokenization·disk 읽기는 제외한다. LoRA는 병합 후 추론한다.', '',
              '| bp | 방법 | 학습 s/run | 평균 s/epoch | Peak MiB | 추론 pair/s | 선택 epoch (42/43/44) | 실행 epoch (42/43/44) |',
              '|---:|---|---:|---:|---:|---:|---|---|']
    for n in lengths:
        for variant in variants:
            c=costs[str(n)]['variants'][variant]
            selected='/'.join(str(c['best_epochs'][str(s)]) for s in plan['seeds'])
            completed='/'.join(str(c['completed_epochs'][str(s)]) for s in plan['seeds'])
            table.append(f"| {n} | {names[variant]} | {c['train_seconds_mean']:.1f} ± {c['train_seconds_sd']:.1f} | "
                         f"{c['mean_train_epoch_seconds']:.1f} | {c['pipeline_peak_mib']:.1f} | "
                         f"{c['inference_samples_per_second_mean']:.1f} | {selected} | {completed} |")
    table += ['', '## 공유 feature cache', '',
              '각 길이의 train/dev/test 전체를 한 번 추출하여 CNN/FNO가 공유한다. Float32 payload만 계산한 저장량이며 index·metadata는 제외한다.', '',
              '| bp | 추출 초 | Feature GiB |', '|---:|---:|---:|']
    for n in lengths:
        c=costs[str(n)]
        table.append(f"| {n} | {c['shared_cache_seconds']:.1f} | {c['cache_float32_payload_bytes']/2**30:.3f} |")
    if (root/'bootstrap.json').exists():
        bootstrap=read(root/'bootstrap.json')
        table += ['', '## Paired bootstrap 참고 구간', '', bootstrap['limitation'], '',
                  '| MCC contrast | 95% 하한 | 95% 상한 |', '|---|---:|---:|']
        for contrast, interval in bootstrap['contrasts'].items():
            table.append(f"| {contrast} | {interval['lower_95']:+.4f} | {interval['upper_95']:+.4f} |")
    (output/'TABLES.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.8))
    for variant in variants:
        metrics=[result['by_length'][f'bp{n}']['summary'][variant]['metrics']['mcc'] for n in lengths]
        axes[0].errorbar(lengths,[r['mean'] for r in metrics],yerr=[r['sd'] for r in metrics],
                         color=colors[variant],label=names[variant],marker='o',capsize=4,lw=2)
        for seed in plan['seeds']:
            axes[0].plot(lengths,[r['per_seed'][str(seed)] for r in metrics],color=colors[variant],alpha=.2,lw=.8)
        cost=[costs[str(n)]['variants'][variant] for n in lengths]
        axes[1].errorbar(lengths,[r['train_seconds_mean'] for r in cost],yerr=[r['train_seconds_sd'] for r in cost],
                         color=colors[variant],marker='o',capsize=4,lw=2)
        axes[2].plot(lengths,[r['pipeline_peak_mib'] for r in cost],color=colors[variant],marker='o',lw=2)
    for ax,title,ylabel in zip(axes,['A   Test performance','B   Training cost per run','C   Peak allocated GPU memory'],
                               ['Test MCC','Training time (s)','MiB']):
        ax.set_xscale('log'); ax.set_xticks(lengths,labels=['1 kb','2 kb','5 kb']);ax.set_xlabel('Total paired input length')
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(alpha=.15)
    axes[0].axhline(0,color='#555555',ls=':',lw=1);axes[0].legend(frameon=False)
    fig.suptitle('DNABERT-2 on matched enhancer–promoter pairs',fontsize=14)
    fig.text(.5,.01,'Same 1,024 / 256 / 256 examples at all lengths; 3 training seeds. Bars show sample SD, not confidence intervals.\n'
             'Training time excludes shared feature extraction; frozen-model memory includes extraction (CNN/FNO overlap). Fixed-LR exploratory pilot.',
             ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.1,1,.96))
    fig.savefig(output/'length_comparison.png',bbox_inches='tight');fig.savefig(output/'length_comparison.svg',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,10),sharex=True,sharey=True)
    for i,n in enumerate(lengths):
        for j,variant in enumerate(variants):
            ax=axes[i,j]
            for p in (root/f'bp{n}/trials').glob(f'{variant}_seed*/training.json'):
                training=read(p);history=read(p.parent/'epochs.json')
                ax.plot([e['epoch'] for e in history],[e['dev']['mcc'] for e in history],color=colors[variant],alpha=.5)
                best=next(e for e in history if e['epoch']==training['best_epoch'])
                ax.scatter(best['epoch'],best['dev']['mcc'],color=colors[variant],s=20)
            ax.set_title(f'{names[variant]} | {n//1000} kb');ax.grid(alpha=.15)
    fig.supxlabel('Epoch');fig.supylabel('Validation MCC');fig.suptitle('Validation learning curves; dots mark selected checkpoints')
    fig.tight_layout();fig.savefig(output/'learning_curves.png',bbox_inches='tight');plt.close(fig)
    print('Exported',len(manifest),'source artifacts and figures to',output.resolve())


if __name__=='__main__':
    main()
