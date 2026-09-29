"""Within-promoter ranking metrics and matched DNABERT feature heads."""
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from dnabert_fno.metrics import binary_metrics
from dnabert_fno.runtime import seed_everything,amp_context,move_batch,timed_start,elapsed
from dnabert_fno.train import loader_for,accumulation_denominator
from epi_information_models import composition_matrix

VARIANTS=['promoter','enhancer','pair','enhancer_distance','pair_distance']


def ranking_metrics(rows,probabilities):
    p=np.asarray(probabilities);y=np.array([r['label'] for r in rows]);groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['promoter_id']].append(i)
    per_group=[]
    for name,indices in sorted(groups.items()):
        idx=np.array(indices);truth=y[idx];score=p[idx]
        if truth.min()==truth.max():continue
        top=score==score.max()
        per_group.append({'promoter_id':name,'gene':rows[idx[0]]['gene'],'chrom':rows[idx[0]]['chrom'],
                          'n':len(idx),'positives':int(truth.sum()),'auroc':float(roc_auc_score(truth,score)),
                          'ap':float(average_precision_score(truth,score)),'top1':float(truth[top].mean()),
                          'positive_fraction':float(truth.mean())})
    assert per_group,'No mixed-label promoter loci'
    metrics=binary_metrics(y,p)
    metrics.update({'macro_auroc':float(np.mean([g['auroc'] for g in per_group])),
                    'macro_ap':float(np.mean([g['ap'] for g in per_group])),
                    'macro_ap_lift':float(np.mean([g['ap']-g['positive_fraction'] for g in per_group])),
                    'macro_top1':float(np.mean([g['top1'] for g in per_group]))})
    return metrics,per_group


def design(features,rows,variant,distance_mean,distance_sd):
    if variant.startswith('pair'):x=np.concatenate([features['enhancer'],features['promoter']],axis=1)
    elif variant.startswith('enhancer'):x=features['enhancer']
    else:x=features['promoter']
    if variant.endswith('_distance'):
        d=(np.log10([r['distance'] for r in rows])-distance_mean)/distance_sd
        x=np.column_stack([x,d])
    return np.asarray(x,dtype=np.float32)


def simple_design(rows,variant):
    d=np.log10([r['distance'] for r in rows])[:,None]
    if variant=='distance':return d
    if variant=='gc_distance':return np.column_stack([composition_matrix(rows,'gc','pair'),d])
    if variant=='kmer4_distance':return np.column_stack([composition_matrix(rows,'kmer4','pair'),d])
    raise ValueError(variant)


class VectorDataset:
    def __init__(self,x,rows,variant):self.x=np.asarray(x,dtype=np.float32);self.rows=rows;self.variant=variant
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return {'hidden':torch.from_numpy(self.x[i:i+1].copy()),'label':self.rows[i]['label'],'row_id':i}


class Head(nn.Module):
    def __init__(self,d,extra,dropout):
        super().__init__();self.operator=None;self.d=d
        w=min(range(1,1000),key=lambda w:abs(2*d+(d+extra+3)*w+2-237571))
        self.norm=nn.LayerNorm(d)
        self.head=nn.Sequential(nn.Linear(d+extra,w),nn.GELU(),nn.Dropout(dropout),nn.Linear(w,2))
    def forward(self,x):return self.head(torch.cat([self.norm(x[:,:self.d]),x[:,self.d:]],dim=1))


class Model(nn.Module):
    def __init__(self,d,extra,dropout):super().__init__();self.encoder=nn.Identity();self.classifier=Head(d,extra,dropout)
    def forward(self,batch):return self.classifier(batch['hidden'][:,0])


def make_model(cfg,variant,seed):
    assert variant in VARIANTS;seed_everything(seed)
    model=Model(1536 if variant.startswith('pair') else 768,int(variant.endswith('_distance')),cfg['dropout']).to(cfg['device'])
    seed_everything(seed);return model


def train_one_epoch(model,loader,optimizer,scaler,cfg):
    model.train();params=[p for p in model.parameters() if p.requires_grad];optimizer.zero_grad(set_to_none=True)
    total=0.;start=timed_start(cfg);weights=torch.tensor(cfg['class_weights'],device=cfg['device'])
    for step,batch in enumerate(loader):
        batch=move_batch(batch,cfg)
        n=accumulation_denominator(step,len(loader.dataset),cfg['batch_size'],cfg['gradient_accumulation'])
        with amp_context(cfg):
            logits=model(batch);loss_sum=F.cross_entropy(logits.float(),batch['labels'],weight=weights,reduction='sum');loss=loss_sum/n
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite weighted loss')
        scaler.scale(loss).backward();total+=float(loss_sum.detach())
        if (step+1)%cfg['gradient_accumulation']==0 or step+1==len(loader):
            scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
    return total/len(loader.dataset),elapsed(cfg,start)


@torch.no_grad()
def evaluate(model,dataset,collator,cfg):
    model.eval();start=timed_start(cfg)
    # Identical input vectors must have exactly identical scores, including across batch shapes.
    # Otherwise tiny GEMM rounding differences can create spurious within-promoter ranks.
    unique,index,inverse=np.unique(dataset.x,axis=0,return_index=True,return_inverse=True)
    unique_ds=VectorDataset(unique,[dataset.rows[i] for i in index],dataset.variant)
    scores=[]
    for batch in loader_for(unique_ds,collator,cfg):
        batch=move_batch(batch,cfg)
        with amp_context(cfg):logits=model(batch)
        scores.extend(logits.float().softmax(-1)[:,1].cpu().tolist())
    p=np.asarray(scores)[inverse]
    metrics,_=ranking_metrics(dataset.rows,p)
    if dataset.variant=='promoter':assert abs(metrics['macro_auroc']-.5)<1e-12
    return metrics,{'row_id':list(range(len(dataset))),'label':[r['label'] for r in dataset.rows],'probability':p.tolist()},elapsed(cfg,start)
