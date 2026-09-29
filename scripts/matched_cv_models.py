"""Shared 10-bp-grid residual mixers with parameter-matched MLP/CNN/FNO."""
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dnabert_fno.models import SpectralConv1d
from dnabert_fno.runtime import seed_everything,amp_context,move_batch,timed_start,elapsed
from dnabert_fno.train import loader_for
from crispr_ranking_models import ranking_metrics

CACHE=None


def grid_weights(offsets,bp=1000,bin_bp=10):
    a=np.asarray(offsets);starts=np.arange(0,bp,bin_bp)[:,None]
    overlap=np.maximum(0,np.minimum(starts+bin_bp,a[:,1])-np.maximum(starts,a[:,0]))
    w=overlap.astype(np.float32)/bin_bp
    if not np.allclose(w.sum(1),1,rtol=0,atol=1e-6):raise ValueError('Offsets do not cover every base exactly once')
    return w


def count_params(kind,w,modes=16,kernel=9):
    common=101634+2305
    return common+(1540 if kind=='fno' else 1541)*w+({'fno':2*modes+1,'cnn':kernel+1,'mlp':2}[kind])*w*w


def width_for(kind):return min(range(16,160),key=lambda w:abs(count_params(kind,w)-237571))


class Mixer(nn.Module):
    def __init__(self,kind,dropout,alpha):
        super().__init__();w=width_for(kind)
        self.norm=nn.LayerNorm(768);self.down=nn.Linear(768,w);self.up=nn.Linear(w,768)
        self.mix=SpectralConv1d(w,16) if kind=='fno' else nn.Conv1d(w,w,9 if kind=='cnn' else 1,padding=4 if kind=='cnn' else 0,padding_mode='circular')
        self.local=nn.Conv1d(w,w,1);self.blocknorm=nn.LayerNorm(w);self.dropout=nn.Dropout(dropout)
        self.alpha=nn.Parameter(torch.tensor(float(alpha)))
    def forward(self,h):
        z=self.down(self.norm(h)).transpose(1,2)
        z=self.mix(z)+self.local(z)
        update=self.up(self.dropout(F.gelu(self.blocknorm(z.transpose(1,2)))))
        return h.float()+self.alpha*update.float()


class Classifier(nn.Module):
    def __init__(self,kind,cfg):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(1537,64),nn.GELU(),nn.Dropout(cfg['dropout']),nn.Linear(64,2))
        self.norm=nn.LayerNorm(1536);self.operator=Mixer(kind,cfg['dropout'],.1)
    def forward(self,h,distance):
        b=h.shape[0];mixed=self.operator(h.reshape(b*2,100,768))
        pooled=mixed.mean(1).reshape(b,1536)
        return self.head(torch.cat([self.norm(pooled),distance[:,None]],dim=1))


class Model(nn.Module):
    def __init__(self,kind,cfg):
        super().__init__();self.encoder=nn.Identity();self.classifier=Classifier(kind,cfg)
    def forward(self,batch):
        if CACHE is None:raise RuntimeError('Feature cache is not installed')
        h=CACHE[batch['indices']]
        return self.classifier(h,batch['distance'])


def make_model(cfg,variant,seed):
    seed_everything(seed);m=Model(variant,cfg).to(cfg['device']);seed_everything(seed);return m


class Dataset:
    def __init__(self,rows,index,mean,sd,pairs):
        self.rows=rows;ids={r['global_id'] for r in rows}
        self.pairs=[p for p in pairs if p['positive'] in ids and p['negative'] in ids]
        self.indices=np.array([[index[r['enhancer']],index[r['promoter']]] for r in rows],dtype=np.int64)
        self.distance=((np.log10([r['distance'] for r in rows])-mean)/sd).astype(np.float32)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        return {'indices':torch.from_numpy(self.indices[i].copy()),'distance':torch.tensor(self.distance[i]),
                'labels':torch.tensor(self.rows[i]['label']),'row_ids':torch.tensor(i)}


def collate(items):return {k:torch.stack([x[k] for x in items]) for k in items[0]}


@torch.no_grad()
def evaluate(model,dataset,collator,cfg):
    model.eval();start=timed_start(cfg)
    keys=np.column_stack([dataset.indices,dataset.distance])
    _,positions,inverse=np.unique(keys,axis=0,return_index=True,return_inverse=True)
    subset=torch.utils.data.Subset(dataset,positions.tolist());scores=[]
    for batch in loader_for(subset,collator,cfg):
        batch=move_batch(batch,cfg)
        with amp_context(cfg):logits=model(batch)
        scores.extend(logits.float().softmax(-1)[:,1].cpu().tolist())
    p=np.asarray(scores)[inverse]
    metrics,_=ranking_metrics(dataset.rows,p)
    metrics['matched_concordance']=matched_concordance(dataset.pairs,{r['global_id']:p[i] for i,r in enumerate(dataset.rows)})[0]
    return metrics,{'row_id':list(range(len(dataset))),'label':[r['label'] for r in dataset.rows],'probability':p.tolist()},elapsed(cfg,start)


def matched_concordance(pairs,probabilities):
    grouped=defaultdict(list)
    for pair in pairs:
        a,b=probabilities[pair['positive']],probabilities[pair['negative']]
        grouped[pair['promoter_id']].append(float(a>b)+.5*float(a==b))
    groups=[{'promoter_id':key,'comparisons':len(v),'concordance':float(np.mean(v))} for key,v in sorted(grouped.items())]
    return float(np.mean([g['concordance'] for g in groups])),groups
