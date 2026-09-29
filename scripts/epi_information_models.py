"""Matched pooled-feature heads and composition features, no backbone fine-tuning."""
import itertools
import numpy as np
import torch
from torch import nn
from dnabert_fno.runtime import seed_everything

NEURAL=['enhancer','promoter','separate_pair','joint_pair','joint_global']


def region_weights(offsets,boundary):
    """Split a boundary-crossing BPE token fractionally; special tokens get zero."""
    a=np.asarray(offsets,dtype=np.int64); start,end=a[:,0],a[:,1]
    length=end-start
    enh=np.divide(np.maximum(0,np.minimum(end,boundary)-start),length,
                  out=np.zeros(len(a),dtype=np.float32),where=length>0)
    prom=(length>0).astype(np.float32)-enh
    assert (enh>=0).all() and (prom>=0).all()
    return np.stack([enh,prom],axis=0)


def pooled_regions(hidden,weights):
    weights=weights.to(device=hidden.device,dtype=torch.float32)
    assert bool((weights.sum(-1)>0).all())
    return torch.einsum('brl,bld->brd',weights,hidden.float())/weights.sum(-1,keepdim=True)


def width_for(d,target=237571):
    # LayerNorm(d), Linear(d,w), GELU, Dropout, Linear(w,2).
    return min(range(1,1000),key=lambda w:abs(2*d+(d+3)*w+2-target))


class PooledHead(nn.Module):
    def __init__(self,d,dropout):
        super().__init__(); self.operator=None
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,width_for(d)),nn.GELU(),nn.Dropout(dropout),nn.Linear(width_for(d),2))
    def forward(self,x): return self.head(x)


class PooledModel(nn.Module):
    def __init__(self,d,dropout):
        super().__init__()
        # Encoder is absent during head fitting; actual encoder hash is audited during extraction.
        self.encoder=nn.Identity();self.classifier=PooledHead(d,dropout)
    def forward(self,batch): return self.classifier(batch['hidden'][:,0])


def make_model(cfg,variant,seed):
    assert variant in NEURAL
    seed_everything(seed)
    d=1536 if variant in ['separate_pair','joint_pair'] else 768
    model=PooledModel(d,cfg['dropout']).to(cfg['device'])
    seed_everything(seed)
    return model


class VectorDataset:
    def __init__(self,x,y): self.x=np.asarray(x,dtype=np.float32);self.y=np.asarray(y,dtype=np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        return {'hidden':torch.from_numpy(self.x[i:i+1].copy()),'label':int(self.y[i]),'row_id':i}


def feature_view(features,variant):
    if variant=='separate_pair': return np.concatenate([features['enhancer'],features['promoter']],axis=1)
    return features[variant]


WORDS=[''.join(x) for x in itertools.product('ACGT',repeat=4)]
INDEX={w:i for i,w in enumerate(WORDS)}


def composition(sequence,kind):
    if kind=='gc': return np.array([(sequence.count('G')+sequence.count('C'))/len(sequence)],dtype=np.float64)
    assert kind=='kmer4'
    counts=np.zeros(256,dtype=np.float64)
    for i in range(len(sequence)-3):
        word=sequence[i:i+4]
        if word in INDEX: counts[INDEX[word]]+=1
    return counts/max(counts.sum(),1)


def composition_matrix(rows,kind,role):
    roles=['enhancer','promoter'] if role=='pair' else [role]
    # Count roles separately: no artificial k-mer crossing enhancer/promoter junction.
    return np.stack([np.concatenate([composition(r[k],kind) for k in roles]) for r in rows])
