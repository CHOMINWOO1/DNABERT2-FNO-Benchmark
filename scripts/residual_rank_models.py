"""Sequence-only zero-initialized corrections to a frozen distance logit."""
from collections import Counter
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matched_cv_models as grid
from dnabert_fno.runtime import seed_everything,amp_context


class Correction(nn.Module):
    def __init__(self,kind,cfg):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(1536,64),nn.GELU(),nn.Dropout(cfg['dropout']),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.norm=nn.LayerNorm(1536);self.operator=grid.Mixer(kind,cfg['dropout'],.1)
    def features(self,h):
        n=h.shape[0];z=self.operator(h.reshape(n*2,100,768))
        return self.head(self.norm(z.mean(1).reshape(n,1536))).squeeze(-1).float()
    def forward(self,indices):
        if grid.CACHE is None:raise RuntimeError('No frozen grid cache')
        return self.features(grid.CACHE[indices])


def make_model(kind,cfg,seed):
    seed_everything(seed);model=Correction(kind,cfg).to(cfg['device']);seed_everything(seed);return model


def parameter_hash(model,head_only=False):
    h=hashlib.sha256()
    for name,p in (model.head if head_only else model).named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def pair_weights(pairs):
    counts=Counter(p['promoter_id'] for p in pairs)
    return np.array([len(pairs)/(len(counts)*counts[p['promoter_id']]) for p in pairs],dtype=np.float32)


def pair_loss(scores,objective):
    if objective=='pairwise':return F.softplus(-(scores[:,0]-scores[:,1]))
    if objective=='pointwise':return .5*(F.softplus(-scores[:,0])+F.softplus(scores[:,1]))
    raise ValueError(objective)


@torch.no_grad()
def predict(model,indices,cfg,batch_size=32):
    model.eval();unique,inverse=np.unique(np.asarray(indices),axis=0,return_inverse=True);scores=[]
    for first in range(0,len(unique),batch_size):
        batch=torch.as_tensor(unique[first:first+batch_size],device=cfg['device'])
        with amp_context(cfg):p=model(batch)
        scores.extend(p.cpu().tolist())
    return np.asarray(scores,dtype=np.float64)[inverse]


def choose_gamma(base,correction,pairs,global_ids,gammas):
    values=[]
    for gamma in gammas:
        score=base+gamma*correction
        metric=grid.matched_concordance(pairs,dict(zip(global_ids,score)))[0]
        values.append({'gamma':gamma,'dev_concordance':metric})
    return sorted(values,key=lambda x:(-x['dev_concordance'],x['gamma']))[0],values
