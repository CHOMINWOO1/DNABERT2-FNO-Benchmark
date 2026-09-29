"""Global mixing over a true contiguous 256-bp feature lattice."""
import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dnabert_fno.models import SpectralConv1d
from dnabert_fno.runtime import seed_everything,amp_context

CACHE=None
MASK_MEAN=None


def count_params(kind,w):
    return 15770+(868 if kind=='attention' else 859)*w+(24 if kind=='attention' else 63)*w*w


def width_for(kind):
    candidates=range(16,161,4 if kind=='attention' else 1)
    return min(candidates,key=lambda w:abs(count_params(kind,w)-300000))


def position_encoding(n,w,device):
    # Positions retain their physical offsets from the central TSS across lengths.
    positions=torch.arange(n,device=device,dtype=torch.float32)-(n-1)/2
    frequencies=torch.exp(torch.arange(0,w,2,device=device,dtype=torch.float32)*(-math.log(10000.)/w))
    angles=positions[:,None]*frequencies[None,:];p=torch.zeros(n,w,device=device)
    p[:,0::2]=torch.sin(angles);p[:,1::2]=torch.cos(angles[:,:w//2]);return p


class Block(nn.Module):
    def __init__(self,kind,w,index,cfg):
        super().__init__();self.kind=kind;self.norm1=nn.LayerNorm(w);self.norm2=nn.LayerNorm(w)
        if kind=='fno':self.mix=SpectralConv1d(w,8);self.local=nn.Conv1d(w,w,1)
        elif kind=='cnn':
            dilation=cfg['cnn_dilations'][index];self.mix=nn.Conv1d(w,w,17,padding=8*dilation,dilation=dilation)
        else:self.mix=nn.MultiheadAttention(w,4,dropout=cfg['dropout'],batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(w,2*w),nn.GELU(),nn.Dropout(cfg['dropout']),nn.Linear(2*w,w))
        self.dropout=nn.Dropout(cfg['dropout'])
    def forward(self,x):
        z=self.norm1(x)
        if self.kind=='attention':z=self.mix(z,z,z,need_weights=False)[0]
        else:
            z=z.transpose(1,2)
            z=(self.mix(z)+self.local(z) if self.kind=='fno' else self.mix(z)).transpose(1,2)
        x=x.float()+self.dropout(z.float());return x+self.dropout(self.ffn(self.norm2(x)).float())


class Model(nn.Module):
    def __init__(self,kind,cfg):
        super().__init__();w=width_for(kind)
        # Common head has identical initialization for every model and length at a seed.
        self.head=nn.Linear(64,218);self.input_norm=nn.LayerNorm(768);self.down=nn.Linear(768,w)
        self.blocks=nn.Sequential(*[Block(kind,w,i,cfg) for i in range(3)])
        self.output_norm=nn.LayerNorm(w);self.readout=nn.Linear(w,64)
    def forward(self,x):
        z=self.down(self.input_norm(x));z=z.float()+position_encoding(z.shape[1],z.shape[2],z.device)[None]
        z=self.output_norm(self.blocks(z));mid=z.shape[1]//2
        # Read only the central 1kb; global models must move distal information here.
        return self.head(F.gelu(self.readout(z[:,mid-2:mid+2].mean(1)))).float()


def make_model(kind,cfg,seed):
    seed_everything(seed);model=Model(kind,cfg).to(cfg['device']);seed_everything(seed);return model


def head_hash(model):
    h=hashlib.sha256()
    for name,p in model.head.named_parameters():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def crop(x,bp):
    length=bp//256;start=(x.shape[1]-length)//2;return x[:,start:start+length]


def perturb(x,kind,permutation=None,donor=None):
    """Preserve central 4kb. Shuffle whole 1kb groups, retaining local encoder context."""
    if kind=='intact':return x
    assert x.shape[1]==256;z=x.clone();outer=torch.cat([torch.arange(120,device=x.device),torch.arange(136,256,device=x.device)])
    if kind=='mask':z[:,outer]=MASK_MEAN
    elif kind=='shuffle':
        chunks=x.reshape(len(x),64,4,768)
        z=chunks[:,permutation].reshape_as(x)
    elif kind=='swap':z[:,outer]=donor[:,outer]
    else:raise ValueError(kind)
    return z


def shuffle_indices(seed):
    outer=np.r_[np.arange(30),np.arange(34,64)];p=np.arange(64)
    p[outer]=np.random.default_rng(seed).permutation(outer);return p


def donor_indices(n,seed):
    assert n>1;rng=np.random.default_rng(seed);order=rng.permutation(n);dest=np.empty(n,dtype=int)
    dest[order]=np.roll(order,1);assert np.all(dest!=np.arange(n));return dest


def pearson_tracks(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    a=y-y.mean(0);b=p-p.mean(0);den=np.sqrt((a*a).sum(0)*(b*b).sum(0))
    return np.divide((a*b).sum(0),den,out=np.zeros(y.shape[1]),where=den>1e-15)


def metrics(y,p):
    values=pearson_tracks(y,p)
    assert np.isfinite(y).all() and np.isfinite(p).all()
    return {'macro_pearson':float(values.mean()),'mse':float(np.mean((np.asarray(y)-p)**2)),'per_track_pearson':values.tolist(),
            'constant_target_tracks':int((np.asarray(y).std(0)<1e-15).sum()),'constant_prediction_tracks':int((np.asarray(p).std(0)<1e-15).sum())}


@torch.no_grad()
def predict(model,ids,bp,cfg,kind='intact',seed=None):
    model.eval();scores=[];permutation=None;donors=None
    if kind=='shuffle':permutation=torch.as_tensor(shuffle_indices(seed),device=cfg['device'])
    if kind=='swap':donors=np.asarray(ids)[donor_indices(len(ids),seed)]
    for first in range(0,len(ids),cfg['batch_size']):
        selected=ids[first:first+cfg['batch_size']];x=CACHE[selected]
        donor=CACHE[donors[first:first+cfg['batch_size']]] if donors is not None else None
        x=perturb(x,kind,permutation,donor);x=crop(x,bp)
        with amp_context(cfg):p=model(x)
        scores.append(p.cpu().numpy())
    return np.concatenate(scores).astype(np.float64)
