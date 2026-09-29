"""Budget-matched token mixers and a masked, initially uniform attention pool."""
import torch
from torch import nn
from dnabert_fno.adaptation import make_classifier, StudyModel
from dnabert_fno.loading import load_backbone
from dnabert_fno.models import masked_mean
from dnabert_fno.runtime import seed_everything


def count_formula(kind, width, structure):
    return 3843 + (1540 if kind == 'fno' else 1541)*width + (2*structure+1 if kind=='fno' else structure+1)*width**2


def matched_width(kind, structure, target=237571):
    return min(range(16,161),key=lambda w:abs(count_formula(kind,w,structure)-target))


class DiagnosticClassifier(nn.Module):
    def __init__(self, hidden_size, variant, cfg):
        super().__init__()
        base=make_classifier(hidden_size,variant,cfg)
        self.head=base.head; self.operator=base.operator
        if variant=='cnn' and cfg['diagnostic_kernel']!=3:
            w=cfg['cnn_width']; k=cfg['diagnostic_kernel']
            for block in self.operator.blocks:
                block.spectral=nn.Conv1d(w,w,k,padding=k//2,padding_mode='circular')
        self.pooling=cfg['diagnostic_pooling']
        if self.pooling not in ['mean','attention']:
            raise ValueError(self.pooling)
        self.pool_score=nn.Linear(hidden_size,1,bias=False) if self.pooling=='attention' else None
        if self.pool_score is not None:
            nn.init.zeros_(self.pool_score.weight)

    def forward(self, hidden, content_mask):
        h=self.operator(hidden.float(),content_mask)
        if self.pool_score is None:
            pooled=masked_mean(h,content_mask)
        else:
            if not content_mask.any(1).all():
                raise ValueError('Empty sequence')
            scores=self.pool_score(h).squeeze(-1).float().masked_fill(~content_mask.bool(),-torch.inf)
            weights=scores.softmax(-1)
            pooled=(weights.unsqueeze(-1)*h).sum(1)
        return self.head(pooled)


def make_model(cfg,variant,seed):
    seed_everything(seed)
    _,encoder,_=load_backbone(cfg,offline=True)
    seed_everything(seed)
    classifier=DiagnosticClassifier(encoder.encoder.config.hidden_size,variant,cfg).to(cfg['device'])
    model=StudyModel(encoder,classifier)
    seed_everything(seed)
    return model
