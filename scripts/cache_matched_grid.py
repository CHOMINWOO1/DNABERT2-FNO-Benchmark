"""Frozen DNABERT features regridded to 100 equal 10-bp bins per 1kb window."""
import gc
import json
from pathlib import Path
import numpy as np
import torch
from dnabert_fno.loading import load_backbone
from dnabert_fno.data import sha256_file
from dnabert_fno.runtime import amp_context,seed_everything,timed_start,elapsed,peak_memory,reset_peak,environment
from dnabert_fno import study
from matched_cv_models import grid_weights


def build(cfg,root,rows):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    seqs=sorted({r[k] for r in rows for k in ['enhancer','promoter']})
    identity={'sequence_hash':study.stable_hash(seqs),'model_id':cfg['model_id'],'revision':cfg['revision'],
              'precision':cfg['precision'],'environment':environment(cfg),'bin_bp':10,'batch_size':4,
              'code_sha256':{p:sha256_file(p) for p in ['scripts/cache_matched_grid.py','scripts/matched_cv_models.py']}}
    meta=root/'manifest.json';path=root/'grid.npy'
    if meta.exists():
        report=json.loads(meta.read_text());assert report['identity']==identity
        assert sha256_file(path)==report['sha256'];return seqs,report
    tokenizer,encoder,_=load_backbone(cfg,offline=True);before=study.frozen_hash(encoder);encoder.cuda().eval()
    assert all(not p.requires_grad for p in encoder.parameters())
    token=tokenizer(seqs,return_offsets_mapping=True,truncation=False)
    lengths=list(map(len,token['input_ids']));assert max(lengths)<=cfg['max_tokens']
    grid=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(seqs),100,768))
    reset_peak(cfg);start=timed_start(cfg)
    for first in range(0,len(seqs),4):
        last=min(first+4,len(seqs));b=last-first;l=max(lengths[first:last])
        ids=torch.full((b,l),tokenizer.pad_token_id,dtype=torch.long,device=cfg['device']);mask=torch.zeros_like(ids)
        weights=np.zeros((b,100,l),dtype=np.float32)
        for j,i in enumerate(range(first,last)):
            n=lengths[i];ids[j,:n]=torch.tensor(token['input_ids'][i],device=cfg['device']);mask[j,:n]=1
            offsets=token['offset_mapping'][i];spans=[(a,z) for a,z in offsets if z>a]
            assert spans[0][0]==0 and spans[-1][1]==1000 and all(z==a2 for (_,z),(a2,_) in zip(spans,spans[1:]))
            weights[j,:,:n]=grid_weights(offsets)
        with torch.no_grad(),amp_context(cfg):h=encoder(ids,mask)
        # Explicit FP32 regridding; no added half precision storage quantization.
        pooled=torch.bmm(torch.from_numpy(weights).to(cfg['device']),h.float())
        grid[first:last]=pooled.cpu().numpy()
        if first%400==0:print(f'GRID {last}/{len(seqs)}',flush=True)
    grid.flush();del grid
    after=study.frozen_hash(encoder);assert before==after
    report={'identity':identity,'shape':[len(seqs),100,768],'sha256':sha256_file(path),'encoder_hash_before':before,'encoder_hash_after':after,
            'seconds':elapsed(cfg,start),'peak_memory':peak_memory(cfg),'tokens_min':min(lengths),'tokens_max':max(lengths),'truncated':0}
    study.atomic_json(meta,report);del encoder,h,pooled;gc.collect();torch.cuda.empty_cache()
    return seqs,report


def gpu_load(path,device='cuda'):
    a=np.load(path,mmap_mode='r');t=torch.empty(a.shape,dtype=torch.float32,device=device)
    for first in range(0,len(a),128):t[first:first+128].copy_(torch.from_numpy(np.array(a[first:first+128],copy=True)))
    return t
