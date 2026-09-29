"""Independently encode contiguous 1kb chunks, then pool each into four 256bp bins."""
import gc
import json
import warnings
from pathlib import Path
import numpy as np
import torch
from dnabert_fno.loading import load_backbone
from dnabert_fno.study import atomic_json,stable_hash,frozen_hash
from dnabert_fno.data import sha256_file
from dnabert_fno.runtime import seed_everything,amp_context,environment,timed_start,elapsed,peak_memory,reset_peak
from matched_cv_models import grid_weights


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def build():
    cfg=read('configs/continuous_rna.json');root=Path(cfg['output_dir'])/'features';root.mkdir(parents=True,exist_ok=True)
    data=Path(cfg['data_dir']);source=read(data/'source.json')
    for name,digest in source['files_sha256'].items():assert sha256_file(data/name)==digest
    prior=read('runs/matched_cv/protocol.json')['base_config'];encoder_cfg={**prior,**{k:cfg[k] for k in ['device','precision']},'max_tokens':1026}
    identity={'source_sha256':sha256_file(data/'source.json'),'model_id':prior['model_id'],'revision':prior['revision'],
              'encoder_batch_size':cfg['encoder_batch_size'],'chunk_bp':1024,'bin_bp':256,'environment':environment(cfg),
              'code_sha256':{p:sha256_file(p) for p in ['scripts/cache_continuous_rna.py','scripts/matched_cv_models.py']}}
    path=root/'grid.npy'
    if (root/'manifest.json').exists():
        old=read(root/'manifest.json');assert old['identity']==identity and sha256_file(path)==old['sha256'];return old
    with (data/'sequences.fasta').open() as f:sequences=[line.strip() for line in f if not line.startswith('>')]
    assert all(len(s)==65536 for s in sequences)
    warnings.filterwarnings('ignore',message='.*clean_up_tokenization_spaces.*');warnings.filterwarnings('ignore',message='Unable to import Triton.*');warnings.filterwarnings('ignore',message='Increasing alibi size.*')
    seed_everything(42);torch.set_num_threads(4);tokenizer,encoder,_=load_backbone(encoder_cfg,offline=True);encoder.to(cfg['device']).eval()
    before=frozen_hash(encoder);assert all(not p.requires_grad for p in encoder.parameters())
    shape=(len(sequences),256,768);array=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=shape)
    flat=array.reshape(-1,4,768);total=len(sequences)*64;lengths=[];reset_peak(cfg);started=timed_start(cfg)
    for first in range(0,total,cfg['encoder_batch_size']):
        last=min(total,first+cfg['encoder_batch_size']);chunks=[sequences[i//64][(i%64)*1024:(i%64+1)*1024] for i in range(first,last)]
        token=tokenizer(chunks,return_offsets_mapping=True,truncation=False,padding=True,return_tensors='pt')
        offsets=token.pop('offset_mapping').numpy();ids=token['input_ids'].to(cfg['device']);mask=token['attention_mask'].to(cfg['device'])
        lengths.extend(mask.sum(1).cpu().tolist());assert ids.shape[1]<=1026
        weights=np.stack([grid_weights(o,bp=1024,bin_bp=256) for o in offsets])
        with torch.no_grad(),amp_context(cfg):hidden=encoder(ids,mask)
        pooled=torch.bmm(torch.from_numpy(weights).to(cfg['device']),hidden.float())
        flat[first:last]=pooled.cpu().numpy()
        if first%2048==0 or last==total:
            array.flush();atomic_json(root/'progress.json',{'chunks_completed':last,'total_chunks':total})
            print(f'ENCODE {last}/{total}',flush=True)
    array.flush();del flat,array
    after=frozen_hash(encoder);assert before==after
    result={'identity':identity,'shape':list(shape),'sha256':sha256_file(path),'encoder_hash_before':before,'encoder_hash_after':after,
            'seconds':elapsed(cfg,started),'peak_memory':peak_memory(cfg),'min_tokens':min(lengths),'max_tokens':max(lengths),'truncated_chunks':0}
    atomic_json(root/'manifest.json',result);del encoder,hidden,pooled;gc.collect();torch.cuda.empty_cache();return result


if __name__=='__main__':print(json.dumps(build(),indent=2))
