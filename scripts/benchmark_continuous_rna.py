"""Small end-to-end inference benchmark on four prespecified dev sequences."""
import gc
import json
import warnings
from pathlib import Path
import numpy as np
import torch
from dnabert_fno.loading import load_backbone
from dnabert_fno.runtime import seed_everything,amp_context,timed_start,elapsed,reset_peak,peak_memory
from dnabert_fno.study import atomic_json,frozen_hash
from dnabert_fno.data import sha256_file
from matched_cv_models import grid_weights
import continuous_rna_models as models


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


@torch.no_grad()
def inference(sequences,bp,model,tokenizer,encoder,cfg):
    start=(65536-bp)//2;chunks=[seq[start+j:start+j+1024] for seq in sequences for j in range(0,bp,1024)];features=[]
    for first in range(0,len(chunks),16):
        token=tokenizer(chunks[first:first+16],return_offsets_mapping=True,truncation=False,padding=True,return_tensors='pt')
        offsets=token.pop('offset_mapping').numpy();weights=np.stack([grid_weights(o,bp=1024,bin_bp=256) for o in offsets])
        ids=token['input_ids'].to(cfg['device']);mask=token['attention_mask'].to(cfg['device'])
        with amp_context(cfg):hidden=encoder(ids,mask)
        features.append(torch.bmm(torch.from_numpy(weights).to(cfg['device']),hidden.float()))
    x=torch.cat(features).reshape(len(sequences),bp//256,768)
    with amp_context(cfg):p=model(x)
    assert torch.isfinite(p).all();return p.cpu().numpy()


def main():
    torch.set_num_threads(4);seed_everything(42);root=Path('runs/continuous_rna');training=read(root/'training_complete.json');cfg=read(root/'protocol.json')['config']
    assert read(root/'status.json')['phase']=='complete'
    warnings.filterwarnings('ignore',message='.*clean_up_tokenization_spaces.*');warnings.filterwarnings('ignore',message='Unable to import Triton.*');warnings.filterwarnings('ignore',message='Increasing alibi size.*')
    data=Path(cfg['data_dir']);rows=read(data/'rows.json');ids=[r['global_id'] for r in rows if r['split']=='dev'][:4]
    with (data/'sequences.fasta').open() as f:allseq=[line.strip() for line in f if not line.startswith('>')]
    sequences=[allseq[i] for i in ids];prior=read('runs/matched_cv/protocol.json')['base_config']
    tokenizer,encoder,_=load_backbone(prior,offline=True);encoder.to(cfg['device']).eval();before=frozen_hash(encoder);results=[]
    for r in training['final']:
        s=r['settings']
        if s['seed']!=42:continue
        model=models.make_model(s['model'],cfg,42);model.load_state_dict(torch.load(Path(r['path'])/'best.pt',map_location=cfg['device'],weights_only=True)['state_dict']);model.eval()
        inference(sequences,s['bp'],model,tokenizer,encoder,cfg);reset_peak(cfg);times=[]
        for _ in range(5):
            start=timed_start(cfg);inference(sequences,s['bp'],model,tokenizer,encoder,cfg);times.append(elapsed(cfg,start))
        median=float(np.median(times));results.append({'model':s['model'],'bp':s['bp'],'checkpoint_sha256':r['checkpoint_sha256'],'seconds':times,
            'median_seconds':median,'min_seconds':min(times),'max_seconds':max(times),'genes_per_second':4/median,'bp_per_second':4*s['bp']/median,'peak_memory':peak_memory(cfg)})
        print(f'ONLINE {s["model"]} {s["bp"]}: {median:.4f}s / 4 genes',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    assert len(results)==9 and frozen_hash(encoder)==before
    report={'results':results,'dev_global_ids':ids,'repeats':5,'warmup_repeats':1,'scope':'RAM DNA through tokenizer, encoder, bin pooling, global mixer, and CPU output. No disk/model loading or resident feature cache.',
            'source_sha256':sha256_file(data/'source.json'),'encoder_hash':before,'timing_protocol_sha256':sha256_file('docs/CONTINUOUS_RNA_TIMING_PROTOCOL.md'),
            'benchmark_script_sha256':sha256_file(__file__)}
    atomic_json(root/'online_timing.json',report)


if __name__=='__main__':main()
