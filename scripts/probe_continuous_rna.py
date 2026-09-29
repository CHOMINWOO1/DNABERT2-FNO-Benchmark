"""Capacity check only, with no real expression labels or performance selection."""
import gc
import json
from pathlib import Path
import torch
from dnabert_fno.runtime import amp_context,reset_peak,peak_memory
import continuous_rna_models as m


def main():
    torch.set_num_threads(4);cfg=json.loads(Path('configs/continuous_rna.json').read_text());rows=json.loads(Path('data/continuous_rna/rows.json').read_text())
    root=Path(cfg['output_dir']);root.mkdir(parents=True,exist_ok=True)
    cache=torch.zeros(len(rows),256,768,device=cfg['device']);results=[]
    for kind in cfg['variants']:
        model=m.make_model(kind,cfg,42);optimizer=torch.optim.AdamW(model.parameters(),lr=.0003);reset_peak(cfg)
        for _ in range(2):
            x=torch.randn(cfg['batch_size'],256,768,device=cfg['device'])
            with amp_context(cfg):loss=model(x).square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad()
        results.append({'model':kind,'width':m.width_for(kind),'params':sum(p.numel() for p in model.parameters()),'peak_memory':peak_memory(cfg),'finite':bool(torch.isfinite(loss))})
        del model,optimizer,x,loss;gc.collect();torch.cuda.empty_cache()
    report={'capacity_only':True,'cache_bytes':cache.numel()*cache.element_size(),'batch_size':cfg['batch_size'],'length_bp':65536,'results':results}
    (root/'capacity_probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
