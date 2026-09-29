"""Paired stratified test-row bootstrap, defined before formal test evaluation."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root=Path('runs/long_epi_pilot')
    assert json.loads((root/'verification.json').read_text())['passed']
    plan=json.loads((root/'protocol.json').read_text())['plan']
    predicted={}; truth=None
    for n in plan['lengths_bp']:
        for variant in plan['variants']:
            values=[]
            for seed in plan['seeds']:
                with (root/f'bp{n}/results/{variant}_seed{seed}/test_predictions.csv').open(newline='') as f:
                    rows=list(csv.DictReader(f))
                y=np.array([int(r['label']) for r in rows])
                if truth is None: truth=y
                assert np.array_equal(y,truth)
                values.append(np.array([float(r['probability']) for r in rows])>=.5)
            predicted[(n,variant)]=np.stack(values)
    rng=np.random.default_rng(20260910); replicates=5000
    index=np.concatenate([rng.choice(np.flatnonzero(truth==label),size=(replicates,int((truth==label).sum())),replace=True)
                          for label in [0,1]],axis=1)
    labels=truth[index].astype(bool)[None,:,:]
    samples={}
    for key,values in predicted.items():
        p=values[:,index]
        tp=(p & labels).sum(-1);tn=(~p & ~labels).sum(-1)
        fp=(p & ~labels).sum(-1);fn=(~p & labels).sum(-1)
        denominator=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc=np.divide(tp*tn-fp*fn,denominator,out=np.zeros_like(denominator),where=denominator>0)
        samples[key]=mcc.mean(0)
    contrasts={}
    for n in plan['lengths_bp']:
        for control in ['cnn','lora']:
            contrasts[f'{n}bp_fno_minus_{control}']=samples[(n,'fno')]-samples[(n,control)]
    short,long=min(plan['lengths_bp']),max(plan['lengths_bp'])
    for variant in plan['variants']:
        contrasts[f'{variant}_{long}_minus_{short}']=samples[(long,variant)]-samples[(short,variant)]
    contrasts['change_in_fno_minus_lora_gap_5000_vs_1000']=(samples[(long,'fno')]-samples[(long,'lora')])-(samples[(short,'fno')]-samples[(short,'lora')])
    output={'replicates':replicates,'seed':20260910,'interval':'paired stratified percentile 95%',
            'unit':'test pair, resampled identically for all models, lengths and training seeds; average MCC across the three fixed trained seeds',
            'limitation':'Conditional on trained models and this balanced held-out cohort. Row resampling does not account for dependence between pairs sharing an enhancer/promoter. No retraining, chromosome-level uncertainty, or multiple-comparison correction. Descriptive pilot intervals.',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'contrasts':{name:{'lower_95':float(np.quantile(v,.025)),'upper_95':float(np.quantile(v,.975))}
                         for name,v in contrasts.items()}}
    (root/'bootstrap.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
