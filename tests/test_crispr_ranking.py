import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from crispr_ranking_models import ranking_metrics,design,make_model,VectorDataset,evaluate,train_one_epoch
from dnabert_fno.data import collate_features
from prepare_crispr_ranking import query


def toy_rows():
    return [{'promoter_id':g,'gene':g,'chrom':'chr1','label':y,'distance':d} for g,y,d in [('a',0,1000),('a',1,2000),('b',1,4000),('b',0,8000),('b',0,16000),('c',0,20000)]]


def test_gene_conditioned_scores_and_tie_aware_top_hit():
    rows=toy_rows();m,g=ranking_metrics(rows,[.2,.2,.9,.9,.9,.5])
    assert m['macro_auroc']==.5 and len(g)==2
    assert np.isclose(m['macro_top1'],(1/2+1/3)/2)
    m,_=ranking_metrics(rows,[.1,.9,.9,.2,.1,.5]);assert m['macro_auroc']==1 and m['macro_top1']==1


def test_coordinate_conversion_and_distance_feature():
    assert query('chr1',99,1099)=='1:100..1099:1'
    rows=toy_rows();f={'enhancer':np.ones((6,768)),'promoter':np.zeros((6,768))}
    x=design(f,rows,'pair_distance',3,1)
    assert x.shape==(6,1537);np.testing.assert_allclose(x[:,-1],np.log10([r['distance'] for r in rows])-3,atol=1e-7)


def test_promoter_scores_identical_across_batches():
    cfg={'device':'cpu','dropout':.1,'batch_size':2,'num_workers':0,'precision':'fp32','seed':42,'length_bucketing':False}
    rows=toy_rows();x=np.stack([np.full(768,ord(r['gene']),dtype=np.float32) for r in rows])
    model=make_model(cfg,'promoter',42)
    m,p,_=evaluate(model,VectorDataset(x,rows,'promoter'),collate_features,cfg)
    assert m['macro_auroc']==.5
    assert p['probability'][2]==p['probability'][3]==p['probability'][4]
    for name in ['promoter','enhancer','pair','enhancer_distance','pair_distance']:
        assert abs(sum(p.numel() for p in make_model(cfg,name,42).parameters())-237571)/237571<.01


def test_weighted_loss_uses_sample_denominator():
    from torch.utils.data import DataLoader
    cfg={'device':'cpu','dropout':0.,'batch_size':2,'gradient_accumulation':2,'precision':'fp32','class_weights':[.75,1.5]}
    rows=toy_rows();x=np.random.default_rng(7).normal(size=(6,768)).astype(np.float32)
    model=make_model(cfg,'enhancer',42)
    with torch.no_grad():
        logits=model({'hidden':torch.from_numpy(x[:,None,:])})
        expected=torch.nn.functional.cross_entropy(logits,torch.tensor([r['label'] for r in rows]),weight=torch.tensor(cfg['class_weights']),reduction='sum').item()/6
    loader=DataLoader(VectorDataset(x,rows,'enhancer'),batch_size=2,collate_fn=collate_features)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0)
    loss,_=train_one_epoch(model,loader,optimizer,torch.amp.GradScaler('cuda',enabled=False),cfg)
    assert abs(loss-expected)<1e-6
