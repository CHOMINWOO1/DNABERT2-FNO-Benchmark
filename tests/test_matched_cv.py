import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from matched_cv_models import grid_weights,make_model,count_params,width_for,matched_concordance
from prepare_matched_cv import matched_pairs


def test_bp_grid_cross_boundary_and_special_tokens():
    w=grid_weights([(0,0),(0,7),(7,13),(13,20),(0,0)],bp=20,bin_bp=10)
    np.testing.assert_allclose(w,[[0,.7,.3,0,0],[0,0,.3,.7,0]],atol=1e-7)
    assert np.allclose(w.sum(1),1)


def test_distance_pairs_do_not_cross_promoters_and_ties():
    rows=[{'promoter_id':g,'gene':g,'global_id':i,'label':y,'fold':0,'distance':d} for i,(g,y,d) in enumerate([('a',1,100),('a',0,125),('a',0,126),('b',0,100)])]
    pairs=matched_pairs(rows,1.25);assert len(pairs)==1 and pairs[0]['negative']==1
    assert matched_concordance(pairs,[.5,.5,.1,.1])[0]==.5
    assert matched_concordance(pairs,[.8,.2,.1,.1])[0]==1


def test_matched_mixer_counts_head_initialization_and_fft_gradient():
    cfg={'dropout':0.,'device':'cpu'};models=[make_model(cfg,k,42) for k in ['mlp','cnn','fno']]
    for kind,model in zip(['mlp','cnn','fno'],models):
        n=sum(p.numel() for p in model.parameters());assert n==count_params(kind,width_for(kind)) and abs(n-237571)/237571<.01
        h=torch.randn(2,2,100,768);out=model.classifier(h,torch.tensor([0.,1.]));assert out.shape==(2,2)
        out.sum().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for a,b in zip(models[0].classifier.head.parameters(),models[2].classifier.head.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
