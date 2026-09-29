import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from residual_rank_models import make_model,pair_loss,pair_weights,choose_gamma
from matched_cv_models import count_params,width_for


def test_pair_objective_shift_invariance_and_gradient_direction():
    x=torch.tensor([[.2,.3],[-1.,2.]],requires_grad=True)
    loss=pair_loss(x,'pairwise');torch.testing.assert_close(loss,pair_loss(x+10,'pairwise'))
    loss.sum().backward();assert (x.grad[:,0]<0).all() and (x.grad[:,1]>0).all()
    expected=(torch.nn.functional.binary_cross_entropy_with_logits(x[:,0],torch.ones(2),reduction='none')+torch.nn.functional.binary_cross_entropy_with_logits(x[:,1],torch.zeros(2),reduction='none'))/2
    torch.testing.assert_close(pair_loss(x,'pointwise'),expected)


def test_promoter_weights_and_gamma_baseline_fallback():
    ps=[{'promoter_id':x} for x in ['a','b','b','b']];w=pair_weights(ps)
    np.testing.assert_allclose(w[0],w[1:].sum())
    pairs=[{'promoter_id':'a','positive':0,'negative':1}]
    chosen,_=choose_gamma(np.array([1.,0.]),np.array([-5.,5.]),pairs,[0,1],[0.,.25,.5,1.])
    assert chosen['gamma']==0 and chosen['dev_concordance']==1


def test_zero_initial_correction_and_gradient_reaches_mixers():
    cfg={'device':'cpu','dropout':0.};h=torch.randn(2,2,100,768)
    for kind in ['mlp','cnn','fno']:
        model=make_model(kind,cfg,42);assert sum(p.numel() for p in model.parameters())==count_params(kind,width_for(kind))-129
        assert torch.equal(model.features(h),torch.zeros(2))
        opt=torch.optim.SGD(model.parameters(),lr=.01)
        model.features(h).sum().backward();opt.step();opt.zero_grad()
        model.features(h).sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.operator.mix.weight.grad.abs().sum()>0
