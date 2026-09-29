import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import continuous_rna_models as m


def test_global_model_budgets_and_distal_gradient():
    cfg={'device':'cpu','dropout':0.,'cnn_dilations':[1,4,16]}
    heads=[]
    for kind in ['cnn','fno','attention']:
        model=m.make_model(kind,cfg,42);n=sum(p.numel() for p in model.parameters())
        assert n==m.count_params(kind,m.width_for(kind)) and abs(n-300000)/300000<.01
        x=torch.randn(1,256,768,requires_grad=True);out=model(x);assert out.shape==(1,218)
        out.sum().backward();assert torch.isfinite(x.grad).all() and x.grad[:,0:4].abs().sum()>0
        heads.append(m.head_hash(model))
    assert len(set(heads))==1


def test_physical_crop_and_positions_align():
    x=torch.arange(256)[None,:,None].expand(1,256,768).float()
    torch.testing.assert_close(m.crop(x,4096),x[:,120:136])
    small=m.position_encoding(16,61,'cpu');large=m.position_encoding(256,61,'cpu')
    torch.testing.assert_close(small,large[120:136])


def test_distal_perturbations_preserve_central_4kb():
    x=torch.randn(2,256,768);m.MASK_MEAN=torch.zeros(768)
    for kind in ['mask','shuffle','swap']:
        p=torch.as_tensor(m.shuffle_indices(42));z=m.perturb(x,kind,p,x.flip(0))
        torch.testing.assert_close(z[:,120:136],x[:,120:136],rtol=0,atol=0)
        assert not torch.equal(z[:,:120],x[:,:120])
    for seed in [42,43,44]:assert np.all(m.donor_indices(20,seed)!=np.arange(20))


def test_track_pearson_affine_invariance_and_constants():
    y=np.arange(20).reshape(10,2).astype(float)
    np.testing.assert_allclose(m.pearson_tracks(y,2*y+5),1)
    np.testing.assert_allclose(m.pearson_tracks(y,-y),-1)
    np.testing.assert_array_equal(m.pearson_tracks(y,np.zeros_like(y)),0)


def test_cluster_bootstrap_equals_explicit_gene_resampling():
    from report_continuous_rna import bootstrap_correlations
    rng=np.random.default_rng(11);y=rng.normal(size=(7,3));p=.3*y+rng.normal(size=(7,3))
    groups=[np.array(x) for x in [[0,1],[2],[3,4],[5,6]]]
    weights=np.array([[1,2,0,1],[0,1,1,2]],dtype=float)
    actual=bootstrap_correlations(y,p,groups,weights)
    expected=[]
    for row in weights:
        ids=np.concatenate([group for group,count in zip(groups,row.astype(int)) for _ in range(count)])
        expected.append(m.pearson_tracks(y[ids],p[ids]).mean())
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
