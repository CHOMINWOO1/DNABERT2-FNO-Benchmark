import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from epi_information_models import region_weights,pooled_regions,composition_matrix,make_model,feature_view


def test_boundary_crossing_tokens_and_specials():
    w=region_weights([(0,0),(0,2),(2,5),(5,8),(0,0)],4)
    np.testing.assert_allclose(w,[[0,1,2/3,0,0],[0,0,1/3,1,0]],atol=1e-7)
    h=torch.tensor([[[999.],[2.],[5.],[8.],[-999.]]])
    got=pooled_regions(h,torch.from_numpy(w[None]))
    torch.testing.assert_close(got,torch.tensor([[[3.2],[7.25]]]))


def test_composition_does_not_cross_role_boundary():
    rows=[{'enhancer':'AAAA','promoter':'CCCC'}]
    x=composition_matrix(rows,'kmer4','pair')
    assert x.shape==(1,512) and np.count_nonzero(x)==2
    np.testing.assert_allclose(composition_matrix(rows,'gc','pair'),[[0,1]])


def test_matched_pair_heads_and_order():
    cfg={'device':'cpu','dropout':0.1}
    a=make_model(cfg,'joint_pair',42);b=make_model(cfg,'separate_pair',42)
    for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q,rtol=0,atol=0)
    for variant in ['enhancer','promoter','joint_global','separate_pair','joint_pair']:
        model=make_model(cfg,variant,42)
        assert abs(sum(p.numel() for p in model.parameters())-237571)/237571<0.01
    np.testing.assert_array_equal(feature_view({'enhancer':np.array([[1,2]]),'promoter':np.array([[3,4]])},'separate_pair'),[[1,2,3,4]])
