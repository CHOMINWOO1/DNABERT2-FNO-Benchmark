import json
from pathlib import Path
import pytest
import torch
from dnabert_fno.adaptation import make_classifier
from dnabert_fno.models import trainable_parameters
from scripts.diagnostic_models import DiagnosticClassifier,matched_width,count_formula

CFG=json.loads(Path('configs/comparative_study.json').read_text())

@pytest.mark.parametrize('variant',['cnn','fno'])
def test_uniform_attention_matches_original_mean_and_learns(variant):
    cfg={**CFG,'width':8,'cnn_width':8,'dropout':0,'diagnostic_kernel':3,'diagnostic_pooling':'attention'}
    torch.manual_seed(42); original=make_classifier(12,variant,cfg).eval()
    torch.manual_seed(42); attention=DiagnosticClassifier(12,variant,cfg).eval()
    x=torch.randn(2,15,12); mask=torch.ones(2,15,dtype=torch.bool);mask[0,9:]=False
    torch.testing.assert_close(original(x,mask),attention(x,mask),atol=1e-6,rtol=1e-5)
    attention(x,mask).square().sum().backward()
    assert attention.pool_score.weight.grad.abs().sum()>0
    with torch.no_grad(): attention.pool_score.weight.normal_()
    padded=torch.cat([x,torch.randn(2,8,12)*100],1)
    pm=torch.cat([mask,torch.zeros(2,8,dtype=torch.bool)],1)
    torch.testing.assert_close(attention(x,mask),attention(padded,pm),atol=1e-6,rtol=1e-5)

def test_parameter_budget_all_structures():
    for variant,values in [('fno',[8,16,64]),('cnn',[3,9,17])]:
        for value in values:
            w=matched_width(variant,value)
            cfg={**CFG,'width':w,'cnn_width':w,'modes':value if variant=='fno' else 16,
                 'diagnostic_kernel':value if variant=='cnn' else 3,'diagnostic_pooling':'mean'}
            count=trainable_parameters(DiagnosticClassifier(768,variant,cfg))
            assert count==count_formula(variant,w,value)
            assert abs(count-237571)/237571<.01
