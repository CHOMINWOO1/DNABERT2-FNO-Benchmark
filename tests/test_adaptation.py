import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dnabert_fno.adaptation import (LoRALinear, LoRAEncoder, StudyModel, ZeroSpectral,
                                    make_classifier, load_trainable, parameter_snapshot)
from dnabert_fno.models import RepresentationClassifier, trainable_parameters
from scripts.prepare_clean_split import canonical, deduplicate


CFG = json.loads(Path("configs/comparative_study.json").read_text())


@pytest.mark.parametrize("variant", ["mlp", "no_spectral", "cnn"])
def test_controls_padding_and_batch_invariance(variant):
    model = make_classifier(12, variant, {**CFG, "width": 8, "mlp_width": 9, "cnn_width": 7, "dropout": 0}).eval()
    x = torch.randn(1, 9, 12)
    mask = torch.tensor([[False, True, True, True, True, True, True, True, False]])
    y = model(x, mask)
    padded = torch.cat([x, torch.randn(1, 10, 12) * 100], 1)
    padded_mask = torch.cat([mask, torch.zeros(1, 10, dtype=torch.bool)], 1)
    batch = torch.cat([padded, torch.randn_like(padded)], 0)
    masks = torch.cat([padded_mask, torch.ones_like(padded_mask)], 0)
    torch.testing.assert_close(model(batch, masks)[:1], y)
    model(batch, masks).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_mlp_is_tokenwise_and_no_spectral_preserves_other_weights():
    torch.manual_seed(42)
    fno = make_classifier(12, "fno", {**CFG, "width": 8, "dropout": 0})
    torch.manual_seed(42)
    no = make_classifier(12, "no_spectral", {**CFG, "width": 8, "dropout": 0})
    with torch.no_grad():
        fno.operator.blocks[0].spectral.weight.zero_()
    x = torch.randn(2, 9, 12)
    mask = torch.ones(2, 9, dtype=torch.bool)
    fno.eval(); no.eval()
    torch.testing.assert_close(fno(x, mask), no(x, mask))
    permutation = torch.randperm(9)
    torch.testing.assert_close(no(x[:, permutation], mask), no(x, mask))


def test_parameter_budget_controls():
    counts = {v: trainable_parameters(make_classifier(768, v, CFG))
              for v in ["baseline", "mlp", "no_spectral", "cnn", "fno"]}
    assert counts["fno"] == 237571
    assert counts["baseline"] == 1538
    assert counts["no_spectral"] == counts["fno"] - 64 * 64 * 16 * 2
    for variant in ["mlp", "cnn"]:
        assert abs(counts[variant] / counts["fno"] - 1) < 0.01


def test_lora_zero_init_gradients_merge_and_checkpoint():
    torch.manual_seed(7)
    base = nn.Linear(8, 24)
    layer = LoRALinear(base, rank=3, alpha=6, dropout=0)
    x = torch.randn(2, 5, 8, requires_grad=True)
    torch.testing.assert_close(layer(x), base(x), atol=0, rtol=0)
    before = base.weight.detach().clone()
    optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
    layer(x).square().mean().backward()
    assert layer.lora_b.grad.abs().sum() > 0
    assert layer.base.weight.grad is None and x.grad.abs().sum() > 0
    optimizer.step(); optimizer.zero_grad()
    layer(x).square().mean().backward()
    assert layer.lora_a.grad.abs().sum() > 0
    assert torch.equal(base.weight, before)
    state = parameter_snapshot(layer)
    other = LoRALinear(nn.Linear(8, 24), rank=3, alpha=6, dropout=0)
    other.base.load_state_dict(base.state_dict())
    load_trainable(other, state)
    torch.testing.assert_close(other(x), layer(x))
    expected = layer.eval()(x)
    layer.merge()
    torch.testing.assert_close(layer(x), expected, atol=3e-7, rtol=3e-6)
    merged_weight = base.weight.clone()
    layer.merge()
    assert torch.equal(merged_weight, base.weight)


def test_dedup_conflicts_and_reverse_complements():
    assert canonical("AAAAC") == canonical("GTTTT")
    rows = {"train": [{"row_id": 0, "sequence": "AAAAC", "label": 1},
                      {"row_id": 1, "sequence": "CCCCA", "label": 1}],
            "dev": [{"row_id": 0, "sequence": "GTTTT", "label": 1},
                    {"row_id": 1, "sequence": "CCCCA", "label": 0}]}
    accepted, conflicts = deduplicate(rows)
    assert len(accepted) == 1 and len(accepted[0]["members"]) == 2
    assert len(conflicts) == 1
