import pytest
import torch
from torch import nn

from dnabert_fno.models import (FrozenEncoder, RepresentationClassifier, ResidualFNO,
                               SpectralConv1d, masked_mean)


def test_spectral_retains_selected_frequency_and_rejects_high_frequency():
    n = 64
    t = torch.arange(n) / n
    low = torch.cos(2 * torch.pi * 2 * t)
    high = torch.cos(2 * torch.pi * 15 * t)
    layer = SpectralConv1d(1, 5)
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[..., 0] = 1
    output = layer((low + high)[None, None])[0, 0]
    torch.testing.assert_close(output, low, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("length", [1, 7, 32, 65])
def test_fft_short_and_odd_lengths_backward(length):
    layer = SpectralConv1d(4, 16)
    x = torch.randn(2, 4, length, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


def test_padding_values_lengths_and_batch_composition_do_not_change_prediction():
    torch.manual_seed(2)
    model = RepresentationClassifier(12, "fno", width=8, modes=16, dropout=0).eval()
    h = torch.randn(1, 9, 12)
    mask = torch.tensor([[False, True, True, True, True, True, True, True, False]])
    expected = model(h, mask)
    padded = torch.cat([h, torch.randn(1, 17, 12) * 100], dim=1)
    padded[:, [0, 8]] = -999
    padded_mask = torch.cat([mask, torch.zeros(1, 17, dtype=torch.bool)], dim=1)
    torch.testing.assert_close(model(padded, padded_mask), expected)
    batch = torch.cat([padded, torch.randn_like(padded)], dim=0)
    batch_mask = torch.cat([padded_mask, torch.ones_like(padded_mask)], dim=0)
    torch.testing.assert_close(model(batch, batch_mask)[:1], expected)


def test_zero_residual_is_exact_baseline_and_initial_head_matches():
    torch.manual_seed(42)
    baseline = RepresentationClassifier(12)
    torch.manual_seed(42)
    fno = RepresentationClassifier(12, "fno", width=8, alpha_init=0)
    for p, q in zip(baseline.head.parameters(), fno.head.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    x, mask = torch.randn(2, 13, 12), torch.ones(2, 13, dtype=torch.bool)
    torch.testing.assert_close(baseline.eval()(x, mask), fno.eval()(x, mask), rtol=0, atol=0)


def test_fno_parameters_receive_gradient_but_backbone_stays_frozen():
    class ToyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(8, 12)
            self.dropout = nn.Dropout(0.9)

        def forward(self, input_ids, **kwargs):
            return (self.dropout(self.emb(input_ids)),)

    encoder = FrozenEncoder(ToyEncoder())
    encoder.train()
    assert not encoder.training and not encoder.encoder.training
    ids = torch.randint(0, 8, (2, 11))
    mask = torch.ones_like(ids, dtype=torch.bool)
    h = encoder(ids, mask)
    torch.testing.assert_close(h, encoder(ids, mask), atol=0, rtol=0)
    model = RepresentationClassifier(12, "fno", width=8, modes=5, dropout=0)
    model(h, mask).square().mean().backward()
    assert not h.requires_grad
    assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.operator.blocks[0].spectral.weight.grad.abs().sum() > 0


def test_empty_pooling_rejected():
    with pytest.raises(ValueError):
        masked_mean(torch.zeros(2, 3, 4), torch.zeros(2, 3, dtype=torch.bool))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_amp_fft_non_power_of_two(dtype):
    model = RepresentationClassifier(24, "fno", width=8, modes=16).cuda()
    h = torch.randn(2, 79, 24, device="cuda")
    mask = torch.ones(2, 79, dtype=torch.bool, device="cuda")
    with torch.autocast("cuda", dtype=dtype):
        logits = model(h, mask)
        loss = logits.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
