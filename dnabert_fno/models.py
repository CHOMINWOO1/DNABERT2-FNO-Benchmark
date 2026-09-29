"""Small residual FNO on the valid BPE-token lattice.

FFT is always float32/complex64, including under AMP. Each sequence is
transformed at its own valid length, so padding and batch mates cannot
change its Fourier frequencies. Special tokens are excluded from mixing.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        if width < 1 or modes < 1:
            raise ValueError("width and modes must be positive")
        self.modes = modes
        # Two real parameters avoid complex-gradient clipping/AMP edge cases.
        self.weight = nn.Parameter(torch.randn(width, width, modes, 2) / width)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft(x.float(), dim=-1, norm="ortho")
            n_modes = min(self.modes, spectrum.shape[-1])
            weights = torch.view_as_complex(self.weight.float().contiguous())
            out = spectrum.new_zeros(x.shape[0], self.weight.shape[1], spectrum.shape[-1])
            out[..., :n_modes] = torch.einsum(
                "bim,iom->bom", spectrum[..., :n_modes], weights[..., :n_modes]
            )
            return torch.fft.irfft(out, n=x.shape[-1], dim=-1, norm="ortho")


class OperatorBlock(nn.Module):
    def __init__(self, width, modes, dropout):
        super().__init__()
        self.spectral = SpectralConv1d(width, modes)
        self.local = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.spectral(x) + self.local(x)
        return self.dropout(F.gelu(self.norm(z.transpose(1, 2)))).transpose(1, 2)


class ResidualFNO(nn.Module):
    def __init__(self, hidden_size, width=64, modes=16, layers=1, dropout=0.1, alpha_init=0.1):
        super().__init__()
        if layers < 1:
            raise ValueError("FNO must contain at least one layer")
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, width)
        self.blocks = nn.Sequential(*[OperatorBlock(width, modes, dropout) for _ in range(layers)])
        self.up = nn.Linear(width, hidden_size)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, hidden, content_mask):
        mask = content_mask.bool()
        lengths = mask.sum(1)
        if (lengths == 0).any():
            raise ValueError("Cannot mix a sequence with no DNA tokens")
        delta = torch.zeros_like(hidden, dtype=torch.float32)
        # Group equal lengths, pack valid positions, and scatter updates back.
        # This keeps input CLS/SEP/PAD representations untouched.
        for length in lengths.unique().tolist():
            rows = (lengths == length).nonzero(as_tuple=True)[0]
            selected = mask[rows].nonzero(as_tuple=True)[1].reshape(len(rows), length)
            packed = hidden[rows[:, None], selected].float()
            z = self.down(self.norm(packed)).transpose(1, 2)
            update = self.up(self.blocks(z).transpose(1, 2)).float()
            delta[rows[:, None], selected] = update
        return hidden.float() + self.alpha * delta


def masked_mean(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    if (mask.sum(1) == 0).any():
        raise ValueError("Cannot pool an empty DNA sequence")
    return (hidden * weights).sum(1) / weights.sum(1)


class RepresentationClassifier(nn.Module):
    def __init__(self, hidden_size, variant="baseline", width=64, modes=16,
                 layers=1, dropout=0.1, alpha_init=0.1):
        super().__init__()
        if variant not in {"baseline", "fno"}:
            raise ValueError(f"Unknown variant: {variant}")
        # Initialize head first, making its seed identical between variants.
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 2))
        self.operator = (ResidualFNO(hidden_size, width, modes, layers, dropout, alpha_init)
                         if variant == "fno" else None)

    def forward(self, hidden, content_mask):
        hidden = hidden.float()
        if self.operator is not None:
            hidden = self.operator(hidden, content_mask)
        return self.head(masked_mean(hidden, content_mask))


class FrozenEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        self.encoder.eval()
        # Explicit types bypass an upstream CPU LongTensor assertion on CUDA.
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=torch.zeros_like(input_ids))
        return output[0].detach()


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
