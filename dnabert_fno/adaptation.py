"""Controls and fused-QKV LoRA for the pinned DNABERT-2 implementation."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import RepresentationClassifier

VARIANTS = ["baseline", "mlp", "no_spectral", "cnn", "fno", "lora", "lora_fno"]


class ZeroSpectral(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x, dtype=torch.float32)


def make_classifier(hidden_size, variant, cfg):
    if variant not in VARIANTS:
        raise ValueError(variant)
    kind = "baseline" if variant in {"baseline", "lora"} else "fno"
    width = {"mlp": cfg["mlp_width"], "cnn": cfg["cnn_width"]}.get(variant, cfg["width"])
    model = RepresentationClassifier(hidden_size, kind, width, cfg["modes"],
                                     cfg["fno_layers"], cfg["dropout"], cfg["alpha_init"])
    if variant in {"mlp", "no_spectral"}:
        for block in model.operator.blocks:
            block.spectral = ZeroSpectral()
    elif variant == "cnn":
        for block in model.operator.blocks:
            block.spectral = nn.Conv1d(width, width, kernel_size=3, padding=1, padding_mode="circular")
    return model


class LoRALinear(nn.Module):
    """W x + (alpha/r) B A x; base weights remain frozen.

    A single rank-r update spans the fused QKV matrix. This updates Q, K and V;
    it is not three separate rank-r updates, nor a Q/V-only PEFT configuration.
    """
    def __init__(self, base, rank=6, alpha=12.0, dropout=0.1):
        super().__init__()
        if rank < 1 or not isinstance(base, nn.Linear):
            raise ValueError("LoRA requires a Linear layer and positive rank")
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):
        result = self.base(x)
        if self.merged:
            return result
        return result + F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scale

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight.add_((self.lora_b.float() @ self.lora_a.float()) * self.scale)
            self.merged = True


class LoRAEncoder(nn.Module):
    def __init__(self, raw_encoder, rank=6, alpha=12, dropout=0.1):
        super().__init__()
        self.encoder = raw_encoder.requires_grad_(False)
        self.target_names = []
        targets = [(name, layer) for name, layer in self.encoder.named_modules()
                   if name.endswith("attention.self.Wqkv") and isinstance(layer, nn.Linear)]
        if len(targets) != self.encoder.config.num_hidden_layers:
            raise ValueError(f"Expected one fused QKV target per layer, found {len(targets)}")
        for name, base in targets:
            parent, leaf = name.rsplit(".", 1)
            setattr(self.encoder.get_submodule(parent), leaf, LoRALinear(base, rank, alpha, dropout))
            self.target_names.append(name)
        self.train(False)

    def train(self, mode=True):
        # Keep pretrained dropout disabled, matching frozen features. Only the
        # explicitly configured LoRA dropout changes mode. eval does not disable autograd.
        self.training = mode
        self.encoder.eval()
        for module in self.encoder.modules():
            if isinstance(module, LoRALinear):
                if mode and module.merged:
                    raise RuntimeError("Cannot train a merged inference model")
                module.dropout.train(mode)
        return self

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=torch.zeros_like(input_ids))[0]

    @torch.no_grad()
    def merge(self):
        self.eval()
        for module in self.encoder.modules():
            if isinstance(module, LoRALinear):
                module.merge()


class StudyModel(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, batch):
        hidden = batch.get("hidden")
        if hidden is None:
            hidden = self.encoder(batch["input_ids"], batch["attention_mask"])
        elif isinstance(self.encoder, LoRAEncoder):
            raise ValueError("LoRA cannot train or evaluate from a frozen feature cache")
        return self.classifier(hidden, batch["content_mask"])


def parameter_snapshot(model, trainable=True):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad == trainable}


def load_trainable(model, state):
    targets = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if targets.keys() != state.keys():
        raise ValueError("Checkpoint trainable parameter names do not match architecture")
    with torch.no_grad():
        for name, p in targets.items():
            p.copy_(state[name])
