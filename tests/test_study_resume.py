"""Interrupted optimizer/RNG continuation must reproduce uninterrupted training."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from dnabert_fno import study
from dnabert_fno.adaptation import LoRALinear, StudyModel
from dnabert_fno.data import TokenCollator
from dnabert_fno.models import RepresentationClassifier
from dnabert_fno.runtime import seed_everything


def test_interrupted_lora_trial_restores_optimizer_and_dropout_rng(tmp_path, monkeypatch):
    class ToyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10, 4).requires_grad_(False)
            self.update = LoRALinear(nn.Linear(4, 4), rank=2, alpha=4, dropout=0.2)

        def forward(self, input_ids, attention_mask):
            return self.update(self.embedding(input_ids))

    def factory(cfg, variant, seed):
        seed_everything(seed)
        return StudyModel(ToyEncoder(), RepresentationClassifier(4, dropout=0.2))

    monkeypatch.setattr(study, "make_model", factory)
    cfg = json.loads(Path("configs/comparative_study.json").read_text())
    cfg.update(device="cpu", precision="fp32", epochs=3, patience=10, batch_size=3,
               gradient_accumulation=2, length_bucketing=False)
    generator = torch.Generator().manual_seed(72)
    rows = [{"input_ids": torch.randint(0, 10, (7,), generator=generator),
             "content_mask": torch.ones(7, dtype=torch.bool), "label": i % 2, "row_id": i} for i in range(14)]
    datasets = {"train": rows, "dev": rows[:8]}
    whole = tmp_path / "whole"; whole.mkdir()
    interrupted = tmp_path / "interrupted"; interrupted.mkdir()
    complete = study.fit_trial(whole, "test_protocol", cfg, "lora", 42, 0.001, datasets, TokenCollator(0))
    original_write = study.atomic_json

    def interrupt_after_checkpoint(path, value):
        original_write(path, value)
        if Path(path) == interrupted / "status.json" and value.get("epoch") == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(study, "atomic_json", interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        study.fit_trial(interrupted, "test_protocol", cfg, "lora", 42, 0.001, datasets, TokenCollator(0))
    monkeypatch.setattr(study, "atomic_json", original_write)
    resumed = study.fit_trial(interrupted, "test_protocol", cfg, "lora", 42, 0.001, datasets, TokenCollator(0))
    a = torch.load(Path(complete["path"]) / "last.pt", weights_only=True)
    b = torch.load(Path(resumed["path"]) / "last.pt", weights_only=True)
    for key in a["trainable"]:
        torch.testing.assert_close(a["trainable"][key], b["trainable"][key], atol=0, rtol=0)
    assert [r["train_loss"] for r in a["history"]] == [r["train_loss"] for r in b["history"]]
    assert complete["best_dev_mcc"] == resumed["best_dev_mcc"]
