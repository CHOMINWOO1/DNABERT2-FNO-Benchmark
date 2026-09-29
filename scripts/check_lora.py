"""Actual DNABERT-2 CUDA LoRA gradients, unchanged backbone, and runtime estimate."""
import json
import time
from pathlib import Path

import torch

from dnabert_fno.adaptation import LoRAEncoder, StudyModel, make_classifier, LoRALinear
from dnabert_fno.data import read_splits, TokenDataset, TokenCollator, save_json
from dnabert_fno.loading import load_backbone
from dnabert_fno.models import trainable_parameters
from dnabert_fno.runtime import seed_everything, amp_context, move_batch, peak_memory, reset_peak


def main():
    cfg = json.loads(Path("configs/comparative_study.json").read_text())
    seed_everything(42)
    torch.set_num_threads(4)
    tokenizer, frozen, _ = load_backbone(cfg, offline=True)
    raw = frozen.encoder
    encoder = LoRAEncoder(raw, cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"])
    model = StudyModel(encoder, make_classifier(768, "lora_fno", cfg)).cuda()
    rows, _ = read_splits(cfg["data_dir"])
    data = TokenDataset(rows["train"][:8], tokenizer, 512)
    b = move_batch(TokenCollator(tokenizer.pad_token_id)(data.items), cfg)
    base_before = [(p, p.detach().cpu().clone()) for n, p in model.encoder.named_parameters() if not p.requires_grad]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    timings = []
    reset_peak(cfg)
    model.train()
    for step in range(5):
        torch.cuda.synchronize(); start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with amp_context(cfg):
            loss = torch.nn.functional.cross_entropy(model(b).float(), b["labels"])
        loss.backward()
        names = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in names.values())
        assert all(p.grad is None for p, _ in base_before)
        if step > 0:
            assert all(p.grad.abs().sum() > 0 for n, p in names.items() if "lora_" in n)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(); timings.append(time.perf_counter()-start)
    assert all(torch.equal(p.detach().cpu(), before) for p, before in base_before)
    assert all(not m.training for m in raw.modules() if isinstance(m, torch.nn.Dropout) and m not in
               [layer.dropout for layer in raw.modules() if isinstance(layer, LoRALinear)])
    output = {"status": "passed", "lora_fno_trainable_params": trainable_parameters(model),
              "targets": encoder.target_names, "batch_size": 8, "token_length": b["input_ids"].shape[1],
              "step_seconds": timings, "peak_memory": peak_memory(cfg),
              "all_lora_layers_receive_gradients": True, "base_weights_unchanged": True}
    Path("runs/study_checks").mkdir(parents=True, exist_ok=True)
    save_json("runs/study_checks/lora_check.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
