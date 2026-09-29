"""Synthetic DNA capacity probe only: no biological performance measurement."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from dnabert_fno.adaptation import LoRAEncoder, StudyModel, make_classifier
from dnabert_fno.loading import load_backbone
from dnabert_fno.data import TokenDataset, TokenCollator
from dnabert_fno.runtime import seed_everything, amp_context, move_batch, peak_memory, reset_peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bp', type=int, required=True)
    p.add_argument('--variant', choices=['fno', 'lora', 'lora_fno'], required=True)
    p.add_argument('--batch', type=int, default=1)
    args = p.parse_args()
    cfg = json.loads(Path('configs/comparative_study.json').read_text())
    seed_everything(42); torch.set_num_threads(4)
    tokenizer, frozen, _ = load_backbone(cfg, offline=True)
    encoder = (LoRAEncoder(frozen.encoder, cfg['lora_rank'], cfg['lora_alpha'], cfg['lora_dropout'])
               if args.variant.startswith('lora') else frozen)
    model = StudyModel(encoder, make_classifier(768, args.variant, cfg)).cuda().train()
    rng = np.random.default_rng(20260908)
    rows = [{'sequence': ''.join(rng.choice(list('ACGT'), args.bp)), 'label': i % 2, 'row_id': i}
            for i in range(args.batch)]
    dataset = TokenDataset(rows, tokenizer, args.bp + 2)
    batch = move_batch(TokenCollator(tokenizer.pad_token_id)(dataset.items), cfg)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    report = {'scope': 'synthetic random DNA capacity test, not biological accuracy',
              'bp': args.bp, 'variant': args.variant, 'batch': args.batch,
              'tokens': list(batch['input_ids'].shape), 'precision': 'bf16', 'step_seconds': []}
    reset_peak(cfg)
    try:
        for step in range(3):
            torch.cuda.synchronize(); start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with amp_context(cfg):
                loss = torch.nn.functional.cross_entropy(model(batch).float(), batch['labels'])
            assert torch.isfinite(loss)
            loss.backward()
            trainable = [p for p in model.parameters() if p.requires_grad]
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            torch.nn.utils.clip_grad_norm_(trainable, 1, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(); report['step_seconds'].append(time.perf_counter()-start)
        report['status'] = 'passed'
    except torch.cuda.OutOfMemoryError:
        report['status'] = 'cuda_oom'
    report['peak_memory'] = peak_memory(cfg)
    path = Path('runs/long_context_probe'); path.mkdir(parents=True, exist_ok=True)
    (path / f'{args.variant}_{args.bp}bp_batch{args.batch}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
