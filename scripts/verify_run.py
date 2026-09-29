"""Verify saved predictions with online encoder inference and stress 512 tokens.

Does not select hyperparameters or checkpoints. Writes engineering checks only.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dnabert_fno.data import TokenCollator, TokenDataset, read_splits, save_json, sha256_file
from dnabert_fno.loading import load_backbone
from dnabert_fno.models import RepresentationClassifier
from dnabert_fno.runtime import (amp_context, move_batch, peak_memory, reset_peak, seed_everything)
from dnabert_fno.train import evaluate, loader_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/promoter_seed42"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((args.run_dir / "config.json").read_text())
    seed_everything(cfg["seed"])
    torch.set_num_threads(4)
    tokenizer, encoder, _ = load_backbone(cfg, args.offline)
    encoder.to(cfg["device"])
    splits, _ = read_splits(cfg["data_dir"])
    dataset = TokenDataset(splits["test"], tokenizer, cfg["max_tokens"])
    collator = TokenCollator(tokenizer.pad_token_id)
    report = {"checkpoint_online_parity": {}}
    for variant in ["baseline", "fno"]:
        checkpoint = torch.load(args.run_dir / variant / "best.pt", map_location="cpu", weights_only=True)
        model = RepresentationClassifier(encoder.encoder.config.hidden_size, variant, cfg["width"], cfg["modes"],
                                         cfg["fno_layers"], cfg["dropout"], cfg["alpha_init"]).to(cfg["device"])
        model.load_state_dict(checkpoint["state_dict"])
        metrics, predictions, seconds = evaluate(model, encoder, loader_for(dataset, collator, cfg), cfg)
        with (args.run_dir / variant / "test_predictions.csv").open() as f:
            stored = list(csv.DictReader(f))
        assert predictions["row_id"] == [int(r["row_id"]) for r in stored]
        assert predictions["label"] == [int(r["label"]) for r in stored]
        reference = np.array([float(r["probability"]) for r in stored])
        difference = float(np.max(np.abs(np.array(predictions["probability"]) - reference)))
        np.testing.assert_allclose(predictions["probability"], reference, rtol=0, atol=1e-5)
        saved = json.loads((args.run_dir / variant / "result.json").read_text())
        for k, v in metrics.items():
            assert abs(v - saved["test"][k]) < 1e-10, (k, v, saved["test"][k])
        report["checkpoint_online_parity"][variant] = {
            "rows": len(stored), "max_abs_probability_difference": difference,
            "metrics_identical": True, "online_test_seconds": seconds,
            "checkpoint_sha256": sha256_file(args.run_dir / variant / "best.pt")}
        print(f"{variant}: online/cached parity verified on {len(stored)} test rows", flush=True)
        del model, checkpoint
    # Engineering stress uses synthetic DNA, never downstream performance evidence.
    batch_size = cfg["batch_size"]
    rng = np.random.default_rng(2026)
    sequences = ["".join(rng.choice(list("ACGT"), size=4000)) for _ in range(batch_size)]
    inputs = tokenizer(sequences, truncation=True, max_length=512, padding="max_length",
                       return_tensors="pt", return_special_tokens_mask=True)
    inputs = move_batch(inputs, cfg)
    mask = inputs["attention_mask"].bool() & ~inputs["special_tokens_mask"].bool()
    assert inputs["attention_mask"].sum(1).tolist() == [512] * batch_size
    model = RepresentationClassifier(768, "fno", cfg["width"], cfg["modes"], cfg["fno_layers"],
                                     cfg["dropout"], cfg["alpha_init"]).to(cfg["device"])
    encoder_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in encoder.parameters())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    reset_peak(cfg)
    with amp_context(cfg):
        h = encoder(inputs["input_ids"], inputs["attention_mask"])
        logits = model(h, mask)
        loss = torch.nn.functional.cross_entropy(logits.float(), torch.arange(batch_size, device=cfg["device"]) % 2)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    assert torch.isfinite(loss) and not h.requires_grad
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    after_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in encoder.parameters())).hexdigest()
    assert encoder_hash == after_hash
    report["synthetic_512_token_stress"] = {
        "batch_size": batch_size, "hidden_shape": list(h.shape), "precision": cfg["precision"],
        "optimizer_step_passed": True, "encoder_weights_sha256_unchanged": encoder_hash,
        "peak_memory": peak_memory(cfg), "is_biological_performance_evidence": False}
    source_files = sorted(Path("dnabert_fno").glob("*.py")) + sorted(Path("scripts").glob("*.py"))
    report["implementation_snapshot_sha256"] = {str(path): sha256_file(path) for path in source_files}
    save_json(args.run_dir / "verification.json", report)
    print(json.dumps(report["synthetic_512_token_stress"], indent=2))


if __name__ == "__main__":
    main()
