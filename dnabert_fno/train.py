"""Paired experiment: dev selects checkpoint; test is evaluated only afterwards."""
import argparse
import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .cache import cache_features
from .data import TokenCollator, TokenDataset, collate_features, read_splits, save_json
from .loading import load_backbone
from .metrics import binary_metrics
from .models import RepresentationClassifier, trainable_parameters
from .runtime import (amp_context, elapsed, environment, move_batch, peak_memory,
                      reset_peak, seed_everything, timed_start)


def validate_config(cfg):
    for key in ["batch_size", "gradient_accumulation", "epochs", "patience", "width", "modes", "fno_layers", "inference_repeats", "inference_warmup"]:
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 3 <= cfg["max_tokens"] <= 512:
        raise ValueError("Stage-1 max_tokens must be between 3 and 512")
    if cfg["learning_rate"] <= 0 or cfg["weight_decay"] < 0 or not 0 <= cfg["dropout"] < 1:
        raise ValueError("Invalid optimization settings")
    if cfg["selection_metric"] not in {"mcc", "f1", "accuracy", "roc_auc", "pr_auc"}:
        raise ValueError("Unknown checkpoint selection metric")
    if cfg["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    if cfg["device"].startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Install a CUDA 12.8+ PyTorch build for RTX 5060 Ti")
        if cfg["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 is unsupported on this GPU; select fp16")
    elif cfg["precision"] == "fp16":
        raise ValueError("CPU mode requires fp32 or bf16")


def loader_for(dataset, collator, cfg, epoch=None):
    # Dedicated generator makes sample order independent of model RNG usage.
    generator = torch.Generator().manual_seed(cfg["seed"] + (epoch or 0))
    if epoch is not None and cfg.get("length_bucketing", False):
        # Full batches of equal length reduce small variable-length FFT launches.
        # Remainders are pooled; every sample appears once, and only the final
        # batch may be short. The same permutation is used for both models.
        if hasattr(dataset, "offsets"):
            lengths = np.diff(dataset.offsets)
        else:
            lengths = [int(x["content_mask"].sum()) for x in dataset.items]
        batches = length_bucket_batches(lengths, cfg["batch_size"], generator)
        return DataLoader(dataset, batch_sampler=batches, num_workers=cfg["num_workers"],
                          generator=generator, collate_fn=collator,
                          pin_memory=cfg["device"].startswith("cuda"))
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=epoch is not None,
                      generator=generator, num_workers=cfg["num_workers"], collate_fn=collator,
                      pin_memory=cfg["device"].startswith("cuda"), drop_last=False)


def length_bucket_batches(lengths, batch_size, generator):
    buckets = {}
    for i, length in enumerate(lengths):
        buckets.setdefault(int(length), []).append(i)
    batches, tails = [], []
    for indices in buckets.values():
        indices = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
        full = len(indices) // batch_size * batch_size
        batches.extend(indices[i:i+batch_size] for i in range(0, full, batch_size))
        tails.extend(indices[full:])
    batches = [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]
    tails = [tails[i] for i in torch.randperm(len(tails), generator=generator).tolist()]
    tail_batches = [tails[i:i+batch_size] for i in range(0, len(tails), batch_size)]
    # Shuffle full remainder batches among regular batches; keep last short last.
    short = tail_batches.pop() if tail_batches and len(tail_batches[-1]) < batch_size else None
    batches += tail_batches
    batches = [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]
    if short:
        batches.append(short)
    return batches


def get_hidden(batch, encoder):
    if "hidden" in batch:
        return batch["hidden"]
    return encoder(batch["input_ids"], batch["attention_mask"])


def accumulation_denominator(step, n_samples, batch_size, accumulation):
    group_start = (step // accumulation) * accumulation * batch_size
    return min(accumulation * batch_size, n_samples - group_start)


def train_epoch(model, encoder, loader, optimizer, scaler, cfg):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = timed_start(cfg)
    loss_sum = 0.0
    for step, batch in enumerate(loader):
        batch = move_batch(batch, cfg)
        denominator = accumulation_denominator(step, len(loader.dataset), cfg["batch_size"], cfg["gradient_accumulation"])
        with amp_context(cfg):
            logits = model(get_hidden(batch, encoder), batch["content_mask"])
            summed_loss = F.cross_entropy(logits.float(), batch["labels"], reduction="sum")
            loss = summed_loss / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss. Retry a new paired run with learning_rate=3e-4")
        scaler.scale(loss).backward()
        loss_sum += summed_loss.detach().item()
        if (step + 1) % cfg["gradient_accumulation"] == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    return loss_sum / len(loader.dataset), elapsed(cfg, start)


@torch.no_grad()
def evaluate(model, encoder, loader, cfg):
    model.eval()
    labels, probs, ids = [], [], []
    start = timed_start(cfg)
    for batch in loader:
        batch = move_batch(batch, cfg)
        with amp_context(cfg):
            logits = model(get_hidden(batch, encoder), batch["content_mask"])
        probs.extend(logits.float().softmax(-1)[:, 1].cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
        ids.extend(batch["row_ids"].cpu().tolist())
    seconds = elapsed(cfg, start)
    return binary_metrics(labels, probs), {"row_id": ids, "label": labels, "probability": probs}, seconds


@torch.no_grad()
def benchmark(model, encoder, tokens, token_collator, cfg):
    """Full encoder+head latency over up to 20 distinct test batches; includes H2D.

    Excludes tokenizer and disk IO. Warmup excluded. Synchronize every batch.
    """
    model.eval()
    loader = loader_for(tokens, token_collator, cfg)
    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)
        if i + 1 >= cfg["inference_repeats"]:
            break
    for _ in range(cfg["inference_warmup"]):
        b = move_batch(batches[0], cfg)
        with amp_context(cfg):
            model(encoder(b["input_ids"], b["attention_mask"]), b["content_mask"])
    del b
    reset_peak(cfg)
    durations = []
    samples = 0
    for batch in batches:
        start = timed_start(cfg)
        b = move_batch(batch, cfg)
        with amp_context(cfg):
            model(encoder(b["input_ids"], b["attention_mask"]), b["content_mask"])
        durations.append(elapsed(cfg, start))
        samples += len(batch["labels"])
    return {"scope": "encoder_plus_head_including_H2D_excluding_tokenization_and_disk",
            "batches": len(durations), "samples": samples, "seconds": sum(durations),
            "mean_batch_ms": float(np.mean(durations) * 1000),
            "median_batch_ms": float(np.median(durations) * 1000),
            "p95_batch_ms": float(np.percentile(durations, 95) * 1000),
            "ms_per_sample": sum(durations) * 1000 / samples,
            "samples_per_second": samples / sum(durations), "peak_memory": peak_memory(cfg)}


def save_predictions(path, predictions):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(predictions.keys())
        writer.writerows(zip(*predictions.values()))


def run_variant(variant, cfg, encoder, datasets, collator, tokens, token_collator, audit, cache_report):
    seed_everything(cfg["seed"])
    model = RepresentationClassifier(encoder.encoder.config.hidden_size, variant,
                                     cfg["width"], cfg["modes"], cfg["fno_layers"], cfg["dropout"], cfg["alpha_init"])
    initial_head_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.head.parameters())).hexdigest()
    model.to(cfg["device"])
    # Reset after construction: common starting RNG state for stochastic training.
    seed_everything(cfg["seed"])
    output = Path(cfg["output_dir"]) / variant
    output.mkdir()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["device"].startswith("cuda") and cfg["precision"] == "fp16")
    if cfg["cache_features"]:
        encoder.to("cpu")
    else:
        encoder.to(cfg["device"])
    best_score, best_epoch, bad_epochs = -math.inf, 0, 0
    history = []
    reset_peak(cfg)
    training_start = timed_start(cfg)
    for epoch in range(1, cfg["epochs"] + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loader = loader_for(datasets["train"], collator, cfg, epoch=epoch)
        loss, train_seconds = train_epoch(model, encoder, train_loader, optimizer, scaler, cfg)
        metrics, _, dev_seconds = evaluate(model, encoder, loader_for(datasets["dev"], collator, cfg), cfg)
        record = {"model": variant, "seed": cfg["seed"], "epoch": epoch, "learning_rate": lr,
                  "train_loss": loss, "train_seconds": train_seconds, "dev_seconds": dev_seconds,
                  "train_samples_per_second": len(datasets["train"]) / train_seconds,
                  "alpha": float(model.operator.alpha.detach()) if model.operator else None,
                  "dev": metrics}
        history.append(record)
        with (output / "epochs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        score = metrics[cfg["selection_metric"]]
        if score is None:
            raise ValueError("Validation selection metric is undefined")
        if score > best_score + cfg["min_delta"]:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg, "variant": variant,
                        "epoch": epoch, "dev": metrics, "data_audit": audit}, output / "best.pt")
        else:
            bad_epochs += 1
        scheduler.step()
        print(f"{variant} epoch {epoch:02d}: loss={loss:.4f} dev MCC={metrics['mcc']:.4f} "
              f"train={train_seconds:.1f}s best={best_epoch}", flush=True)
        if bad_epochs >= cfg["patience"]:
            break
    training_seconds = elapsed(cfg, training_start)
    train_memory = peak_memory(cfg)
    best = torch.load(output / "best.pt", map_location=cfg["device"], weights_only=True)
    model.load_state_dict(best["state_dict"])
    test_metrics, predictions, test_seconds = evaluate(model, encoder, loader_for(datasets["test"], collator, cfg), cfg)
    save_predictions(output / "test_predictions.csv", predictions)
    excluded = set(audit["overlap"]["train_test"]["overlapping_row_ids_in_second_split"])
    excluded.update(audit["overlap"]["dev_test"]["overlapping_row_ids_in_second_split"])
    clean_indices = [i for i, row in enumerate(predictions["row_id"]) if row not in excluded]
    sensitivity = {"excluded_test_rows": len(predictions["label"]) - len(clean_indices),
                   "remaining_rows": len(clean_indices),
                   "definition": "Exclude test sequences identical or reverse-complement-identical to train/dev",
                   "metrics": binary_metrics([predictions["label"][i] for i in clean_indices],
                                             [predictions["probability"][i] for i in clean_indices]) if clean_indices else None}
    # Remove optimizer tensors before timing inference (same policy for both models).
    del optimizer, scheduler, scaler, best
    gc.collect()
    encoder.to(cfg["device"])
    inference = benchmark(model, encoder, tokens["test"], token_collator, cfg)
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    result = {"model": variant, "config": cfg, "trainable_params": trainable_parameters(model),
              "encoder_params": sum(p.numel() for p in encoder.parameters()),
              "initial_head_sha256": initial_head_hash, "encoder_frozen_verified": True,
              "best_epoch": best_epoch, "best_dev_score": best_score, "epochs_completed": len(history),
              "test": test_metrics, "test_without_train_dev_overlap": sensitivity, "test_seconds": test_seconds,
              "test_scope": "cached_representation_plus_head" if cfg["cache_features"] else "encoder_plus_head",
              "training": {"mode": "cached_features" if cfg["cache_features"] else "online_frozen_encoder",
                           "wall_seconds_including_dev_and_checkpoints": training_seconds,
                           "train_only_seconds": sum(r["train_seconds"] for r in history),
                           "mean_epoch_seconds": float(np.mean([r["train_seconds"] for r in history])),
                           "peak_memory": train_memory}, "inference": inference,
              "cache_generation_seconds": sum(r["generation_seconds"] for r in cache_report.values()),
              "alpha": float(model.operator.alpha.detach()) if model.operator else None}
    save_json(output / "result.json", result)
    del model
    gc.collect()
    return result


def write_comparison(results, output):
    metric_names = ["accuracy", "f1", "mcc", "roc_auc", "pr_auc", "average_precision"]
    rows = []
    for r in results:
        row = {"model": r["model"], "seed": r["config"]["seed"], "trainable_params": r["trainable_params"],
               "max_tokens": r["config"]["max_tokens"], **r["test"],
               "train_peak_mib": r["training"]["peak_memory"]["allocated_mib"],
               "inference_peak_mib": r["inference"]["peak_memory"]["allocated_mib"],
               "train_seconds": r["training"]["train_only_seconds"],
               "epoch_seconds": r["training"]["mean_epoch_seconds"],
               "inference_ms_per_sample": r["inference"]["ms_per_sample"],
               "samples_per_second": r["inference"]["samples_per_second"], "best_epoch": r["best_epoch"]}
        rows.append(row)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    a, b = results
    assert a["initial_head_sha256"] == b["initial_head_sha256"]
    delta = {k: b["test"][k] - a["test"][k] for k in metric_names}
    summary = {"test_delta_fno_minus_baseline": delta,
               "technical_minimum_mcc_within_0_01": delta["mcc"] >= -0.01,
               "all_five_metrics_improved": all(delta[k] > 0 for k in metric_names[:5]),
               "interpretation": "Single-seed short-promoter PoC; no statistical significance or long-context claim.",
               "results": results}
    save_json(output / "comparison.json", summary)
    header = ["Model", "Params", "Accuracy", "F1", "MCC", "ROC-AUC", "PR-AUC", "Train peak MiB", "Inference peak MiB", "Train s", "Inference samples/s"]
    lines = ["# DNABERT-2 frozen PoC", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        values = [row["model"], str(row["trainable_params"])]
        values += [f"{row[k]:.4f}" for k in metric_names[:5]]
        values += [f"{row[k]:.2f}" if row[k] is not None else "N/A" for k in ["train_peak_mib", "inference_peak_mib", "train_seconds", "samples_per_second"]]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "MCC delta (FNO - baseline): " + f"{delta['mcc']:+.4f}", "",
              "PR-AUC uses trapezoidal integration; average precision is logged separately.",
              "Training uses cached frozen features when configured; cache generation is shared and logged separately.",
              "Inference includes frozen encoder + classifier, H2D and synchronization; excludes tokenization and disk IO.",
              "GPU memory is PyTorch process allocated memory, not whole-device VRAM including Windows desktop.",
              "512 is a token cap, not 512 bp. Actual lengths and truncation counts are in data_audit.json.",
              "Single seed and one short-sequence task do not establish significance or long-range benefit."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/promoter.json")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit-per-split", type=int, help="Stratified engineering smoke test only; not a benchmark")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    cfg["smoke_limit_per_split"] = args.limit_per_split
    validate_config(cfg)
    seed_everything(cfg["seed"])
    # Avoid very high CPU thread counts for small projection/pooling operations.
    torch.set_num_threads(4)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", cfg)
    env = environment(cfg)
    save_json(output / "environment.json", env)
    splits, audit = read_splits(cfg["data_dir"])
    source = Path(cfg["data_dir"]) / "source.json"
    if source.exists():
        audit["source"] = json.loads(source.read_text())
    if args.limit_per_split:
        from sklearn.model_selection import train_test_split
        for name, rows in splits.items():
            if len(rows) > args.limit_per_split:
                selected, _ = train_test_split(rows, train_size=args.limit_per_split, random_state=cfg["seed"],
                                               stratify=[r["label"] for r in rows])
                splits[name] = sorted(selected, key=lambda r: r["row_id"])
            # Include selected indices in cache key; raw file hash alone is insufficient.
            audit[name]["selection_sha256"] = hashlib.sha256(json.dumps([r["row_id"] for r in splits[name]]).encode()).hexdigest()
            audit[name]["selected_rows"] = [r["row_id"] for r in splits[name]]
        audit["smoke_only"] = True
    tokenizer, encoder, info = load_backbone(cfg, args.offline)
    save_json(output / "weight_loading.json", info)
    tokens = {s: TokenDataset(rows, tokenizer, cfg["max_tokens"]) for s, rows in splits.items()}
    for s, dataset in tokens.items():
        audit[s]["tokenization"] = dataset.stats
    save_json(output / "data_audit.json", audit)
    token_collator = TokenCollator(tokenizer.pad_token_id)
    encoder.to(cfg["device"])
    cache_report = {}
    if cfg["cache_features"]:
        datasets, cache_report = cache_features(cfg, encoder, tokens, token_collator, audit, env)
        collator = collate_features
    else:
        datasets, collator = tokens, token_collator
    save_json(output / "feature_cache.json", cache_report)
    results = []
    for variant in ["baseline", "fno"]:
        results.append(run_variant(variant, cfg, encoder, datasets, collator, tokens, token_collator, audit, cache_report))
    write_comparison(results, output)
    print(f"Completed paired comparison: {output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
