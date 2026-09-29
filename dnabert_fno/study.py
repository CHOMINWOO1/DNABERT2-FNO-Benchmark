"""Seven methods, dev-only LR selection on seed 42, then five paired seeds.

Test metrics are unavailable until ALL training/selection is complete. Existing
completed trials and per-epoch optimizer/RNG checkpoints support continuation.
"""
import argparse
import csv
import gc
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

from .adaptation import (VARIANTS, LoRAEncoder, StudyModel, make_classifier,
                         parameter_snapshot, load_trainable)
from .cache import cache_features
from .data import (TokenDataset, TokenCollator, collate_features, read_splits,
                   save_json, sha256_file)
from .loading import load_backbone
from .models import trainable_parameters
from .runtime import (seed_everything, environment, amp_context, move_batch,
                      timed_start, elapsed, reset_peak, peak_memory)
from .train import loader_for, accumulation_denominator, save_predictions, validate_config


def atomic_json(path, value):
    path = Path(path)
    partial = path.with_suffix(".tmp")
    save_json(partial, value)
    partial.replace(path)


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def frozen_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_model(cfg, variant, seed):
    seed_everything(seed)
    _, frozen, _ = load_backbone(cfg, offline=True)
    # Head and adapter init is independent of backbone construction RNG.
    seed_everything(seed)
    classifier = make_classifier(frozen.encoder.config.hidden_size, variant, cfg)
    if variant.startswith("lora"):
        # Identical LoRA initialization in the LoRA and LoRA+FNO arms, independent
        # of how many random values the classifier/adapter construction used.
        seed_everything(seed + 100000)
        encoder = LoRAEncoder(frozen.encoder, cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"])
    else:
        encoder = frozen
    model = StudyModel(encoder, classifier)
    model.classifier.to(cfg["device"])
    if variant.startswith("lora"):
        model.encoder.to(cfg["device"])
    seed_everything(seed)
    return model


def train_one_epoch(model, loader, optimizer, scaler, cfg):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0
    start = timed_start(cfg)
    for step, batch in enumerate(loader):
        batch = move_batch(batch, cfg)
        n = accumulation_denominator(step, len(loader.dataset), cfg["batch_size"], cfg["gradient_accumulation"])
        with amp_context(cfg):
            logits = model(batch)
            loss_sum = F.cross_entropy(logits.float(), batch["labels"], reduction="sum")
            loss = loss_sum / n
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss; trial stopped, not silently retried with a new LR")
        scaler.scale(loss).backward()
        total_loss += float(loss_sum.detach())
        if (step + 1) % cfg["gradient_accumulation"] == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    return total_loss / len(loader.dataset), elapsed(cfg, start)


@torch.no_grad()
def evaluate_study(model, dataset, collator, cfg):
    from .metrics import binary_metrics
    model.eval()
    predictions = {"row_id": [], "label": [], "probability": []}
    start = timed_start(cfg)
    for batch in loader_for(dataset, collator, cfg):
        batch = move_batch(batch, cfg)
        with amp_context(cfg):
            logits = model(batch)
        predictions["row_id"].extend(batch["row_ids"].cpu().tolist())
        predictions["label"].extend(batch["labels"].cpu().tolist())
        predictions["probability"].extend(logits.float().softmax(-1)[:, 1].cpu().tolist())
    seconds = elapsed(cfg, start)
    return binary_metrics(predictions["label"], predictions["probability"]), predictions, seconds


@torch.no_grad()
def benchmark_study(model, tokens, collator, cfg):
    model.eval()
    batches = []
    for i, batch in enumerate(loader_for(tokens, collator, cfg)):
        batches.append(batch)
        if i + 1 >= cfg["inference_repeats"]:
            break
    for _ in range(cfg["inference_warmup"]):
        model(move_batch(batches[0], cfg)) if cfg["precision"] == "fp32" else _amp_forward(model, batches[0], cfg)
    reset_peak(cfg)
    durations, samples = [], 0
    for batch in batches:
        start = timed_start(cfg)
        _amp_forward(model, batch, cfg)
        durations.append(elapsed(cfg, start))
        samples += len(batch["labels"])
    return {"scope": "encoder_plus_head_H2D_included_tokenization_and_disk_excluded",
            "lora_merged": isinstance(model.encoder, LoRAEncoder),
            "samples": samples, "batches": len(batches),
            "seconds": sum(durations), "samples_per_second": samples / sum(durations),
            "ms_per_sample": sum(durations) * 1000 / samples,
            "median_batch_ms": float(np.median(durations) * 1000),
            "p95_batch_ms": float(np.percentile(durations, 95) * 1000),
            "peak_memory": peak_memory(cfg)}


def _amp_forward(model, batch, cfg):
    batch = move_batch(batch, cfg)
    with amp_context(cfg):
        return model(batch)


def fit_trial(root, study_id, cfg, variant, seed, lr, datasets, collator):
    trial = root / "trials" / f"{variant}_seed{seed}_lr{lr:g}"
    trial.mkdir(parents=True, exist_ok=True)
    settings = {"study_id": study_id, "variant": variant, "seed": seed, "lr": lr}
    done = trial / "training.json"
    if done.exists():
        result = json.loads(done.read_text())
        if result["settings"] != settings:
            raise ValueError("Existing trial belongs to another protocol")
        return result
    cfg = {**cfg, "seed": seed}
    atomic_json(root / "status.json", {**settings, "phase": "training", "epoch": 0})
    model = make_model(cfg, variant, seed)
    head_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.classifier.head.parameters())).hexdigest()
    lora_hash = (hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.encoder.parameters()
                                        if p.requires_grad)).hexdigest() if variant.startswith("lora") else None)
    pretrained_hash = frozen_hash(model.encoder)
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    if variant.startswith("lora"):
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": lr},
            {"params": classifier_params, "lr": cfg["lora_head_learning_rate"]}
        ], weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(classifier_params, lr=lr, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["precision"] == "fp16")
    start_epoch, best_score, best_epoch, bad_epochs, history = 1, -float("inf"), 0, 0, []
    prior_peak = {"allocated_mib": 0, "reserved_mib": 0}
    resume_path = trial / "last.pt"
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        if state["settings"] != settings:
            raise ValueError("Resume checkpoint protocol mismatch")
        load_trainable(model, state["trainable"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_score, best_epoch, bad_epochs = state["best_score"], state["best_epoch"], state["bad_epochs"]
        history, prior_peak = state["history"], state["peak_memory"]
        torch.set_rng_state(state["cpu_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
    reset_peak(cfg)
    if bad_epochs < cfg["patience"]:
        for epoch in range(start_epoch, cfg["epochs"] + 1):
            epoch_start = timed_start(cfg)
            lrs = [g["lr"] for g in optimizer.param_groups]
            loss, train_s = train_one_epoch(model, loader_for(datasets["train"], collator, cfg, epoch), optimizer, scaler, cfg)
            dev, _, dev_s = evaluate_study(model, datasets["dev"], collator, cfg)
            score = dev[cfg["selection_metric"]]
            if score > best_score + cfg["min_delta"]:
                best_score, best_epoch, bad_epochs = score, epoch, 0
                torch.save({"settings": settings, "config": cfg, "epoch": epoch, "dev": dev,
                            "trainable": parameter_snapshot(model)}, trial / "best.pt")
            else:
                bad_epochs += 1
            scheduler.step()
            record = {"epoch": epoch, "train_loss": loss, "train_seconds": train_s, "dev_seconds": dev_s,
                      "epoch_wall_seconds": elapsed(cfg, epoch_start), "learning_rates": lrs,
                      "dev": dev, "best_epoch": best_epoch,
                      "alpha": float(model.classifier.operator.alpha.detach()) if model.classifier.operator else None}
            history.append(record)
            memory = {k: max(prior_peak[k] or 0, v or 0) for k, v in peak_memory(cfg).items()}
            torch.save({"settings": settings, "epoch": epoch, "trainable": parameter_snapshot(model),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        "history": history, "best_score": best_score, "best_epoch": best_epoch,
                        "bad_epochs": bad_epochs, "peak_memory": memory}, trial / "last.tmp")
            (trial / "last.tmp").replace(resume_path)
            atomic_json(trial / "epochs.json", history)
            atomic_json(root / "status.json", {**settings, "phase": "training", "epoch": epoch,
                                               "best_epoch": best_epoch, "best_dev_mcc": best_score,
                                               "train_epoch_seconds": train_s})
            print(f"{variant} seed={seed} lr={lr:g} epoch={epoch:02d} loss={loss:.4f} "
                  f"dev_MCC={score:.4f} best={best_epoch} train={train_s:.1f}s", flush=True)
            if bad_epochs >= cfg["patience"]:
                break
    assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    assert frozen_hash(model.encoder) == pretrained_hash
    memory = {k: max(prior_peak[k] or 0, v or 0) for k, v in peak_memory(cfg).items()}
    result = {"settings": settings, "path": str(trial), "trainable_params": trainable_parameters(model),
              "head_initial_sha256": head_hash, "lora_initial_sha256": lora_hash, "frozen_weights_sha256": pretrained_hash,
              "frozen_weights_unchanged": True, "epochs_completed": len(history),
              "best_epoch": best_epoch, "best_dev_mcc": best_score,
              "training_mode": "online_lora" if variant.startswith("lora") else "cached_frozen",
              "train_seconds": sum(r["train_seconds"] for r in history),
              "dev_seconds": sum(r["dev_seconds"] for r in history),
              "epoch_wall_seconds_total": sum(r["epoch_wall_seconds"] for r in history),
              "mean_train_epoch_seconds": float(np.mean([r["train_seconds"] for r in history])),
              "peak_memory": memory, "test_evaluated": False}
    atomic_json(done, result)
    del model, optimizer, scheduler, scaler
    gc.collect(); torch.cuda.empty_cache()
    return result


def evaluate_trial(root, cfg, selected, tokens, features, token_collator):
    variant, seed = selected["settings"]["variant"], selected["settings"]["seed"]
    path = root / "results" / f"{variant}_seed{seed}"
    path.mkdir(parents=True, exist_ok=True)
    done = path / "result.json"
    if done.exists():
        result = json.loads(done.read_text())
        if result["training"]["settings"] != selected["settings"]:
            raise ValueError("Evaluation checkpoint selection mismatch")
        return result
    cfg = {**cfg, "seed": seed}
    model = make_model(cfg, variant, seed)
    checkpoint = torch.load(Path(selected["path"]) / "best.pt", map_location="cpu", weights_only=True)
    load_trainable(model, checkpoint["trainable"])
    lora = variant.startswith("lora")
    model.encoder.to(cfg["device"])
    # Main biological metric uses the actual trained (unmerged) model.
    test, predictions, test_seconds = evaluate_study(model, tokens["test"] if lora else features["test"],
                                                     token_collator if lora else collate_features, cfg)
    save_predictions(path / "test_predictions.csv", predictions)
    merge_report = None
    if lora:
        model.encoder.merge()
        merged_metrics, merged_predictions, _ = evaluate_study(model, tokens["test"], token_collator, cfg)
        p, q = np.array(predictions["probability"]), np.array(merged_predictions["probability"])
        merge_report = {"test": merged_metrics, "max_abs_probability_difference": float(np.max(np.abs(p-q))),
                        "mean_abs_probability_difference": float(np.mean(np.abs(p-q))),
                        "changed_class_predictions": int(np.sum((p >= 0.5) != (q >= 0.5))),
                        "note": "BF16 separate versus merged matmuls can differ by rounding; FP32 algebra tested separately."}
        save_predictions(path / "merged_test_predictions.csv", merged_predictions)
    inference = benchmark_study(model, tokens["test"], token_collator, cfg)
    result = {"variant": variant, "seed": seed, "test": test, "test_seconds": test_seconds,
              "training": selected, "inference": inference, "lora_merge": merge_report,
              "best_checkpoint_sha256": sha256_file(Path(selected["path"]) / "best.pt")}
    atomic_json(done, result)
    print(f"TEST {variant} seed={seed}: MCC={test['mcc']:.4f}", flush=True)
    del model, checkpoint
    gc.collect(); torch.cuda.empty_cache()
    return result


def aggregate(root, cfg, results, selection, cache_report):
    measures = ["accuracy", "f1", "mcc", "roc_auc", "pr_auc", "average_precision"]
    summary = {}
    for variant in cfg["variants"]:
        subset = sorted([r for r in results if r["variant"] == variant], key=lambda r: r["seed"])
        summary[variant] = {"n_seeds": len(subset), "trainable_params": subset[0]["training"]["trainable_params"],
                            "selected_lr": selection[variant]["lr"], "metrics": {}}
        for name in measures:
            values = [r["test"][name] for r in subset]
            summary[variant]["metrics"][name] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values)>1 else None,
                                                 "per_seed": {str(r["seed"]): r["test"][name] for r in subset}}
        summary[variant]["mean_train_seconds"] = float(np.mean([r["training"]["train_seconds"] for r in subset]))
        summary[variant]["max_train_peak_mib"] = max(r["training"]["peak_memory"]["allocated_mib"] for r in subset)
        summary[variant]["mean_inference_samples_per_second"] = float(np.mean([r["inference"]["samples_per_second"] for r in subset]))
    comparisons = [("fno", "baseline"), ("fno", "mlp"), ("fno", "cnn"), ("fno", "no_spectral"),
                   ("fno", "lora"), ("lora_fno", "lora"), ("lora_fno", "fno")]
    paired = {}
    for a, b in comparisons:
        if a not in summary or b not in summary:
            continue
        av, bv = summary[a]["metrics"]["mcc"]["per_seed"], summary[b]["metrics"]["mcc"]["per_seed"]
        diffs = {s: av[s]-bv[s] for s in av}
        values = list(diffs.values())
        paired[f"{a}_minus_{b}"] = {"per_seed": diffs, "mean_delta_mcc": float(np.mean(values)),
                                    "sd_delta_mcc": float(np.std(values, ddof=1)) if len(values)>1 else None,
                                    "positive_seeds": sum(v > 0 for v in values)}
    output = {"protocol": cfg, "summary": summary, "paired_mcc_differences": paired,
              "cache_generation_seconds_shared": sum(r["generation_seconds"] for r in cache_report.values()),
              "limitation": "One cleaned random split of one short task. Seed SD is training variation, not external generalization uncertainty."}
    atomic_json(root / "summary.json", output)
    rows = []
    for r in results:
        rows.append({"variant": r["variant"], "seed": r["seed"], **r["test"],
                     "trainable_params": r["training"]["trainable_params"], "lr": r["training"]["settings"]["lr"],
                     "best_epoch": r["training"]["best_epoch"], "epochs": r["training"]["epochs_completed"],
                     "train_seconds": r["training"]["train_seconds"],
                     "train_peak_mib": r["training"]["peak_memory"]["allocated_mib"],
                     "inference_samples_per_second": r["inference"]["samples_per_second"]})
    with (root / "all_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Controlled DNABERT-2 adapter comparison", "",
             "Mean ± sample SD across the same seeds. Primary test metrics use unmerged LoRA; throughput uses merged LoRA.", "",
             "| Model | Params | Accuracy | F1 | MCC | ROC-AUC | PR-AUC | Train s | Inference samples/s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, r in summary.items():
        vals = [f"{r['metrics'][m]['mean']:.4f} ± {r['metrics'][m]['sd']:.4f}" if r['metrics'][m]['sd'] is not None else f"{r['metrics'][m]['mean']:.4f}" for m in measures[:5]]
        lines.append(f"| {variant} | {r['trainable_params']} | " + " | ".join(vals) + f" | {r['mean_train_seconds']:.1f} | {r['mean_inference_samples_per_second']:.1f} |")
    lines += ["", "## Paired MCC differences", "", "| Comparison | Mean delta | SD | Positive seeds |", "|---|---:|---:|---:|"]
    for name, r in paired.items():
        sd = f"{r['sd_delta_mcc']:.4f}" if r['sd_delta_mcc'] is not None else "N/A"
        lines.append(f"| {name} | {r['mean_delta_mcc']:+.4f} | {sd} | {r['positive_seeds']}/{len(cfg['seeds'])} |")
    lines += ["", output["limitation"], "",
              "Frozen methods use shared cached features; LoRA uses online forward/backward. Shared cache generation is separately logged.",
              "MLP/CNN parameter counts match FNO within 1%; rank-6 fused-QKV LoRA has about 6.3% fewer parameters than FNO.",
              "LoRA+FNO has more capacity; an improvement alone cannot establish synergy. All hyperparameter selection used dev only."]
    (root / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Unable to import Triton.*", category=UserWarning)
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/comparative_study.json")
    p.add_argument("--output-dir")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.smoke:
        cfg.update(epochs=2, seeds=[42], frozen_learning_rates=[0.001], lora_learning_rates=[0.0001], smoke=True)
    validate_config(cfg)
    seed_everything(cfg["seed"]); torch.set_num_threads(4)
    root = Path(cfg["output_dir"]); root.mkdir(parents=True, exist_ok=True)
    source_files = sorted(Path("dnabert_fno").glob("*.py"))
    identity = {"config": cfg, "environment": environment(cfg),
                "data_sha256": {s: sha256_file(Path(cfg["data_dir"])/f"{s}.csv") for s in ["train","dev","test"]},
                "code_sha256": {str(path): sha256_file(path) for path in source_files}}
    study_id = stable_hash(identity)
    if (root / "protocol.json").exists():
        if json.loads((root / "protocol.json").read_text()) != identity:
            raise ValueError("Protocol/code/data changed; choose a new output directory")
    else:
        atomic_json(root / "protocol.json", identity)
    atomic_json(root / "environment.json", environment(cfg))
    splits, audit = read_splits(cfg["data_dir"])
    if any(v["including_reverse_complement"] for v in audit["overlap"].values()):
        raise ValueError("Controlled study requires zero exact/RC overlap")
    if args.smoke:
        for name, rows in splits.items():
            sample, _ = train_test_split(rows, train_size=32, random_state=42, stratify=[r["label"] for r in rows])
            splits[name] = sorted(sample, key=lambda r: r["row_id"])
            audit[name]["selection_sha256"] = stable_hash([r["row_id"] for r in splits[name]])
    tokenizer, frozen, info = load_backbone(cfg, offline=True)
    token_collator = TokenCollator(tokenizer.pad_token_id)
    tokens = {s: TokenDataset(rows, tokenizer, cfg["max_tokens"]) for s, rows in splits.items()}
    for s in tokens:
        audit[s]["tokenization"] = tokens[s].stats
    atomic_json(root / "data_audit.json", audit)
    atomic_json(root / "weight_loading.json", info)
    frozen.to(cfg["device"])
    features, cache_report = cache_features(cfg, frozen, tokens, token_collator, audit, environment(cfg))
    atomic_json(root / "feature_cache.json", cache_report)
    del frozen
    gc.collect(); torch.cuda.empty_cache()
    selection = {}
    selected_runs = []
    for variant in cfg["variants"]:
        is_lora = variant.startswith("lora")
        data, collator = (tokens, token_collator) if is_lora else (features, collate_features)
        learning_rates = cfg["lora_learning_rates"] if is_lora else cfg["frozen_learning_rates"]
        candidates = [fit_trial(root, study_id, cfg, variant, 42, lr, data, collator) for lr in learning_rates]
        # Tie -> earlier (lower) LR, fixed before looking at any test predictions.
        best = max(candidates, key=lambda x: x["best_dev_mcc"])
        selection[variant] = {"lr": best["settings"]["lr"], "pilot_seed": 42,
                              "candidate_dev_mcc": {str(c["settings"]["lr"]): c["best_dev_mcc"] for c in candidates}}
        for seed in cfg["seeds"]:
            selected_runs.append(best if seed == 42 else fit_trial(root, study_id, cfg, variant, seed,
                                                                  best["settings"]["lr"], data, collator))
        atomic_json(root / "selection.json", selection)
    # Check fair initial head per seed before opening test outcomes.
    for seed in cfg["seeds"]:
        hashes = {r["head_initial_sha256"] for r in selected_runs if r["settings"]["seed"] == seed}
        assert len(hashes) == 1
        lora_hashes = {r["lora_initial_sha256"] for r in selected_runs if r["settings"]["seed"] == seed
                       and r["settings"]["variant"].startswith("lora")}
        assert len(lora_hashes) <= 1
    atomic_json(root / "training_complete.json", {"study_id": study_id, "selected_runs": selected_runs, "selection": selection})
    results = []
    for chosen in selected_runs:
        atomic_json(root / "status.json", {"phase": "test_evaluation", **chosen["settings"]})
        results.append(evaluate_trial(root, cfg, chosen, tokens, features, token_collator))
    aggregate(root, cfg, results, selection, cache_report)
    atomic_json(root / "status.json", {"phase": "complete", "selected_runs": len(selected_runs), "study_id": study_id})
    print(f"STUDY COMPLETE: {root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
