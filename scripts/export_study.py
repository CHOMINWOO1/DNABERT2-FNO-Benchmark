"""Export small, reviewable study artifacts without model weights or feature caches."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stats(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
            "min": float(min(values)), "max": float(max(values))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=Path("runs/comparative_study"))
    p.add_argument("--output", type=Path, default=Path("docs/comparative_results"))
    args = p.parse_args(); root, output = args.run_dir, args.output
    assert read(root / "status.json")["phase"] == "complete"
    assert read(root / "verification.json")["passed"]
    output.mkdir(parents=True, exist_ok=True)
    manifest = []

    def copy(source, relative):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest.append({"source": str(source), "export": str(relative),
                         "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})

    for name in ["protocol.json", "environment.json", "selection.json", "training_complete.json",
                 "data_audit.json", "weight_loading.json", "feature_cache.json", "status.json",
                 "summary.json", "summary.md", "all_runs.csv", "verification.json"]:
        copy(root / name, Path(name))
    for name in ["pytest.txt", "merge_precision_check.json"]:
        if (root / name).exists():
            copy(root / name, Path(name))
    cfg = read(root / "protocol.json")["config"]
    copy(Path(cfg["data_dir"]) / "source.json", Path("data_source.json"))
    copy(Path(cfg["data_dir"]) / "assignment.json", Path("data_assignment.json"))
    trials = [read(path) for path in (root / "trials").glob("*/training.json")]
    for path in (root / "trials").glob("*/training.json"):
        copy(path, path.relative_to(root))
        copy(path.parent / "epochs.json", path.relative_to(root).with_name("epochs.json"))
    records = []
    for path in (root / "results").glob("*/result.json"):
        records.append(read(path))
        for child in path.parent.iterdir():
            if child.suffix in {".json", ".csv"}:
                copy(child, child.relative_to(root))
    extra = {"seed_43_to_46_only": {}, "costs": {}, "merged_lora": {}}
    for variant in cfg["variants"]:
        selected = [r for r in records if r["variant"] == variant]
        heldout_seeds = [r for r in selected if r["seed"] != 42]
        extra["seed_43_to_46_only"][variant] = stats([r["test"]["mcc"] for r in heldout_seeds])
        selected_paths = {r["training"]["path"] for r in selected}
        pilots = [t for t in trials if t["settings"]["variant"] == variant and t["path"] not in selected_paths]
        extra["costs"][variant] = {
            "selected_train_seconds": stats([r["training"]["train_seconds"] for r in selected]),
            "selected_train_plus_dev_seconds": stats([r["training"]["train_seconds"] + r["training"]["dev_seconds"] for r in selected]),
            "discarded_pilot_train_seconds": sum(r["train_seconds"] for r in pilots),
            "all_trials_train_seconds": sum(r["train_seconds"] for r in trials if r["settings"]["variant"] == variant),
            "train_epochs": {str(r["seed"]): r["training"]["epochs_completed"] for r in selected},
            "best_epochs": {str(r["seed"]): r["training"]["best_epoch"] for r in selected},
            "inference_samples_per_second": stats([r["inference"]["samples_per_second"] for r in selected]),
            "max_train_allocated_mib": max(r["training"]["peak_memory"]["allocated_mib"] for r in selected),
            "max_inference_allocated_mib": max(r["inference"]["peak_memory"]["allocated_mib"] for r in selected)}
        if variant.startswith("lora"):
            extra["merged_lora"][variant] = {
                "metrics": {m: stats([r["lora_merge"]["test"][m] for r in selected]) for m in selected[0]["test"]},
                "changed_class_predictions_per_seed": {str(r["seed"]): r["lora_merge"]["changed_class_predictions"] for r in selected},
                "max_abs_probability_difference": max(r["lora_merge"]["max_abs_probability_difference"] for r in selected),
                "mean_abs_probability_difference": float(np.mean([r["lora_merge"]["mean_abs_probability_difference"] for r in selected]))}
    paired = read(root / "summary.json")["paired_mcc_differences"]
    extra["paired_mcc_seed_43_to_46_only"] = {
        name: {**stats([v for s, v in record["per_seed"].items() if s != "42"]),
               "positive_seeds": sum(v > 0 for s, v in record["per_seed"].items() if s != "42")}
        for name, record in paired.items()}
    extra["shared_cache_seconds"] = sum(r["generation_seconds"] for r in read(root / "feature_cache.json").values())
    extra["shared_cache_max_allocated_mib"] = max(r["peak_memory"]["allocated_mib"] for r in read(root / "feature_cache.json").values())
    extra["all_trials_total_train_seconds"] = sum(r["train_seconds"] for r in trials)
    extra["timing_note"] = "Train time excludes dev, model loading, checkpoint I/O, and shared feature extraction. No cross-session timing replication."
    (output / "supplementary.json").write_text(json.dumps(extra, indent=2) + "\n", encoding="utf-8")
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(manifest)} source artifacts to {output.resolve()}")


if __name__ == "__main__":
    main()
