"""Post-study numerical diagnostic; never changes primary scores or selection."""
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from dnabert_fno.adaptation import load_trainable
from dnabert_fno.data import read_splits, TokenDataset, TokenCollator
from dnabert_fno.loading import load_backbone
from dnabert_fno.study import make_model, evaluate_study


def main():
    root = Path("runs/comparative_study")
    assert json.loads((root / "status.json").read_text())["phase"] == "complete"
    cfg = json.loads((root / "protocol.json").read_text())["config"]
    cfg = {**cfg, "precision": "fp32"}
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    tokenizer, frozen, _ = load_backbone(cfg, offline=True)
    del frozen
    splits, _ = read_splits(cfg["data_dir"])
    dataset = TokenDataset(splits["test"], tokenizer, cfg["max_tokens"])
    collator = TokenCollator(tokenizer.pad_token_id)
    output = {"scope": "Post hoc numerical check only; primary BF16 results unchanged. All 604 test rows.",
              "precision": "fp32", "tf32": False, "runs": []}
    for path in sorted((root / "results").glob("lora*/result.json")):
        result = json.loads(path.read_text()); variant, seed = result["variant"], result["seed"]
        model = make_model(cfg, variant, seed)
        state = torch.load(Path(result["training"]["path"]) / "best.pt", map_location="cpu", weights_only=True)
        load_trainable(model, state["trainable"])
        metrics, before, _ = evaluate_study(model, dataset, collator, cfg)
        model.encoder.merge()
        merged_metrics, after, _ = evaluate_study(model, dataset, collator, cfg)
        p, q = np.array(before["probability"]), np.array(after["probability"])
        with (path.parent / "test_predictions.csv").open(newline="") as f:
            bf16 = np.array([float(r["probability"]) for r in csv.DictReader(f)])
        record = {"variant": variant, "seed": seed, "unmerged_test": metrics, "merged_test": merged_metrics,
                  "max_merge_abs_probability_difference": float(np.max(np.abs(p-q))),
                  "mean_merge_abs_probability_difference": float(np.mean(np.abs(p-q))),
                  "merge_changed_class_predictions": int(((p >= 0.5) != (q >= 0.5)).sum()),
                  "unmerged_fp32_vs_primary_bf16_max_probability_difference": float(np.max(np.abs(p-bf16))),
                  "unmerged_fp32_vs_primary_bf16_changed_classes": int(((p >= 0.5) != (bf16 >= 0.5)).sum())}
        output["runs"].append(record)
        print(f"{variant} seed={seed}: FP32 merge max delta={record['max_merge_abs_probability_difference']:.3g}, "
              f"class flips={record['merge_changed_class_predictions']}", flush=True)
        del model, state
        gc.collect(); torch.cuda.empty_cache()
    output["max_merge_probability_difference"] = max(r["max_merge_abs_probability_difference"] for r in output["runs"])
    output["merge_class_flips_total"] = sum(r["merge_changed_class_predictions"] for r in output["runs"])
    for variant in ["lora", "lora_fno"]:
        values = [r["unmerged_test"]["mcc"] for r in output["runs"] if r["variant"] == variant]
        output[f"{variant}_fp32_unmerged_mcc"] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}
    (root / "merge_precision_check.json").write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
