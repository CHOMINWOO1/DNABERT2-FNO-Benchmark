import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FeatureDataset, save_json
from .runtime import amp_context, elapsed, move_batch, peak_memory, reset_peak, timed_start


def cache_features(cfg, encoder, tokens, collator, audit, env):
    identity = {"schema": 1, "model": cfg["model_id"], "revision": cfg["revision"],
                "max_tokens": cfg["max_tokens"], "precision": cfg["precision"],
                "environment": env, "batch_size": cfg["batch_size"],
                "splits": {s: {"file_sha256": audit[s]["sha256"],
                               "selection_sha256": audit[s].get("selection_sha256")} for s in tokens}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    root = Path(cfg["cache_dir"]) / "features" / key
    root.mkdir(parents=True, exist_ok=True)
    result, reports = {}, {}
    for name, dataset in tokens.items():
        dest = root / name
        manifest_path = dest / "manifest.json"
        if manifest_path.exists():
            report = json.loads(manifest_path.read_text())
            if report["identity"] != identity:
                raise ValueError("Cache metadata mismatch")
            hidden = np.load(dest / "hidden.npy", mmap_mode="r")
            offsets = np.load(dest / "offsets.npy")
            if list(hidden.shape) != report["shape"] or len(offsets) != len(dataset)+1 or offsets[-1] != len(hidden):
                raise ValueError("Cache is incomplete; use a new cache directory")
            del hidden
            report["reused"] = True
        else:
            dest.mkdir(parents=True, exist_ok=True)
            offsets = np.array([0] + [int(x["content_mask"].sum()) for x in dataset.items]).cumsum()
            np.save(dest / "offsets.npy", offsets)
            shape = (int(offsets[-1]), encoder.encoder.config.hidden_size)
            # Float32 storage preserves actual encoder outputs without an extra quantization step.
            hidden = np.lib.format.open_memmap(dest / "hidden.npy", mode="w+", dtype="float32", shape=shape)
            loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False,
                                collate_fn=collator, pin_memory=cfg["device"].startswith("cuda"),
                                num_workers=cfg["num_workers"])
            reset_peak(cfg)
            start = timed_start(cfg)
            row_index = 0
            for step, batch in enumerate(loader):
                batch = move_batch(batch, cfg)
                with torch.no_grad(), amp_context(cfg):
                    h = encoder(batch["input_ids"], batch["attention_mask"])
                packed = h[batch["content_mask"]].float().cpu().numpy()
                next_index = row_index + len(batch["labels"])
                hidden[offsets[row_index]:offsets[next_index]] = packed
                row_index = next_index
                if step % 100 == 0:
                    print(f"cache {name}: {row_index}/{len(dataset)}", flush=True)
            hidden.flush()
            del hidden, h, packed, batch
            report = {"identity": identity, "shape": list(shape), "storage_dtype": "float32",
                      "generation_seconds": elapsed(cfg, start), "peak_memory": peak_memory(cfg),
                      "reused": False}
            save_json(manifest_path, report)
        reports[name] = report
        result[name] = FeatureDataset(dest, dataset)
    return result, reports
