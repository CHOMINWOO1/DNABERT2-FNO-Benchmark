import os
import platform
import random
import time
from contextlib import nullcontext
from importlib.metadata import version

import numpy as np
import torch


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False


def amp_context(cfg):
    if cfg["precision"] == "fp32":
        return nullcontext()
    return torch.autocast(device_type=torch.device(cfg["device"]).type,
                          dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[cfg["precision"]])


def sync(cfg):
    if torch.device(cfg["device"]).type == "cuda":
        torch.cuda.synchronize()


def timed_start(cfg):
    sync(cfg)
    return time.perf_counter()


def elapsed(cfg, start):
    sync(cfg)
    return time.perf_counter() - start


def reset_peak(cfg):
    if torch.device(cfg["device"]).type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory(cfg):
    cuda = torch.device(cfg["device"]).type == "cuda"
    return {"allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
            "reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if cuda else None}


def move_batch(batch, cfg):
    return {k: v.to(cfg["device"], non_blocking=True) for k, v in batch.items()}


def environment(cfg):
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: version(p) for p in ["torch", "transformers", "numpy", "scikit-learn", "einops", "huggingface-hub"]},
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.device(cfg["device"]).type == "cuda" else None,
            "attention": "upstream_pytorch_eager", "deterministic_algorithms": True,
            "precision": cfg["precision"], "fft_precision": "float32_complex64"}
