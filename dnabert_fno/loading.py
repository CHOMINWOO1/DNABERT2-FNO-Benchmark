"""Pinned original DNABERT-2 code with the upstream PyTorch attention path."""
import os
from pathlib import Path


def configure_cache(root):
    root = Path(root).resolve()
    os.environ.setdefault("HF_HOME", str(root / "huggingface"))
    os.environ.setdefault("HF_MODULES_CACHE", str(root / "modules"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_backbone(cfg, offline=False):
    configure_cache(cfg["cache_dir"])
    import sys
    from transformers import AutoModel, AutoTokenizer, BertConfig
    from .models import FrozenEncoder
    common = dict(revision=cfg["revision"], local_files_only=offline,
                  cache_dir=str(Path(cfg["cache_dir"]) / "huggingface" / "hub"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=False, **common)
    # Official README workaround for AutoConfig's custom BertConfig mismatch.
    config = BertConfig.from_pretrained(cfg["model_id"], **common)
    model, loading_info = AutoModel.from_pretrained(
        cfg["model_id"], config=config, trust_remote_code=True,
        output_loading_info=True, add_pooling_layer=False, **common
    )
    missing = loading_info["missing_keys"]
    if missing or loading_info.get("mismatched_keys"):
        raise RuntimeError(f"Pretrained encoder weights did not fully load: {loading_info}")
    # Select the author's ordinary torch path even on systems with Triton.
    module = sys.modules[type(model).__module__]
    module.flash_attn_qkvpacked_func = None
    return tokenizer, FrozenEncoder(model), loading_info
