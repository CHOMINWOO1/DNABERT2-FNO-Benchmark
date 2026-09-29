"""Download pinned weights and exercise actual CUDA/BPE/AMP/frozen inference."""
import argparse
import json
import torch
from dnabert_fno.loading import load_backbone
from dnabert_fno.models import RepresentationClassifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/promoter.json")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    cfg = json.load(open(args.config))
    tokenizer, encoder, info = load_backbone(cfg, args.offline)
    device = cfg["device"]
    encoder.to(device)
    inputs = tokenizer(["ACGT" * 75, "TATAAA" + "GCTAT" * 43], padding=True,
                       return_tensors="pt", return_special_tokens_mask=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    content = inputs["attention_mask"].bool() & ~inputs["special_tokens_mask"].bool()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        hidden = encoder(inputs["input_ids"], inputs["attention_mask"])
        model = RepresentationClassifier(hidden.shape[-1], variant="fno").to(device)
        logits = model(hidden, content)
        logits.square().mean().backward()
    assert not hidden.requires_grad
    assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
    assert model.operator.alpha.grad is not None
    assert torch.isfinite(logits).all()
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
                      "hidden_shape": list(hidden.shape), "dtype": str(hidden.dtype),
                      "content_tokens": content.sum(1).tolist(),
                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
                      "loading_info": info, "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
