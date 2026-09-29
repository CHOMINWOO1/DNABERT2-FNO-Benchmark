import csv

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dnabert_fno.data import TokenCollator, collate_features, read_splits
from dnabert_fno.metrics import binary_metrics
from dnabert_fno.models import RepresentationClassifier
from dnabert_fno.train import accumulation_denominator, loader_for, train_epoch, length_bucket_batches


def test_partial_accumulation_matches_true_large_batch_update():
    torch.manual_seed(7)
    a = RepresentationClassifier(4, dropout=0)
    b = RepresentationClassifier(4, dropout=0)
    b.load_state_dict(a.state_dict())
    items = [{"hidden": torch.randn(5, 4), "label": i % 2, "row_id": i} for i in range(11)]
    cfg = {"precision": "fp32", "device": "cpu", "batch_size": 3, "gradient_accumulation": 3}
    optimizer = torch.optim.SGD(a.parameters(), lr=0.01)
    train_epoch(a, None, DataLoader(items, batch_size=3, collate_fn=collate_features), optimizer,
                torch.amp.GradScaler("cuda", enabled=False), cfg)
    optimizer_b = torch.optim.SGD(b.parameters(), lr=0.01)
    for start, end in [(0, 9), (9, 11)]:
        batch = collate_features(items[start:end])
        optimizer_b.zero_grad()
        F.cross_entropy(b(batch["hidden"], batch["content_mask"]), batch["labels"]).backward()
        nn.utils.clip_grad_norm_(b.parameters(), 1.0)
        optimizer_b.step()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)


def test_shuffle_order_independent_of_model_rng():
    items = [{"hidden": torch.zeros(2, 4), "label": i % 2, "row_id": i} for i in range(19)]
    cfg = {"seed": 42, "batch_size": 4, "device": "cpu", "num_workers": 0}
    ids = lambda: torch.cat([b["row_ids"] for b in loader_for(items, collate_features, cfg, epoch=3)])
    a = ids()
    torch.randn(900)
    torch.testing.assert_close(a, ids())


def test_length_buckets_preserve_all_samples_and_only_last_batch_is_short():
    lengths = [10] * 19 + [11] * 7 + [20] * 14 + [3] * 3
    batches = length_bucket_batches(lengths, 8, torch.Generator().manual_seed(42))
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    assert all(len(b) == 8 for b in batches[:-1])
    assert len(batches[-1]) == len(lengths) % 8
    same = length_bucket_batches(lengths, 8, torch.Generator().manual_seed(42))
    assert batches == same


def test_metrics_perfect_reversed_and_single_class():
    perfect = binary_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert all(v == 1.0 for v in perfect.values())
    assert binary_metrics([0, 1], [0.9, 0.1])["mcc"] == -1
    one_class = binary_metrics([0, 0], [0.1, 0.2])
    assert one_class["roc_auc"] is None and one_class["pr_auc"] is None


def test_special_mask_collation():
    items = [{"input_ids": torch.tensor([1, 5, 2]), "content_mask": torch.tensor([False, True, False]), "label": 0, "row_id": 0},
             {"input_ids": torch.tensor([1, 6, 7, 2]), "content_mask": torch.tensor([False, True, True, False]), "label": 1, "row_id": 1}]
    batch = TokenCollator(0)(items)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert batch["content_mask"].sum(1).tolist() == [1, 2]


def test_split_audit_detects_reverse_complement_leakage(tmp_path):
    rows = {"train": [("AAAAC", 0), ("CCCCA", 1)], "dev": [("GTTTT", 0), ("ACCCC", 1)],
            "test": [("TATAT", 0), ("GGGGA", 1)]}
    for split, data in rows.items():
        with (tmp_path / f"{split}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence", "label"])
            writer.writerows(data)
    with pytest.warns(UserWarning, match="overlaps"):
        _, audit = read_splits(tmp_path)
    assert audit["overlap"]["train_dev"]["including_reverse_complement"] == 1
    assert audit["overlap"]["train_dev"]["exact_unique_sequences"] == 0
