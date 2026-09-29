import csv
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_splits(data_dir):
    splits, audit = {}, {}
    for name in ["train", "dev", "test"]:
        path = Path(data_dir) / f"{name}.csv"
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            seq_key = next((s for s in ["sequence", "seq", "text"] if s in fields), None)
            label_key = next((s for s in ["label", "labels"] if s in fields), None)
            if not seq_key or not label_key:
                raise ValueError(f"{path}: expected sequence,label columns, found {fields}")
            rows = []
            for i, row in enumerate(reader):
                seq = row[seq_key].strip().upper()
                if not seq or set(seq) - set("ACGTN"):
                    raise ValueError(f"{path} row {i+2}: expected raw ACGTN sequence, not k-mers")
                label = int(row[label_key])
                if label not in (0, 1):
                    raise ValueError("Only binary labels 0/1 are supported")
                rows.append({"sequence": seq, "label": label, "row_id": i})
        if set(r["label"] for r in rows) != {0, 1}:
            raise ValueError(f"{name} must contain both classes")
        splits[name] = rows
        counts = Counter(r["sequence"] for r in rows)
        audit[name] = {"rows": len(rows), "label_counts": dict(Counter(r["label"] for r in rows)),
                       "bp_min": min(len(r["sequence"]) for r in rows),
                       "bp_max": max(len(r["sequence"]) for r in rows),
                       "duplicate_rows": sum(n - 1 for n in counts.values()),
                       "sha256": sha256_file(path)}
    canonical = lambda s: min(s, s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1])
    audit["overlap"] = {}
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        exact = len({r["sequence"] for r in splits[a]} & {r["sequence"] for r in splits[b]})
        rc = len({canonical(r["sequence"]) for r in splits[a]} & {canonical(r["sequence"]) for r in splits[b]})
        a_canonical = {canonical(r["sequence"]) for r in splits[a]}
        matching_rows = [r["row_id"] for r in splits[b] if canonical(r["sequence"]) in a_canonical]
        audit["overlap"][f"{a}_{b}"] = {"exact_unique_sequences": exact, "including_reverse_complement": rc,
                                       "overlapping_row_ids_in_second_split": matching_rows}
        if rc:
            warnings.warn(f"Provided {a}/{b} have {rc} identical/RC overlaps. Splits preserved; inspect audit.")
    return splits, audit


class TokenDataset(Dataset):
    def __init__(self, rows, tokenizer, max_tokens):
        self.rows = rows
        sequences = [r["sequence"] for r in rows]
        raw = tokenizer(sequences, truncation=False, add_special_tokens=True)
        lengths = [len(x) for x in raw["input_ids"]]
        self.stats = {"tokens_min_before_truncation": min(lengths), "tokens_max_before_truncation": max(lengths),
                      "tokens_mean_before_truncation": float(np.mean(lengths)),
                      "truncated_rows": sum(n > max_tokens for n in lengths), "max_tokens": max_tokens}
        encoded = tokenizer(sequences, truncation=True, max_length=max_tokens,
                            return_special_tokens_mask=True, padding=False)
        self.items = []
        for i, row in enumerate(rows):
            ids = torch.tensor(encoded["input_ids"][i], dtype=torch.long)
            content = ~torch.tensor(encoded["special_tokens_mask"][i], dtype=torch.bool)
            if not content.any():
                raise ValueError(f"No valid content tokens at row {row['row_id']}")
            self.items.append({"input_ids": ids, "content_mask": content,
                               "label": row["label"], "row_id": row["row_id"]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class TokenCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, items):
        n = max(len(x["input_ids"]) for x in items)
        ids = torch.full((len(items), n), self.pad_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        content = torch.zeros_like(ids, dtype=torch.bool)
        for i, x in enumerate(items):
            length = len(x["input_ids"])
            ids[i, :length] = x["input_ids"]
            attention[i, :length] = 1
            content[i, :length] = x["content_mask"]
        return {"input_ids": ids, "attention_mask": attention, "content_mask": content,
                "labels": torch.tensor([x["label"] for x in items]),
                "row_ids": torch.tensor([x["row_id"] for x in items])}


class FeatureDataset(Dataset):
    def __init__(self, directory, token_dataset):
        self.directory = Path(directory)
        self.tokens = token_dataset
        self.offsets = np.load(self.directory / "offsets.npy")
        self.hidden = None

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        if self.hidden is None:
            self.hidden = np.load(self.directory / "hidden.npy", mmap_mode="r")
        x = self.tokens[i]
        start, end = self.offsets[i:i+2]
        return {"hidden": torch.from_numpy(np.array(self.hidden[start:end], copy=True)),
                "label": x["label"], "row_id": x["row_id"]}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["hidden"] = None
        return state


def collate_features(items):
    n = max(len(x["hidden"]) for x in items)
    h = torch.zeros(len(items), n, items[0]["hidden"].shape[-1])
    mask = torch.zeros(len(items), n, dtype=torch.bool)
    for i, x in enumerate(items):
        length = len(x["hidden"])
        h[i, :length] = x["hidden"]
        mask[i, :length] = True
    return {"hidden": h, "content_mask": mask,
            "labels": torch.tensor([x["label"] for x in items]),
            "row_ids": torch.tensor([x["row_id"] for x in items])}
