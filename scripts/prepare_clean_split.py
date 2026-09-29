"""Deduplicate exact/RC groups, reject conflicting groups, then stratify 80/10/10."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from dnabert_fno.data import read_splits, save_json, sha256_file


def canonical(sequence):
    return min(sequence, sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1])


def deduplicate(splits):
    groups = {}
    for split, rows in splits.items():
        for row in rows:
            groups.setdefault(canonical(row["sequence"]), []).append({**row, "original_split": split})
    accepted, conflicts = [], []
    for key in sorted(groups):
        rows = groups[key]
        record = {"group_sha256": hashlib.sha256(key.encode()).hexdigest(),
                  "members": [{"split": r["original_split"], "row_id": r["row_id"], "label": r["label"]} for r in rows]}
        if len({r["label"] for r in rows}) != 1:
            conflicts.append(record)
        else:
            # Keep a deterministically chosen ORIGINAL orientation (do not RC canonicalize inputs).
            accepted.append({**record, "sequence": rows[0]["sequence"], "label": rows[0]["label"]})
    return accepted, conflicts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("data/GUE/prom/prom_300_tata"))
    p.add_argument("--output", type=Path, default=Path("data/GUE_clean/prom_300_tata"))
    p.add_argument("--split-seed", type=int, default=20260908)
    args = p.parse_args()
    splits, original_audit = read_splits(args.source)
    rows, conflicts = deduplicate(splits)
    train, heldout = train_test_split(rows, test_size=0.2, random_state=args.split_seed,
                                     stratify=[r["label"] for r in rows])
    dev, test = train_test_split(heldout, test_size=0.5, random_state=args.split_seed,
                                stratify=[r["label"] for r in heldout])
    args.output.mkdir(parents=True, exist_ok=False)
    assignment = {}
    for name, records in {"train": train, "dev": dev, "test": test}.items():
        records.sort(key=lambda r: r["group_sha256"])
        with (args.output / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence", "label"])
            writer.writerows((r["sequence"], r["label"]) for r in records)
        assignment[name] = [{"row_id": i, "group_sha256": r["group_sha256"], "members": r["members"]}
                            for i, r in enumerate(records)]
    _, audit = read_splits(args.output)
    assert all(r["including_reverse_complement"] == 0 for r in audit["overlap"].values())
    source = {"method": "exact_and_RC_deduplicated_stratified_80_10_10",
              "split_seed": args.split_seed, "original_rows": sum(len(s) for s in splits.values()),
              "retained_unique_groups": len(rows), "conflicting_groups_excluded": conflicts,
              "removed_redundant_or_conflicting_rows": sum(len(s) for s in splits.values()) - len(rows),
              "original_audit": original_audit, "clean_audit": audit,
              "upstream": json.loads((args.source / "source.json").read_text()),
              "limitation": "New split of an already explored dataset, not an independent external test; near-homology unchecked."}
    save_json(args.output / "source.json", source)
    save_json(args.output / "assignment.json", assignment)
    print(json.dumps({"retained_unique_groups": len(rows), "conflicting_groups": len(conflicts),
                      "sizes": {s: audit[s]["rows"] for s in ["train", "dev", "test"]},
                      "overlap": audit["overlap"]}, indent=2))


if __name__ == "__main__":
    main()
