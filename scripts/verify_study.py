"""Audit a completed comparative study from saved predictions and provenance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_curve, auc, average_precision_score)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_predictions(path, expected_labels, metrics):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [int(r["row_id"]) for r in rows]
    assert ids == list(range(len(expected_labels))), (path, "missing, duplicated, or reordered rows")
    y = np.array([int(r["label"]) for r in rows])
    p = np.array([float(r["probability"]) for r in rows])
    assert np.array_equal(y, expected_labels), (path, "label mismatch")
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    predicted = p >= 0.5
    tp = int(((y == 1) & predicted).sum()); tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum()); fn = int(((y == 1) & ~predicted).sum())
    denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    precision, recall, _ = precision_recall_curve(y, p)
    recomputed = {"accuracy": accuracy_score(y, predicted), "f1": f1_score(y, predicted),
                  "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
                  "roc_auc": roc_auc_score(y, p), "pr_auc": auc(recall, precision),
                  "average_precision": average_precision_score(y, p)}
    for name, value in recomputed.items():
        assert np.isclose(value, metrics[name], atol=1e-12, rtol=0), (path, name)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/comparative_study"))
    args = parser.parse_args(); root = args.run_dir
    status = read_json(root / "status.json")
    assert status["phase"] == "complete", "Study is not complete"
    protocol = read_json(root / "protocol.json"); cfg = protocol["config"]
    assert not cfg.get("smoke"), "This audit requires the full test set"
    for path, digest in protocol["code_sha256"].items():
        assert sha256(path) == digest, (path, "code changed since training")
    for split, digest in protocol["data_sha256"].items():
        assert sha256(Path(cfg["data_dir"]) / f"{split}.csv") == digest
    with (Path(cfg["data_dir"]) / "test.csv").open(newline="", encoding="utf-8-sig") as f:
        labels = np.array([int(r["label"]) for r in csv.DictReader(f)])
    complete = read_json(root / "training_complete.json")
    selected = complete["selected_runs"]
    pairs = {(r["settings"]["variant"], r["settings"]["seed"]) for r in selected}
    assert pairs == {(v, s) for v in cfg["variants"] for s in cfg["seeds"]}
    trials = [read_json(p) for p in (root / "trials").glob("*/training.json")]
    expected_trials = sum(len(cfg["seeds"]) + len(cfg["lora_learning_rates"] if v.startswith("lora")
                                                    else cfg["frozen_learning_rates"]) - 1 for v in cfg["variants"])
    assert len(trials) == expected_trials
    assert all(t["frozen_weights_unchanged"] and not t["test_evaluated"] for t in trials)
    for seed in cfg["seeds"]:
        same_seed = [t for t in selected if t["settings"]["seed"] == seed]
        assert len({t["head_initial_sha256"] for t in same_seed}) == 1
        assert len({t["lora_initial_sha256"] for t in same_seed if t["settings"]["variant"].startswith("lora")}) == 1
    records = []
    for chosen in selected:
        variant, seed = chosen["settings"]["variant"], chosen["settings"]["seed"]
        path = root / "results" / f"{variant}_seed{seed}"
        record = read_json(path / "result.json")
        assert record["training"] == chosen
        assert record["best_checkpoint_sha256"] == sha256(Path(chosen["path"]) / "best.pt")
        p = verify_predictions(path / "test_predictions.csv", labels, record["test"])
        if record["lora_merge"]:
            report = record["lora_merge"]
            q = verify_predictions(path / "merged_test_predictions.csv", labels, report["test"])
            assert np.isclose(np.max(np.abs(p - q)), report["max_abs_probability_difference"], atol=1e-12)
            assert np.isclose(np.mean(np.abs(p - q)), report["mean_abs_probability_difference"], atol=1e-12)
            assert int(((p >= 0.5) != (q >= 0.5)).sum()) == report["changed_class_predictions"]
        records.append(record)
    summary = read_json(root / "summary.json")
    for variant, group in summary["summary"].items():
        subset = [r for r in records if r["variant"] == variant]
        for metric, report in group["metrics"].items():
            values = [r["test"][metric] for r in subset]
            assert np.isclose(np.mean(values), report["mean"], atol=1e-12)
            assert np.isclose(np.std(values, ddof=1), report["sd"], atol=1e-12)
    for comparison, report in summary["paired_mcc_differences"].items():
        a, b = comparison.split("_minus_")
        av = summary["summary"][a]["metrics"]["mcc"]["per_seed"]
        bv = summary["summary"][b]["metrics"]["mcc"]["per_seed"]
        assert all(np.isclose(av[s] - bv[s], d, atol=1e-12) for s, d in report["per_seed"].items())
    output = {"passed": True, "study_id": status["study_id"], "training_trials": len(trials),
              "selected_runs": len(records), "test_rows_per_run": len(labels),
              "checks": ["source/data/checkpoint SHA256", "complete variant/seed coverage",
                         "unchanged frozen weights recorded in all trials", "paired head and LoRA initialization",
                         "prediction row IDs and labels", "six metrics recomputed from probabilities",
                         "MCC independently calculated from confusion counts", "LoRA merge differences",
                         "summary means and sample SD", "paired MCC differences"]}
    (root / "verification.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
