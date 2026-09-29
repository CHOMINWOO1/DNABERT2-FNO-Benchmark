"""Standalone scientific figures from COMPLETED study outputs (no GPU usage)."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {"baseline": "Frozen + linear", "mlp": "Frozen + MLP", "no_spectral": "No spectral (64)",
         "cnn": "Frozen + CNN", "fno": "Frozen + FNO", "lora": "LoRA", "lora_fno": "LoRA + FNO"}
COLORS = {"baseline": "#6b7280", "mlp": "#7189bf", "no_spectral": "#9b8dba", "cnn": "#57a99a",
          "fno": "#de8a32", "lora": "#2475a8", "lora_fno": "#ae426a"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=Path("runs/comparative_study"))
    p.add_argument("--output", type=Path, default=Path("docs/comparative_results"))
    args = p.parse_args()
    if json.loads((args.run_dir / "status.json").read_text())["phase"] != "complete":
        raise RuntimeError("Test results are not final yet")
    result = json.loads((args.run_dir / "summary.json").read_text())
    selected = json.loads((args.run_dir / "training_complete.json").read_text())["selected_runs"]
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    variants = list(result["summary"])
    n_seeds = len(result["protocol"]["seeds"])
    fig, (ax, delta_ax) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})
    jitter = np.linspace(-0.13, 0.13, len(result["protocol"]["seeds"]))
    for i, variant in enumerate(variants):
        record = result["summary"][variant]["metrics"]["mcc"]
        values = list(record["per_seed"].values())
        ax.scatter(values, i+jitter, color=COLORS[variant], s=23, alpha=0.75, zorder=3)
        ax.errorbar(record["mean"], i, xerr=record["sd"] or 0, fmt="D", color=COLORS[variant],
                    ms=6, capsize=4, linewidth=2, zorder=4)
        ax.annotate(f"{record['mean']:.3f}", (1.01, i), xycoords=("axes fraction", "data"), va="center", fontsize=10)
    minimum = min(v for r in result["summary"].values() for v in r["metrics"]["mcc"]["per_seed"].values())
    lower = min(0.0, float(np.floor((minimum - 0.02) * 10) / 10))
    ax.set(yticks=range(len(variants)), yticklabels=[NAMES[v] for v in variants], xlabel="Test MCC",
           xlim=(lower, 1), title=f"A   Performance across {n_seeds} paired seeds")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.18)
    comparisons = result["paired_mcc_differences"]
    labels = []
    for i, (name, record) in enumerate(comparisons.items()):
        a, b = name.split("_minus_")
        labels.append(f"{NAMES[a]}\nminus {NAMES[b]}")
        values = list(record["per_seed"].values())
        delta_ax.scatter(values, i+jitter, color=COLORS[a], s=23, alpha=0.7, zorder=3)
        delta_ax.errorbar(record["mean_delta_mcc"], i, xerr=record["sd_delta_mcc"] or 0,
                          fmt="D", color=COLORS[a], capsize=4, ms=5, linewidth=2)
    delta_ax.axvline(0, color="#333333", linewidth=1, linestyle="--")
    delta_ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel="Paired difference in MCC",
                 title="B   Within-seed differences")
    delta_ax.invert_yaxis(); delta_ax.grid(axis="x", alpha=0.18)
    fig.suptitle("DNABERT-2 adaptation on deduplicated GUE TATA promoters", fontsize=14, y=1.01)
    fig.text(0.5, 0.01, "Dots: individual training seeds. Diamonds and bars: mean ± sample SD, not confidence intervals.\n"
             "One fixed 300-bp task/split; hyperparameters selected on dev. Main test scores use unmerged LoRA.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.09, 1, 0.98), w_pad=3)
    fig.savefig(args.output / "performance.png", bbox_inches="tight")
    fig.savefig(args.output / "performance.svg", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    for ax, variant in zip(axes.flat, variants):
        for trial in selected:
            if trial["settings"]["variant"] != variant:
                continue
            history = json.loads((Path(trial["path"]) / "epochs.json").read_text())
            x = [r["epoch"] for r in history]; y = [r["dev"]["mcc"] for r in history]
            ax.plot(x, y, color=COLORS[variant], alpha=0.45, linewidth=1.1)
            chosen = next(r for r in history if r["epoch"] == trial["best_epoch"])
            ax.scatter(chosen["epoch"], chosen["dev"]["mcc"], color=COLORS[variant], s=18)
        ax.set_title(NAMES[variant]); ax.set_ylim(0, 1); ax.grid(alpha=0.15)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, 0.8, "Each line: one seed\nDot: selected checkpoint\nNo extrapolation after early stopping\n\nSelection metric: dev MCC\nPatience: 6 epochs\nMaximum: 25 epochs", va="top")
    fig.supxlabel("Epoch"); fig.supylabel("Validation MCC")
    fig.suptitle("Learning curves for the selected learning rate", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.output / "learning_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
