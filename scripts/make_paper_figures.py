from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "experiments" / "outputs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_training_history() -> None:
    rows = read_csv(OUTPUT_DIR / "training_history.csv")
    epochs = np.arange(1, len(rows) + 1)
    aucs = np.array([float(row["val_auc"]) for row in rows])
    losses = np.array([float(row["loss"]) for row in rows])
    stages = [row["stage"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(epochs, aucs, marker="o", linewidth=2, color="#1f77b4", label="validation AUC")
    ax1.set_xlabel("Training epoch")
    ax1.set_ylabel("Validation AUC", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(max(0.5, aucs.min() - 0.04), min(1.0, aucs.max() + 0.04))
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(epochs, losses, marker="s", linewidth=1.6, color="#d62728", label="training loss")
    ax2.set_ylabel("Training loss", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    fine_tune_start = next((i + 1 for i, stage in enumerate(stages) if stage == "fine_tune"), None)
    if fine_tune_start is not None:
        ax1.axvline(fine_tune_start - 0.5, color="black", linestyle="--", linewidth=1)
        ax1.text(
            fine_tune_start - 0.35,
            ax1.get_ylim()[0] + 0.01,
            "fine-tuning",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_history.png", dpi=180)
    plt.close(fig)


def make_calibration_summary() -> None:
    rows = read_csv(OUTPUT_DIR / "validation_predictions.csv")
    y = np.array([int(row["target"]) for row in rows])
    p = np.array([float(row["prob_melanoma"]) for row in rows])

    bins = np.linspace(0.0, 1.0, 11)
    conf = []
    empirical = []
    counts = []
    for low, high in zip(bins[:-1], bins[1:]):
        if high == 1.0:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)
        counts.append(int(mask.sum()))
        if mask.any():
            conf.append(float(p[mask].mean()))
            empirical.append(float(y[mask].mean()))
        else:
            conf.append(np.nan)
            empirical.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1))

    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="perfect calibration")
    axes[0].plot(conf, empirical, marker="o", linewidth=2, color="#1f77b4", label="validation bins")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Mean predicted melanoma probability")
    axes[0].set_ylabel("Empirical melanoma frequency")
    axes[0].set_title("Reliability")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].hist(p[y == 0], bins=bins, alpha=0.78, color="#4c72b0", label="benign")
    axes[1].hist(p[y == 1], bins=bins, alpha=0.78, color="#c44e52", label="melanoma")
    axes[1].axvline(1 / 11, color="black", linestyle="--", linewidth=1, label="10:1 threshold")
    axes[1].set_xlabel("Predicted melanoma probability")
    axes[1].set_ylabel("Validation images")
    axes[1].set_title("Probability distribution")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration_probability_summary.png", dpi=180)
    plt.close(fig)

    with (OUTPUT_DIR / "calibration_bins.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_low", "bin_high", "count", "mean_probability", "empirical_melanoma_rate"])
        for low, high, count, mean_prob, rate in zip(bins[:-1], bins[1:], counts, conf, empirical):
            writer.writerow([low, high, count, mean_prob, rate])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_training_history()
    make_calibration_summary()
    print(f"Wrote figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
