from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = repo_root / "experiments" / "outputs"
    roc_points_path = outputs_dir / "gmm_roc_points.csv"
    out_path = outputs_dir / "roc_curve_hsv_gmm_only.png"

    if not roc_points_path.exists():
        raise FileNotFoundError(
            f"Missing {roc_points_path}. Run the notebook classical GMM section first to export ROC points."
        )

    roc_arr = np.loadtxt(roc_points_path, delimiter=",", skiprows=1)
    if roc_arr.ndim == 1:
        roc_arr = roc_arr.reshape(1, -1)
    if roc_arr.shape[1] < 2:
        raise ValueError(f"{roc_points_path} must contain two columns: fpr,tpr")

    fpr = roc_arr[:, 0]
    tpr = roc_arr[:, 1]
    gmm_auc = float(np.trapezoid(tpr, fpr))

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2.2, color="#e67e22", label=f"HSV GMM AUC = {gmm_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: HSV GMM (Classical Pipeline)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
