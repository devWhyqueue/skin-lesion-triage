# Bayesian Decision and Uncertainty: Skin Lesion Triage

This project studies melanoma-vs-benign skin lesion triage under asymmetric clinical risk. Instead of using a default symmetric threshold, it applies Bayesian decision thresholds where the cost of a false negative is much higher than the cost of a false positive.

## Artifacts

- `experiments/skin-lesion-triage.ipynb`: Kaggle notebook for the HAM10000 experiment.
- `report/skin_lesion_triage_report.tex`: LaTeX report source.
- `report/skin_lesion_triage_refs.bib`: BibTeX references.
- `report/skin_lesion_triage_report.pdf`: compiled report, if present.

## Experiment Summary

The notebook compares two pipelines:

- A classical HSV histogram baseline using class-conditional Gaussian Mixture Models.
- A deep EfficientNet-B0 classifier with MC Dropout for epistemic uncertainty estimation.

The binary task maps melanoma (`mel`) to the positive class and benign lesions (`nv`, `bkl`, `df`, `vasc`) to the negative class. Basal cell carcinoma (`bcc`) and actinic keratoses (`akiec`) are excluded because treating them as benign would be clinically unsafe.

Validation uses lesion-aware grouping with `lesion_id` so multiple images of the same physical lesion do not leak across train and validation splits.

## Kaggle Execution

1. Create a Kaggle notebook.
2. Attach dataset `kmader/skin-cancer-mnist-ham10000`.
3. Enable a GPU accelerator. The notebook is tuned for Kaggle GPU runs with a larger batch size, more DataLoader workers, persistent workers, pinned memory, prefetching, and mixed precision. On `GPU T4 x2`, it detects both devices but defaults to one-GPU training because `torch.nn.DataParallel` often underutilizes notebooks when image loading is the bottleneck.
4. Upload or copy `experiments/skin-lesion-triage.ipynb`.
5. Run the notebook top to bottom.

The notebook writes paper-ready outputs to `/kaggle/working/outputs/`:

- `training_history.csv`
- `summary_metrics.csv`
- `threshold_metrics.csv`
- `sensitivity_operating_points.csv`
- `triage_zone_metrics.csv`
- `validation_predictions.csv`
- `roc_curve_triage.png`
- `roc_curve_triage_calibrated.png`
- `uncertainty_scatter.png`
- `uncertainty_scatter_calibrated.png`
- `example_lesions_grid.png`

The final notebook cell also creates `/kaggle/working/outputs.zip` and displays a clickable download link for the whole output folder. Download the zip and copy its contents back into `experiments/outputs/` after the Kaggle run so the paper can be updated with observed metrics.

## Paper Build

From the `report/` directory:

```bash
pdflatex skin_lesion_triage_report.tex
bibtex skin_lesion_triage_report
pdflatex skin_lesion_triage_report.tex
pdflatex skin_lesion_triage_report.tex
```

`latexmk` can also be used if available:

```bash
latexmk -pdf skin_lesion_triage_report.tex
```

## Local Figure Generation

This repository includes a small `uv` project for regenerating paper figures from the Kaggle CSV outputs:

```bash
uv run python scripts/make_paper_figures.py
```

The script reads `experiments/outputs/training_history.csv` and `experiments/outputs/validation_predictions.csv`, then writes `training_history.png`, `calibration_probability_summary.png`, and `calibration_bins.csv` back to `experiments/outputs/`.

## Notes on Results

The current report uses the copied Kaggle outputs in `experiments/outputs/`. After rerunning the latest notebook version, refresh those files and rerun the local figure script before finalizing the report tables.
