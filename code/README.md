# `code/` Source Code Directory

This directory contains the Python source code for the rat thermal imaging anatomical segmentation pipeline. It implements a U-Net model with a ResNet-34 encoder to segment radiometric thermal images into four classes: **Background**, **Head**, **Body**, and **Tail**.

---

## Directory Structure

```
code/
  ├── segmentation_unet.py       # Main command-line entry point & interactive menu
  ├── visualize_worst.py         # Diagnostic tool to render worst-performing frame panels
  ├── segmentation_unet/         # Package containing core pipeline modules
  │   ├── config.py              # Directory paths, model parameters, & hyperparameter settings
  │   ├── dataset.py             # Custom PyTorch Dataset & Albumentations data augmentation
  │   ├── model.py               # Model definition (U-Net + ResNet-34) & connected-components post-processing
  │   ├── loss.py                # Combined Dice + Focal Cross-Entropy loss
  │   ├── train.py               # Training and validation epoch loops & k-fold orchestrator
  │   ├── inference.py           # Preprocessing & inference functions for test evaluations
  │   ├── plots.py               # Plotting utilities for training curves
  │   └── aggregate.py           # Aggregates metrics & confusion matrices across folds
  └── data_sample/               # Smoke tests and utility scripts for the bundled data sample
      └── README.md              # Detailed documentation for data_sample code
```

---

## Modules Breakdown

### Main Scripts

*   **[segmentation_unet.py](segmentation_unet.py)**: The central interface for running the pipeline. If executed with no arguments, it displays an interactive CLI menu. Otherwise, it processes command-line flags to trigger training, evaluation, inference, or smoke tests.
*   **[visualize_worst.py](visualize_worst.py)**: A diagnostic tool that parses `segmentation_unet_runs/metrics/per_image_iou.csv`, identifies the top $N$ worst-performing frames (by test IoU), and generates a 5-panel diagnostic visualization for each:
    1.  Raw Radiometric Thermal Image (re-rendered using the `inferno` colormap)
    2.  Ground-Truth Segmentation Mask (colorized)
    3.  Predicted Segmentation Mask (colorized)
    4.  Ground-Truth Overlay (thermal image blended with ground-truth mask)
    5.  Predicted Overlay (thermal image blended with predicted mask)
    Visuals are saved to `segmentation_unet_runs/graphs/worst_visualizations/`.


*   **[data_sample/](data_sample/)**: Contains lightweight test scripts (`smoke_test.py` and `make_overlay.py`) designed specifically for offline smoke-testing against a small subset of the dataset. For more information, see [data_sample/README.md](data_sample/README.md).
