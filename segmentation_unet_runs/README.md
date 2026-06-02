# `segmentation_unet_runs/` Evaluation Outputs

This directory stores execution logs, metrics, plots, and diagnostic visualizations generated during validation and error analysis of the U-Net segmentation pipeline.

---

## Directory Structure

```
segmentation_unet_runs/
  ├── eval_all.log                    # Logging output from the full 5-fold evaluation run
  ├── test_rats/                      # Log of subject allocations per fold
  │   └── test_rats_{1..5}.json       # Lists of rat IDs assigned to each fold's test set
  ├── metrics/                        # Quantitative results (IoU and pixel counts)
  │   ├── summary_kfolds.csv          # Mean, median, standard deviation, and per-fold metrics
  │   ├── aggregate_test_metrics.json # Cumulative mIoU, per-class IoU, and confusion matrix
  │   ├── test_metrics_{1..5}.json    # Individual fold-level metrics and confusion matrices
  │   ├── per_image_iou.csv           # Master CSV listing IoUs for every individual frame
  │   └── per_image_iou_{1..5}.csv    # Per-fold frame-level IoUs
  └── graphs/                         # Visual results and error analysis
      ├── aggregate_confusion_matrix.png # Overall confusion matrix (PNG format)
      ├── aggregate_confusion_matrix.pdf # Overall confusion matrix (PDF format for papers)
      ├── confusion_matrix_{1..5}.png # Confusion matrix heatmaps for each individual fold
      └── worst_visualizations/       # 5-panel failure mode plots for the worst N frames
          └── {rank}_{rat}_{frame}_fold{fold}.png
```
