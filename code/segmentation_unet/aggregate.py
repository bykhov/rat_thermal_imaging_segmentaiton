"""
Aggregate per-fold test metrics into a single summary + confusion matrix.

Reads each ``OUTPUT_DIR/metrics/test_metrics_<k>.json`` (written by
``inference.evaluate_fold``), sums the per-fold confusion matrices, and writes:
  - OUTPUT_DIR/metrics/aggregate_test_metrics.json
  - OUTPUT_DIR/metrics/summary_kfolds.csv          (one row per fold + mean/std/median)
  - OUTPUT_DIR/graphs/aggregate_confusion_matrix.png (+ .pdf, row-normalised heatmap)
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

CLASS_NAMES = config.CLASS_NAMES


def _load_fold(fold):
    path = config.OUTPUT_DIR / 'metrics' / f'test_metrics_{fold}.json'
    with open(path) as f:
        return json.load(f)


def _per_class_iou_from_cm(cm):
    cm = np.asarray(cm, dtype=np.int64)
    ious = np.zeros(cm.shape[0], dtype=np.float64)
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        ious[c] = tp / denom if denom > 0 else float('nan')
    return ious


def main(folds=None):
    if folds is None:
        folds = list(range(1, config.K_FOLDS + 1))

    metrics_dir = config.OUTPUT_DIR / 'metrics'
    graphs_dir  = config.OUTPUT_DIR / 'graphs'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    cm_agg = None

    print(f'\n{"Fold":>4} {"n":>5} {"mIoU":>8} {"  ".join(f"{c:>8}" for c in CLASS_NAMES)}')
    for k in folds:
        rep = _load_fold(k)
        cm = np.asarray(rep['confusion_matrix'], dtype=np.int64)
        cm_agg = cm if cm_agg is None else cm_agg + cm
        ious = [rep['per_class_iou'][c] for c in CLASS_NAMES]
        rows.append({'fold': k, 'n': rep['num_samples'], 'test_miou': rep['mean_iou'],
                     **{f'iou_{c.lower()}': v for c, v in zip(CLASS_NAMES, ious)}})
        per_class_str = '  '.join(f'{v:8.4f}' for v in ious)
        print(f'{k:>4} {rep["num_samples"]:>5} {rep["mean_iou"]:>8.4f} {per_class_str}')

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in ('fold', 'n')]
    means   = df[metric_cols].mean()
    stds    = df[metric_cols].std(ddof=1)
    medians = df[metric_cols].median()
    summary_mean   = {**{'fold': 'mean',   'n': df['n'].sum()}, **means.to_dict()}
    summary_std    = {**{'fold': 'std',    'n': 0},             **stds.to_dict()}
    summary_median = {**{'fold': 'median', 'n': 0},             **medians.to_dict()}
    df_out = pd.concat(
        [df, pd.DataFrame([summary_mean, summary_std, summary_median])],
        ignore_index=True)
    summary_path = metrics_dir / 'summary_kfolds.csv'
    df_out.to_csv(summary_path, index=False)
    print()
    print('Mean +/- std  (median) across folds:')
    for k, m in means.items():
        print(f'  {k:>12}: {m:.4f}  +/- {stds[k]:.4f}   (median {medians[k]:.4f})')

    # Aggregate per-class IoU from summed confusion matrix
    ious_agg = _per_class_iou_from_cm(cm_agg)
    miou_agg = float(np.nanmean(ious_agg))
    print()
    print('Aggregate (from summed confusion matrix):')
    print(f'  mIoU: {miou_agg:.4f}')
    for c, v in zip(CLASS_NAMES, ious_agg):
        print(f'  {c:>12}: {v:.4f}')

    report = {
        'mean_iou': miou_agg,
        'per_class_iou': dict(zip(CLASS_NAMES, ious_agg.tolist())),
        'confusion_matrix': cm_agg.tolist(),
        'total_pixels': int(cm_agg.sum()),
        'folds_aggregated': list(folds),
    }
    agg_path = metrics_dir / 'aggregate_test_metrics.json'
    with open(agg_path, 'w') as f:
        json.dump(report, f, indent=4)

    # --- Heatmap: row-normalised ---
    row_sums = cm_agg.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm_agg / row_sums, 0.0)
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
    })
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                vmin=0.0, vmax=1.0, linewidths=0.5,
                annot_kws={'fontsize': 10},
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_ylabel('True', fontsize=10)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.tick_params(axis='both', labelsize=10, pad=3)
    ax.set_aspect('equal')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    cm_png = graphs_dir / 'aggregate_confusion_matrix.png'
    fig.savefig(cm_png, dpi=200)
    fig.savefig(graphs_dir / 'aggregate_confusion_matrix.pdf')
    plt.close(fig)

    print(f'\nWrote: {agg_path}')
    print(f'Wrote: {cm_png}')
    print(f'Wrote: {summary_path}')


if __name__ == '__main__':
    main()
