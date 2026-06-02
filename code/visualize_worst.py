"""Generate human-readable thermal/GT/pred/overlay panels for the worst N test images.

Reads the combined per-image IoU CSV produced by `segmentation_unet.py --mode eval-all`
(`segmentation_unet_runs/metrics/per_image_iou.csv`) and renders a 5-panel figure per
image for the worst N. Frame paths are derived locally from `config.DATA_DIR` + rat/frame.

Usage:
    python code/visualize_worst.py        # worst 10
    python code/visualize_worst.py 20      # worst 20
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()  # fallback if __file__ is undefined
sys.path.insert(0, os.path.join(_here, 'segmentation_unet'))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import config  # noqa: E402
from model import RatThermalSegmentor  # noqa: E402
from inference import _preprocess_csv  # noqa: E402

CLASS_COLORS = np.array([
    [0, 0, 0],          # 0 background - black
    [220, 60, 60],      # 1 head       - red
    [60, 180, 80],      # 2 body       - green
    [60, 120, 220],     # 3 tail       - blue
], dtype=np.uint8)

OUT_DIR = config.OUTPUT_DIR / 'graphs' / 'worst_visualizations'


def colorize(mask):
    return CLASS_COLORS[mask]


def load_mask(path, h, w):
    arr = np.fromfile(path, dtype=np.uint8)
    m = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if m.ndim == 3:
        m = m[..., 0]
    if (m.shape[0], m.shape[1]) != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.int64)


def panel(thermal, gt_mask, pred_mask, miou, title, save_path):
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))
    fig.suptitle(title, fontsize=11)

    axes[0].imshow(thermal, cmap='inferno')
    axes[0].set_title('Thermal'); axes[0].axis('off')

    axes[1].imshow(colorize(gt_mask))
    axes[1].set_title('Ground-truth mask'); axes[1].axis('off')

    axes[2].imshow(thermal, cmap='inferno')
    axes[2].imshow(colorize(gt_mask), alpha=0.45)
    axes[2].set_title('Ground-truth overlay'); axes[2].axis('off')

    axes[3].imshow(colorize(pred_mask))
    axes[3].set_title(f'Predicted mask  (mIoU {miou:.3f})'); axes[3].axis('off')

    axes[4].imshow(thermal, cmap='inferno')
    axes[4].imshow(colorize(pred_mask), alpha=0.45)
    axes[4].set_title('Predicted overlay'); axes[4].axis('off')

    handles = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[i] / 255.0)
               for i in range(len(config.CLASS_NAMES))]
    fig.legend(handles, config.CLASS_NAMES, loc='lower center',
               ncol=len(config.CLASS_NAMES), bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main(top_n=10):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.OUTPUT_DIR / 'metrics' / 'per_image_iou.csv'
    if not csv_path.exists():
        print(f'[ERROR] {csv_path} not found. Run "segmentation_unet.py --mode eval-all" first.')
        return
    df = pd.read_csv(csv_path).sort_values('image_miou').head(top_n).reset_index(drop=True)

    models = {}
    for fold in sorted(df['fold'].unique()):
        mp = config.fold_model_path(int(fold))
        m = RatThermalSegmentor().to(config.DEVICE)
        m.load_state_dict(torch.load(str(mp), map_location=config.DEVICE))
        m.eval()
        models[int(fold)] = m

    for i, row in df.iterrows():
        rat = row['rat']
        frame = int(row['frame'])
        fold = int(row['fold'])
        # Frame paths derived locally from data_full + rat/frame.
        csv_abs = config.DATA_DIR / rat / 'CSV' / f'Thermal_{frame}_CSV.csv'
        mask_abs = config.DATA_DIR / rat / 'Mask' / f'Thermal_{frame}.png'

        tensor, thermal_2d = _preprocess_csv(str(csv_abs))
        if tensor is None:
            print(f'[WARN] skipped {rat} frame {frame}: CSV load failed')
            continue
        with torch.no_grad():
            pred = models[fold].predict_and_clean(tensor)[0]
        gt = load_mask(str(mask_abs), pred.shape[0], pred.shape[1])
        therm_disp = cv2.resize(thermal_2d, (pred.shape[1], pred.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

        rank = i + 1
        title = (f'#{rank}  fold {fold}  {rat} frame {frame}   '
                 f'per-image mIoU = {row["image_miou"]:.3f}')
        out_path = OUT_DIR / f'{rank:02d}_{rat}_frame{frame}_fold{fold}.png'
        panel(therm_disp, gt, pred, row['image_miou'], title, out_path)
        print(f'  wrote {out_path}')

    print(f'\n{top_n} panels saved to {OUT_DIR}')


if __name__ == '__main__':
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(top_n=top_n)
