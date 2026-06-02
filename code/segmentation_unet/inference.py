import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from model import RatThermalSegmentor
from dataset import RatThermalDataset, get_val_transforms
from loss import compute_test_metrics, per_image_iou


def load_model(fold):
    """Load best saved model for the given fold (1-indexed)."""
    model_path = config.fold_model_path(fold)
    if not model_path.exists():
        print(f'[ERROR] Model not found: {model_path}')
        return None
    model = RatThermalSegmentor().to(config.DEVICE)
    model.load_state_dict(torch.load(str(model_path), map_location=config.DEVICE))
    model.eval()
    print(f'Loaded model: {model_path}')
    return model


def _preprocess_csv(csv_path):
    """Return (tensor (1,3,H,W), thermal_2d (H,W)) or (None, None) on error."""
    try:
        temp_df = pd.read_csv(csv_path, header=None, skiprows=2)
        temp = temp_df.iloc[:, 1:].values.astype(np.float32)
    except Exception as e:
        print(f'[ERROR] Failed to load CSV: {e}')
        return None, None

    if temp.shape[0] < temp.shape[1]:
        temp = cv2.rotate(temp, cv2.ROTATE_90_CLOCKWISE)

    image = np.clip((temp - 20.0) / (40.0 - 20.0), 0.0, 1.0)
    image_3c = np.stack([image, image, image], axis=-1).astype(np.float32)

    tfm = A.Compose([
        A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH, interpolation=cv2.INTER_NEAREST),
        ToTensorV2(),
    ])
    tensor = tfm(image=image_3c)['image'].unsqueeze(0).to(config.DEVICE, dtype=torch.float32)
    return tensor, image


def run_single_inference(fold, csv_path=None):
    """Run inference on one CSV and show a 3-panel figure."""
    if csv_path is None:
        try:
            from tkinter import Tk
            from tkinter.filedialog import askopenfilename
            Tk().withdraw()
            csv_path = askopenfilename(title='Select Thermal CSV', filetypes=[('CSV', '*.csv')])
        except Exception:
            csv_path = input('Enter path to thermal CSV: ').strip()
    if not csv_path:
        print('No file selected.')
        return

    model = load_model(fold)
    if model is None:
        return

    tensor, thermal_2d = _preprocess_csv(csv_path)
    if tensor is None:
        return

    mask_pred = model.predict_and_clean(tensor)[0]

    cmap = plt.cm.get_cmap('jet', config.NUM_CLASSES)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Inference: {os.path.basename(csv_path)}  (fold {fold} model)')

    axes[0].imshow(thermal_2d, cmap='inferno')
    axes[0].set_title('Thermal Input')
    axes[0].axis('off')

    im = axes[1].imshow(mask_pred, cmap=cmap, vmin=0, vmax=config.NUM_CLASSES - 1,
                        interpolation='nearest')
    axes[1].set_title('Prediction (Cleaned)')
    axes[1].axis('off')

    axes[2].imshow(thermal_2d, cmap='inferno')
    axes[2].imshow(mask_pred, cmap=cmap, vmin=0, vmax=config.NUM_CLASSES - 1,
                   alpha=0.5, interpolation='nearest')
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax, ticks=list(range(config.NUM_CLASSES)))
    cbar.ax.set_yticklabels(config.CLASS_NAMES)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def evaluate_fold(df, fold):
    """Evaluate the saved model for a fold on its held-out test set."""
    rats_path = config.OUTPUT_DIR / 'test_rats' / f'test_rats_{fold}.json'
    if not rats_path.exists():
        print(f'[ERROR] test_rats_{fold}.json not found for fold {fold}. Train first.')
        return

    with open(rats_path) as f:
        test_rats = json.load(f)

    df_test = df[df['rat_unique_id'].isin(test_rats)].reset_index(drop=True)
    if df_test.empty:
        print(f'[ERROR] No samples for test rats {test_rats}')
        return

    model = load_model(fold)
    if model is None:
        return

    loader = DataLoader(
        RatThermalDataset(df_test, get_val_transforms()),
        batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    repo_root = config.DATA_DIR.parent

    def _local_path(abspath):
        try:
            return str(Path(abspath).relative_to(repo_root))
        except ValueError:
            return str(abspath)

    all_preds, all_targets = [], []
    per_image_rows = []
    idx = 0
    for images, masks in tqdm(loader, desc=f'Evaluating fold {fold}'):
        images = images.to(config.DEVICE, dtype=torch.float32)
        preds = model.predict_and_clean(images)
        masks_np = masks.numpy()
        all_preds.extend(preds.flatten())
        all_targets.extend(masks_np.flatten())
        for b in range(preds.shape[0]):
            img_miou, per_class = per_image_iou(preds[b], masks_np[b], config.NUM_CLASSES)
            row = df_test.iloc[idx]
            frame_m = re.search(r'Thermal_(\d+)\.png', str(row['mask_abspath']))
            per_image_rows.append({
                'fold': fold,
                'rat': row['rat_unique_id'],
                'frame': int(frame_m.group(1)) if frame_m else -1,
                'image_miou': img_miou,
                **{f'iou_{c.lower()}': v for c, v in zip(config.CLASS_NAMES, per_class)},
                'csv_path': _local_path(row['csv_abspath']),
            })
            idx += 1

    ious, miou, cm = compute_test_metrics(np.array(all_targets), np.array(all_preds))

    print(f'\nFold {fold} Test Results  (n={len(df_test)}):')
    print(f'  mIoU: {miou:.4f}')
    for name, iou in zip(config.CLASS_NAMES, ious):
        print(f'  {name}: {iou:.4f}')

    metrics_dir = config.OUTPUT_DIR / 'metrics'
    graphs_dir  = config.OUTPUT_DIR / 'graphs'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'fold': fold,
        'mean_iou': miou,
        'per_class_iou': dict(zip(config.CLASS_NAMES, ious.tolist())),
        'confusion_matrix': cm.tolist(),
        'num_samples': len(df_test),
    }
    metrics_path = metrics_dir / f'test_metrics_{fold}.json'
    with open(metrics_path, 'w') as f:
        json.dump(report, f, indent=4)

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
                linewidths=0.5)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = graphs_dir / f'confusion_matrix_{fold}.png'
    plt.savefig(cm_path)
    plt.close()

    per_image_df = pd.DataFrame(per_image_rows)
    per_image_path = metrics_dir / f'per_image_iou_{fold}.csv'
    per_image_df.to_csv(per_image_path, index=False)
    print(f'  Saved {metrics_path.name}, {cm_path.name} and {per_image_path.name}')
    return per_image_df
