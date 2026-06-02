"""End-to-end smoke test for the supplementary repo.

Loads the bundled U-Net checkpoint at `models/unet_resnet34_fold2.pt`,
runs inference on the bundled sample frames under `data_sample/`, and:
  - writes a colorized prediction mask to `data_sample/<rat>/Predicted Mask (color)/<stem>.png`
  - writes a blended overlay to `data_sample/<rat>/Predicted Overlay/<stem>.png`
  - computes per-class IoU vs the ground-truth mask in `data_sample/<rat>/Mask/`
  - saves a 3-panel headline figure at `data_sample/smoke_test_output.png`

By default it processes the first frame of every bundled rat. Pass `--all`
to iterate every frame.

Usage:
    python code/data_sample/smoke_test.py
    python code/data_sample/smoke_test.py --all
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HERE = Path(__file__).resolve().parent                       # python3/code/data_sample/
REPO = HERE.parent.parent                                    # python3/
UNET_PKG = REPO / 'code' / 'segmentation_unet'
sys.path.insert(0, str(UNET_PKG))

import config  # noqa: E402
from model import RatThermalSegmentor  # noqa: E402
from inference import _preprocess_csv  # noqa: E402

DATA_SAMPLE = REPO / 'data_sample'
SMOKE_FOLD = 2
CHECKPOINT = config.fold_model_path(SMOKE_FOLD)
HEADLINE_FIG = DATA_SAMPLE / 'smoke_test_output.png'

CLASS_COLORS = {
    1: (255,  60,  60),
    2: ( 60, 200,  60),
    3: ( 60, 120, 255),
}
ALPHA = 0.45


def colorize_pred(pred: np.ndarray) -> np.ndarray:
    color = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for cls, rgb in CLASS_COLORS.items():
        color[pred == cls] = rgb
    return color


def per_class_iou(gt: np.ndarray, pred: np.ndarray, n_classes: int = 4):
    ious = []
    for c in range(n_classes):
        g = gt == c
        p = pred == c
        inter = np.logical_and(g, p).sum()
        union = np.logical_or(g, p).sum()
        ious.append(float(inter) / union if union else float('nan'))
    return ious


def load_checkpoint():
    if not CHECKPOINT.exists():
        print(f'[smoke][FAIL] checkpoint missing: {CHECKPOINT}')
        print('              download from Figshare (see models/README.md)')
        return None
    model = RatThermalSegmentor().to(config.DEVICE)
    state = torch.load(str(CHECKPOINT), map_location=config.DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f'[smoke] device: {config.DEVICE}  |  checkpoint: {CHECKPOINT.name}')
    return model


def predict_frame(model, csv_path: Path, mask_path: Path, thermal_path: Path):
    tensor, _ = _preprocess_csv(str(csv_path))
    if tensor is None:
        return None, None
    pred = model.predict_and_clean(tensor)[0]
    gt = np.array(Image.open(mask_path).convert('L'))
    if gt.shape != pred.shape:
        gt_img = Image.fromarray(gt).resize((pred.shape[1], pred.shape[0]), Image.NEAREST)
        gt = np.array(gt_img)
    ious = per_class_iou(gt, pred, n_classes=config.NUM_CLASSES)
    return pred, ious


def write_mask_and_overlay(mask: np.ndarray, thermal_path: Path,
                           mask_dir: Path, overlay_dir: Path, stem: str) -> None:
    """Write a colorized class mask and a thermal+mask blended overlay."""
    color = colorize_pred(mask)
    mask_dir.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)
    Image.fromarray(color).save(mask_dir / f'{stem}.png')

    if thermal_path is not None and thermal_path.exists():
        thermal = np.array(Image.open(thermal_path).convert('RGB'), dtype=np.float32)
        if thermal.shape[:2] != color.shape[:2]:
            thermal_img = Image.fromarray(thermal.astype(np.uint8)).resize(
                (color.shape[1], color.shape[0]), Image.BILINEAR)
            thermal = np.array(thermal_img, dtype=np.float32)
        blend = thermal.copy()
        fg = mask > 0
        blend[fg] = (1 - ALPHA) * thermal[fg] + ALPHA * color[fg].astype(np.float32)
        Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8)).save(
            overlay_dir / f'{stem}.png')


def write_prediction_outputs(pred: np.ndarray, thermal_path: Path,
                             rat_dir: Path, stem: str) -> None:
    """Prediction visualizations -> Predicted Mask (color)/ and Predicted Overlay/."""
    write_mask_and_overlay(pred, thermal_path,
                           rat_dir / 'Predicted Mask (color)',
                           rat_dir / 'Predicted Overlay', stem)


def write_gt_outputs(mask_path: Path, thermal_path: Path,
                     rat_dir: Path, stem: str) -> None:
    """Ground-truth visualizations -> Ground Truth Mask (color)/ and Ground Truth Overlay/."""
    gt = np.array(Image.open(mask_path).convert('L'))
    write_mask_and_overlay(gt, thermal_path,
                           rat_dir / 'Ground Truth Mask (color)',
                           rat_dir / 'Ground Truth Overlay', stem)


def iter_frames(rat_dir: Path, all_frames: bool):
    csv_dir = rat_dir / 'CSV'
    mask_dir = rat_dir / 'Mask'
    thermal_dir = rat_dir / 'Thermal Imaging'
    csv_paths = sorted(csv_dir.glob('*_CSV.csv'))
    if not all_frames:
        csv_paths = csv_paths[:1]
    for csv_path in csv_paths:
        stem = csv_path.stem.replace('_CSV', '')
        mask_path = mask_dir / f'{stem}.png'
        if not mask_path.exists():
            continue
        thermal_path = next(thermal_dir.glob(f'{stem}.*'), None)
        yield csv_path, mask_path, thermal_path, stem


def headline_figure(model, rat_dir: Path) -> None:
    frames = list(iter_frames(rat_dir, all_frames=False))
    if not frames:
        return
    csv_path, mask_path, thermal_path, stem = frames[0]
    pred, ious = predict_frame(model, csv_path, mask_path, thermal_path)
    if pred is None:
        return
    miou = float(np.nanmean(ious))

    thermal = np.array(Image.open(thermal_path).convert('RGB')) if thermal_path else None
    gt = np.array(Image.open(mask_path).convert('L'))
    if thermal is not None and thermal.shape[:2] != pred.shape:
        thermal = np.array(Image.fromarray(thermal).resize(
            (pred.shape[1], pred.shape[0]), Image.BILINEAR))
    if gt.shape != pred.shape:
        gt = np.array(Image.fromarray(gt).resize(
            (pred.shape[1], pred.shape[0]), Image.NEAREST))

    cmap = mcolors.ListedColormap([(0, 0, 0, 0), (1, 0.2, 0.2, 0.6),
                                   (0.2, 0.8, 0.2, 0.6), (0.2, 0.5, 1.0, 0.6)])
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    if thermal is not None:
        ax[0].imshow(thermal); ax[1].imshow(thermal); ax[2].imshow(thermal)
    ax[0].set_title('Input'); ax[0].axis('off')
    ax[1].imshow(gt, cmap=cmap, vmin=0, vmax=3)
    ax[1].set_title('GT overlay'); ax[1].axis('off')
    ax[2].imshow(pred, cmap=cmap, vmin=0, vmax=3)
    ax[2].set_title(f'Predicted overlay (mIoU={miou:.3f})'); ax[2].axis('off')
    ax[3].imshow(colorize_pred(pred))
    ax[3].set_title('Prediction mask'); ax[3].axis('off')
    fig.suptitle(f'U-Net smoke test on {rat_dir.name} / {stem}')
    fig.tight_layout()
    fig.savefig(HEADLINE_FIG, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'[smoke] wrote {HEADLINE_FIG}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true',
                        help='Iterate every frame of every bundled rat (default: 1 frame per rat)')
    args = parser.parse_args()

    rat_dirs = sorted(DATA_SAMPLE.glob('Rat*'))
    if not rat_dirs:
        print(f'[smoke][FAIL] no Rat* under {DATA_SAMPLE}')
        return 1

    model = load_checkpoint()
    if model is None:
        return 2

    all_ious = []
    for rat_dir in rat_dirs:
        n = 0
        for csv_path, mask_path, thermal_path, stem in iter_frames(rat_dir, args.all):
            pred, ious = predict_frame(model, csv_path, mask_path, thermal_path)
            if pred is None:
                continue
            write_prediction_outputs(pred, thermal_path, rat_dir, stem)
            write_gt_outputs(mask_path, thermal_path, rat_dir, stem)
            all_ious.append(ious)
            n += 1
        print(f'[{rat_dir.name}] processed {n} frame(s)')

    if not all_ious:
        print('[smoke][FAIL] no frames processed')
        return 3

    arr = np.array(all_ious)
    per_class_mean = np.nanmean(arr, axis=0)
    miou = float(np.nanmean(per_class_mean))
    print('[smoke] mean per-class IoU across {} frame(s):'.format(arr.shape[0]))
    for name, iou in zip(config.CLASS_NAMES, per_class_mean):
        print(f'          {name:<10}: {iou:.4f}')
    print(f'[smoke] mean mIoU: {miou:.4f}')

    headline_figure(model, rat_dirs[0])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
