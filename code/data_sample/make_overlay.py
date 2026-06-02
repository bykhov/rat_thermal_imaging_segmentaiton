"""Regenerate Ground Truth Overlay/ and Ground Truth Mask (color)/ PNGs for the bundled sample rats.

Reads `<rat_dir>/Thermal Imaging/*.png|jpg` and `<rat_dir>/Mask/*.png`,
blends them with a 4-class colormap (Background transparent,
Head=red, Body=green, Tail=blue), and writes results to:
  `<rat_dir>/Ground Truth Overlay/<name>.png`        thermal blended with mask
  `<rat_dir>/Ground Truth Mask (color)/<name>.png`   colorized mask on black background

Default targets are every `Rat*` directory under `python3/data_sample/`.

Usage:
    python code/data_sample/make_overlay.py                    # all bundled rats
    python code/data_sample/make_overlay.py <rat_dir> [...]    # specific dirs
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent                # python3/code/data_sample/
DEFAULT_SAMPLE_ROOT = HERE.parent.parent / 'data_sample'

CLASS_COLORS = {
    1: (255,  60,  60),
    2: ( 60, 200,  60),
    3: ( 60, 120, 255),
}
ALPHA = 0.45


def colorize_mask(mask_path: Path, out_path: Path) -> None:
    mask = np.array(Image.open(mask_path).convert('L'))
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, rgb in CLASS_COLORS.items():
        color[mask == cls] = rgb
    Image.fromarray(color).save(out_path)


def overlay(thermal_path: Path, mask_path: Path, out_path: Path) -> None:
    thermal = np.array(Image.open(thermal_path).convert('RGB'), dtype=np.float32)
    mask = np.array(Image.open(mask_path).convert('L'))
    color = np.zeros_like(thermal)
    for cls, rgb in CLASS_COLORS.items():
        color[mask == cls] = rgb
    blend = thermal.copy()
    fg = mask > 0
    blend[fg] = (1 - ALPHA) * thermal[fg] + ALPHA * color[fg]
    Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8)).save(out_path)


def process_rat(rat_dir: Path) -> int:
    thermal_dir = rat_dir / 'Thermal Imaging'
    mask_dir = rat_dir / 'Mask'
    out_dir = rat_dir / 'Ground Truth Overlay'
    vis_dir = rat_dir / 'Ground Truth Mask (color)'
    out_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)
    n = 0
    for tpath in sorted(list(thermal_dir.glob('*.png')) + list(thermal_dir.glob('*.jpg'))):
        mpath = mask_dir / (tpath.stem + '.png')
        if not mpath.exists():
            continue
        overlay(tpath, mpath, out_dir / (tpath.stem + '.png'))
        colorize_mask(mpath, vis_dir / (tpath.stem + '.png'))
        n += 1
    return n


if __name__ == '__main__':
    if len(sys.argv) > 1:
        rat_dirs = [Path(a) for a in sys.argv[1:]]
    else:
        rat_dirs = sorted(DEFAULT_SAMPLE_ROOT.glob('Rat*'))
    if not rat_dirs:
        print(f'[ERROR] no rat directories found under {DEFAULT_SAMPLE_ROOT}')
        raise SystemExit(1)
    for rd in rat_dirs:
        n = process_rat(rd)
        print(f'[{rd.name}] wrote {n} overlays')
