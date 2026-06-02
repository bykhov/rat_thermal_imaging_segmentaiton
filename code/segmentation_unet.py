"""
Entry point for rat thermal segmentation (U-Net + ResNet34).

Interactive menu (default):
    python segmentation_unet.py

Non-interactive (command-line):
    python segmentation_unet.py --mode smoke
    python segmentation_unet.py --mode train-all
    python segmentation_unet.py --mode train-fold --fold 2
    python segmentation_unet.py --mode eval       --fold 2
    python segmentation_unet.py --mode infer      --fold 2 [--csv path/to/Thermal_1_CSV.csv]
"""
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()  # fallback if __file__ is undefined
sys.path.insert(0, os.path.join(_here, 'segmentation_unet'))

import numpy as np
import pandas as pd
import torch
#%%
import config


def _build_metadata():
    """Scan data_full/ and build a metadata DataFrame from matched triplets."""
    from pathlib import Path
    data_dir = Path(config.DATA_DIR)
    rows = []
    rat_dirs = [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith('Rat')]
    for rat_dir in sorted(rat_dirs, key=lambda p: int(p.name.replace('Rat', ''))):
        rat_id = rat_dir.name
        mask_dir = rat_dir / 'Mask'
        csv_dir  = rat_dir / 'CSV'
        masks = {int(p.stem.split('_')[1]): p for p in mask_dir.glob('Thermal_*.png')}
        csvs  = {int(p.stem.split('_')[1]): p for p in csv_dir.glob('Thermal_*_CSV.csv')}
        for n in sorted(masks.keys() & csvs.keys()):
            rows.append({
                'rat_unique_id': rat_id,
                'mask_abspath':  str(masks[n]),
                'csv_abspath':   str(csvs[n]),
            })
    return pd.DataFrame(rows)


def load_metadata():
    """Build the sample metadata in memory by scanning data_full/ (no CSV cache)."""
    print(f'[INFO] Building metadata by scanning {config.DATA_DIR} ...')
    df = _build_metadata()
    if df.empty:
        print(f'[ERROR] No matched mask/CSV pairs found under {config.DATA_DIR}')
    else:
        print(f'Loaded {len(df)} samples  ({df["rat_unique_id"].nunique()} rats)')
    return df


def run_smoke_test(df):
    """Run the full single-fold pipeline for one epoch per stage into a dedicated
    subdirectory, then assert the expected artifacts were produced.

    Returns True on success, False on failure. Results are kept under
    ``segmentation_unet_runs/smoke_test/`` for inspection (cleared on next run).
    """
    import shutil
    from train import run_single_fold
    from inference import evaluate_fold

    if df is None or df.empty:
        print('[smoke][FAIL] no metadata; is data_full/ populated?')
        return False

    rats = list(df['rat_unique_id'].unique())
    if len(rats) < 4:
        print('[smoke][FAIL] need at least 4 rats in data_full/ (got %d)' % len(rats))
        return False

    smoke_dir = config.OUTPUT_DIR / 'smoke_test'
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    # First 4 rats x first ~3 frames each. With K_FOLDS=2 and 4 rats, every fold
    # yields 2 test + 2 rest rats, so the inner GroupShuffleSplit (needs >=2
    # groups) succeeds. (3 rats can leave a fold with only 1 rest rat -> break.)
    frames_per_rat = 3
    df_small = pd.concat(
        [df[df['rat_unique_id'] == r].head(frames_per_rat) for r in rats[:4]]
    ).reset_index(drop=True)

    # Save and override config for a fast, isolated run; restored in finally.
    keys = ('OUTPUT_DIR', 'MODELS_DIR', 'EPOCHS_STAGE1', 'EPOCHS_STAGE2',
            'BATCH_SIZE', 'NUM_WORKERS', 'K_FOLDS', 'PATIENCE')
    saved = {k: getattr(config, k) for k in keys}
    config.OUTPUT_DIR = smoke_dir
    config.MODELS_DIR = smoke_dir / 'models'
    config.EPOCHS_STAGE1 = 1
    config.EPOCHS_STAGE2 = 1
    config.BATCH_SIZE = 2
    config.NUM_WORKERS = 0
    config.K_FOLDS = 2
    config.PATIENCE = 999

    print(f'\n=== SMOKE TEST (1 epoch/stage -> {smoke_dir}) ===')
    print(f'Device: {config.DEVICE}  |  {len(df_small)} frames from rats {rats[:4]}')
    ok = False
    try:
        run_single_fold(df_small, fold=1)
        evaluate_fold(df_small, fold=1)

        expected = [
            config.MODELS_DIR / 'unet_resnet34_fold1.pt',
            smoke_dir / 'test_rats' / 'test_rats_1.json',
            smoke_dir / 'metrics' / 'metrics_kfolds.csv',
            smoke_dir / 'metrics' / 'test_metrics_1.json',
            smoke_dir / 'graphs' / 'fold_1_training.png',
            smoke_dir / 'graphs' / 'confusion_matrix_1.png',
        ]
        missing = [p for p in expected if not p.exists()]
        if missing:
            print('[smoke][FAIL] missing artifacts:')
            for p in missing:
                print(f'    {p}')
        else:
            print('Artifacts produced:')
            for p in expected:
                print(f'    OK  {p.relative_to(smoke_dir.parent)}')
            ok = True
    finally:
        for k, v in saved.items():
            setattr(config, k, v)

    if ok:
        print(f'=== SMOKE TEST PASSED ===  (results kept in {smoke_dir})\n')
    else:
        print(f'=== SMOKE TEST FAILED ===  (see {smoke_dir})\n')
    return ok


def run_eval_all(df, top_n=5):
    """Full inference: evaluate every fold on its held-out test rats, then aggregate.

    Also writes a combined per-image IoU CSV (sorted worst-first) and prints the
    worst ``top_n`` test images.
    """
    from inference import evaluate_fold
    import aggregate

    if df is None or df.empty:
        print('[ERROR] no metadata; is data_full/ populated?')
        return
    per_image = []
    for fold in range(1, config.K_FOLDS + 1):
        print(f'\n{"#"*40}\nEVAL FOLD {fold}/{config.K_FOLDS}\n{"#"*40}')
        df_pi = evaluate_fold(df, fold)
        if df_pi is not None and not df_pi.empty:
            per_image.append(df_pi)
    aggregate.main()

    if per_image:
        combined = (pd.concat(per_image, ignore_index=True)
                    .sort_values('image_miou', ascending=True)
                    .reset_index(drop=True))
        out_path = config.OUTPUT_DIR / 'metrics' / 'per_image_iou.csv'
        combined.to_csv(out_path, index=False)
        print(f'\nWrote {out_path}  ({len(combined)} rows)')
        cols = ['fold', 'rat', 'frame', 'image_miou',
                'iou_background', 'iou_head', 'iou_body', 'iou_tail']
        pd.set_option('display.float_format', lambda x: f'{x:.4f}' if pd.notna(x) else 'nan')
        print(f'\n=== Worst {top_n} test images by per-image mIoU ===')
        print(combined.head(top_n)[cols].to_string(index=False))


def _ask_fold():
    return int(input(f'Enter fold number (1-{config.K_FOLDS}): ').strip())


def menu():
    print('\n' + '='*50)
    print('Rat Thermal Segmentation  (U-Net + ResNet34)')
    print('='*50)
    print('1. Smoke test')
    print('2. Train all folds  (skip already cached)')
    print('3. Train single fold  (skip if already cached)')
    print('4. Inference on saved model')
    print('5. Evaluate saved model on test set')
    print('6. Evaluate ALL folds  (full inference + aggregate)')
    print('7. Exit')
    print('='*50)
    return input('Select: ').strip()


def _run_menu():
    """Interactive loop. Metadata is loaded lazily the first time it is needed."""
    df = [None]  # boxed so the helper can populate it once

    def _df():
        if df[0] is None:
            df[0] = load_metadata()
        return df[0]

    while True:
        choice = menu()
        if choice == '1':
            run_smoke_test(_df())
        elif choice == '2':
            from train import run_kfold_training
            run_kfold_training(_df())
        elif choice == '3':
            from train import run_single_fold
            run_single_fold(_df(), _ask_fold())
        elif choice == '4':
            from inference import run_single_inference
            run_single_inference(_ask_fold())
        elif choice == '5':
            from inference import evaluate_fold
            evaluate_fold(_df(), _ask_fold())
        elif choice == '6':
            run_eval_all(_df())
        elif choice == '7':
            break
        else:
            print('Invalid option.')


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Rat thermal segmentation (U-Net + ResNet34). '
                    'Run with no arguments for the interactive menu.')
    parser.add_argument(
        '--mode', choices=['smoke', 'train-all', 'train-fold', 'infer', 'eval', 'eval-all'],
        help='Operation to run non-interactively. Omit for the interactive menu.')
    parser.add_argument(
        '--fold', type=int,
        help='Fold number (1-based). Required for train-fold, eval, and infer.')
    parser.add_argument(
        '--csv', help='Path to a thermal CSV (optional, for --mode infer).')
    args = parser.parse_args()

    # Default: no flags -> interactive menu.
    if args.mode is None:
        _run_menu()
        return 0

    def _need_fold():
        if args.fold is None:
            parser.error(f'--mode {args.mode} requires --fold N')
        return args.fold

    # Validate required args before doing any heavy work (metadata scan).
    if args.mode in ('train-fold', 'eval', 'infer'):
        fold = _need_fold()

    if args.mode == 'smoke':
        return 0 if run_smoke_test(load_metadata()) else 1

    if args.mode == 'eval-all':
        run_eval_all(load_metadata())
        return 0

    if args.mode == 'train-all':
        from train import run_kfold_training
        run_kfold_training(load_metadata())

    elif args.mode == 'train-fold':
        from train import run_single_fold
        run_single_fold(load_metadata(), fold)

    elif args.mode == 'infer':
        from inference import run_single_inference
        run_single_inference(fold, args.csv)

    elif args.mode == 'eval':
        from inference import evaluate_fold
        evaluate_fold(load_metadata(), fold)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
