import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
from tqdm import tqdm

import config
from model import RatThermalSegmentor
from dataset import RatThermalDataset, get_train_transforms, get_val_transforms
from loss import WeightedCombinedDiceFocalLoss, calculate_metrics, compute_test_metrics
from plots import plot_training_progress


def _build_loss():
    weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(config.DEVICE)
    return WeightedCombinedDiceFocalLoss(class_weights=weights).to(config.DEVICE)


def _all_splits(df):
    """Return list of (df_rest, df_test) for all K_FOLDS folds (deterministic)."""
    groups = df['rat_unique_id'].values
    if config.STRATIFY_COL and config.STRATIFY_COL in df.columns:
        splitter = StratifiedGroupKFold(n_splits=config.K_FOLDS)
        y = df[config.STRATIFY_COL].values
    else:
        if config.STRATIFY_COL:
            print(f"[WARN] STRATIFY_COL='{config.STRATIFY_COL}' not in metadata; using GroupKFold.")
        splitter = GroupKFold(n_splits=config.K_FOLDS)
        y = groups
    return [(df.iloc[tv].reset_index(drop=True), df.iloc[te].reset_index(drop=True))
            for tv, te in splitter.split(df, y, groups)]


def _inner_split(df_rest):
    """Split into train (80%) and val (20%) at rat-group level."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(gss.split(df_rest, groups=df_rest['rat_unique_id']))
    return df_rest.iloc[train_idx].reset_index(drop=True), df_rest.iloc[val_idx].reset_index(drop=True)


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    batch_losses = []
    for images, masks in tqdm(loader, desc='Train', leave=False):
        images = images.to(config.DEVICE, dtype=torch.float32)
        masks = masks.to(config.DEVICE, dtype=torch.long)
        optimizer.zero_grad()
        loss = loss_fn(model(images), masks)
        loss.backward()
        optimizer.step()
        v = loss.item()
        batch_losses.append(v)
        running_loss += v * images.size(0)
    return running_loss / len(loader.dataset), batch_losses


def validate_one_epoch(model, loader, loss_fn):
    model.eval()
    running_loss = 0.0
    running_miou = 0.0
    running_per_class = torch.zeros(config.NUM_CLASSES).to(config.DEVICE)
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Val', leave=False):
            images = images.to(config.DEVICE, dtype=torch.float32)
            masks = masks.to(config.DEVICE, dtype=torch.long)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            miou, per_class = calculate_metrics(outputs, masks)
            n = images.size(0)
            running_loss += loss.item() * n
            running_miou += miou.item() * n
            running_per_class += per_class.to(config.DEVICE) * n
    n_total = len(loader.dataset)
    return (running_loss / n_total,
            running_miou / n_total,
            (running_per_class / n_total).cpu().tolist())


def _eval_test(model, df_test):
    dataset = RatThermalDataset(df_test, transforms=get_val_transforms())
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                        shuffle=False, num_workers=config.NUM_WORKERS)
    all_preds, all_targets = [], []
    for images, masks in tqdm(loader, desc='Test eval', leave=False):
        images = images.to(config.DEVICE, dtype=torch.float32)
        preds = model.predict_and_clean(images)
        all_preds.extend(preds.flatten())
        all_targets.extend(masks.numpy().flatten())
    ious, miou, _ = compute_test_metrics(np.array(all_targets), np.array(all_preds))
    return float(miou), ious.tolist()


def _train_fold(model, train_loader, val_loader, model_path):
    """Two-stage training; return history dict."""
    loss_fn = _build_loss()
    history = dict(train_loss=[], val_loss=[], val_miou=[], val_per_class=[], batch_losses=[])
    best_miou = -1.0

    def _run_stage(epochs, lr, freeze_encoder):
        nonlocal best_miou
        if freeze_encoder:
            for p in model.base_model.encoder.parameters():
                p.requires_grad = False
            params = filter(lambda p: p.requires_grad, model.parameters())
        else:
            for p in model.parameters():
                p.requires_grad = True
            params = model.parameters()

        optimizer = torch.optim.Adam(params, lr=lr)
        no_improve = 0
        min_val_loss = float('inf')

        for ep in range(epochs):
            ep_loss, bl = train_one_epoch(model, train_loader, optimizer, loss_fn)
            val_loss, val_miou, val_per_class = validate_one_epoch(model, val_loader, loss_fn)
            history['train_loss'].append(ep_loss)
            history['val_loss'].append(val_loss)
            history['val_miou'].append(val_miou)
            history['val_per_class'].append(val_per_class)
            history['batch_losses'].extend(bl)
            stage = 'S1' if freeze_encoder else 'S2'
            print(f'    {stage} ep{ep+1:02d}: loss={ep_loss:.4f}  val_miou={val_miou:.4f}')
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(), str(model_path))
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= config.PATIENCE:
                    print(f'    Early stop {stage}')
                    break

    print('  Stage 1 (decoder only)')
    _run_stage(config.EPOCHS_STAGE1, config.LR_STAGE1, freeze_encoder=True)

    print('  Stage 2 (full network)')
    if model_path.exists():
        model.load_state_dict(torch.load(str(model_path), map_location=config.DEVICE))
    _run_stage(config.EPOCHS_STAGE2, config.LR_STAGE2, freeze_encoder=False)

    return history


def _setup_dirs():
    metrics_dir   = config.OUTPUT_DIR / 'metrics'
    graphs_dir    = config.OUTPUT_DIR / 'graphs'
    test_rats_dir = config.OUTPUT_DIR / 'test_rats'
    for d in [config.MODELS_DIR, metrics_dir, graphs_dir, test_rats_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return metrics_dir, graphs_dir, test_rats_dir


def _process_fold(df, fold, df_rest, df_test, graphs_dir, test_rats_dir):
    """Train (or skip if cached) one fold. Returns a metrics DataFrame row."""
    model_path = config.fold_model_path(fold)
    rats_path  = test_rats_dir / f'test_rats_{fold}.json'

    test_rats = df_test['rat_unique_id'].unique().tolist()

    if model_path.exists():
        print(f'  [SKIP] {model_path.name} already exists — loading cached fold {fold}.')
        model = RatThermalSegmentor().to(config.DEVICE)
        model.load_state_dict(torch.load(str(model_path), map_location=config.DEVICE))
        test_miou, test_per_class = _eval_test(model, df_test)
        print(f'  Fold {fold} test mIoU: {test_miou:.4f}')
        # Return a single-row DataFrame so the summary still works
        return pd.DataFrame([{
            'fold': fold, 'epoch': 0,
            'train_loss': None, 'val_loss': None, 'val_miou': None,
            'test_miou_final': test_miou,
            **{f'val_iou_{n.lower()}': None for n in config.CLASS_NAMES},
        }])

    # Save test rat list (only when actually training)
    with open(rats_path, 'w') as f:
        json.dump(test_rats, f)

    df_train, df_val = _inner_split(df_rest)
    print(f'  Train rats: {sorted(df_train["rat_unique_id"].unique())}')
    print(f'  Val rats:   {sorted(df_val["rat_unique_id"].unique())}')
    print(f'  Test rats:  {test_rats}')
    print(f'  Sizes: train={len(df_train)}  val={len(df_val)}  test={len(df_test)}')

    train_loader = DataLoader(
        RatThermalDataset(df_train, get_train_transforms()),
        batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(
        RatThermalDataset(df_val, get_val_transforms()),
        batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    model = RatThermalSegmentor().to(config.DEVICE)
    history = _train_fold(model, train_loader, val_loader, model_path)

    plot_training_progress(
        history['batch_losses'], history['val_loss'],
        history['val_miou'], history['val_per_class'],
        save_path=graphs_dir / f'fold_{fold}_training.png',
        title=f'Fold {fold}')

    model.load_state_dict(torch.load(str(model_path), map_location=config.DEVICE))
    test_miou, test_per_class = _eval_test(model, df_test)
    print(f'  Fold {fold} test mIoU: {test_miou:.4f}')
    for name, iou in zip(config.CLASS_NAMES, test_per_class):
        print(f'    {name}: {iou:.4f}')

    per_class_arr = np.array(history['val_per_class'])
    return pd.DataFrame({
        'fold': fold,
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_miou': history['val_miou'],
        'test_miou_final': test_miou,
        **{f'val_iou_{n.lower()}': per_class_arr[:, i]
           for i, n in enumerate(config.CLASS_NAMES)},
    })


def run_single_fold(df, fold):
    """Train (or skip if cached) a single fold. fold is 1-indexed."""
    if fold < 1 or fold > config.K_FOLDS:
        print(f'[ERROR] fold must be 1-{config.K_FOLDS}')
        return
    metrics_dir, graphs_dir, test_rats_dir = _setup_dirs()
    splits = _all_splits(df)
    df_rest, df_test = splits[fold - 1]
    print(f'\n{"#"*40}\nFOLD {fold}/{config.K_FOLDS}\n{"#"*40}')
    fold_df = _process_fold(df, fold, df_rest, df_test, graphs_dir, test_rats_dir)

    # Merge with existing metrics CSV if present
    csv_path = metrics_dir / 'metrics_kfolds.csv'
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[existing['fold'] != fold]
        fold_df = pd.concat([existing, fold_df], ignore_index=True)
    fold_df.to_csv(csv_path, index=False)


def run_kfold_training(df):
    """Train all K_FOLDS folds, skipping any that already have a saved model."""
    metrics_dir, graphs_dir, test_rats_dir = _setup_dirs()
    splits = _all_splits(df)
    all_metrics = []

    for fold, (df_rest, df_test) in enumerate(splits, start=1):
        print(f'\n{"#"*40}\nFOLD {fold}/{config.K_FOLDS}\n{"#"*40}')
        fold_df = _process_fold(df, fold, df_rest, df_test, graphs_dir, test_rats_dir)
        all_metrics.append(fold_df)
        pd.concat(all_metrics).to_csv(metrics_dir / 'metrics_kfolds.csv', index=False)

    summary = pd.concat(all_metrics).groupby('fold')['test_miou_final'].first()
    print('\nPer-fold test mIoU:')
    for fold_num, miou in summary.items():
        print(f'  Fold {fold_num}: {miou:.4f}')
    print(f'  Mean: {summary.mean():.4f}')
