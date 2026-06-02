from pathlib import Path
import torch

try:
    _HERE = Path(__file__).resolve().parent          # python4/code/segmentation_unet/
except NameError:
    _HERE = Path.cwd()                               # fallback if __file__ is undefined
_REPO = _HERE.parent.parent                          # python4/

DATA_DIR = _REPO / 'data_full'
OUTPUT_DIR = _REPO / 'segmentation_unet_runs'
MODELS_DIR = _REPO / 'models'


def fold_model_path(fold):
    """Path to the numbered per-fold checkpoint in the main models/ directory."""
    return MODELS_DIR / f'unet_resnet34_fold{fold}.pt'

IMG_HEIGHT = 640
IMG_WIDTH = 480
NUM_CLASSES = 4
CLASS_NAMES = ['Background', 'Head', 'Body', 'Tail']

BATCH_SIZE = 8
NUM_WORKERS = 0

K_FOLDS = 5
# Set to a metadata column name to use StratifiedGroupKFold (e.g. 'paradigm').
# None uses plain GroupKFold.
STRATIFY_COL = None

LR_STAGE1 = 1e-3   # decoder-only stage
LR_STAGE2 = 1e-4   # full-network fine-tune
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 15
PATIENCE = 5

# Inverse-square-root class weights from dataset analysis
CLASS_WEIGHTS = [0.2186, 1.1523, 0.6859, 1.9432]  # Background, Head, Body, Tail

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
