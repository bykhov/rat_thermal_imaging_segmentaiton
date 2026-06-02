import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.metrics.functional import get_stats, iou_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import config


class WeightedCombinedDiceFocalLoss(nn.Module):
    def __init__(self, class_weights=None, dice_weight=0.5, focal_weight=0.5, gamma=2.0, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.class_weights = class_weights
        self.smooth = smooth
        self.gamma = gamma

    def forward(self, outputs, targets):
        num_classes = outputs.size(1)
        log_probs = F.log_softmax(outputs, dim=1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Focal component
        pt = (probs * targets_one_hot).sum(dim=1)
        log_pt = (log_probs * targets_one_hot).sum(dim=1)
        loss_per_pixel = -(1 - pt).pow(self.gamma) * log_pt
        if self.class_weights is not None:
            w = self.class_weights.to(outputs.device)
            loss_per_pixel = loss_per_pixel * w[targets]
        focal_loss = loss_per_pixel.mean()

        # Dice component
        probs_s = F.softmax(outputs, dim=1)
        dice_loss = 0.0
        total_w = 0.0
        for c in range(num_classes):
            p_c = probs_s[:, c]
            t_c = targets_one_hot[:, c]
            intersection = (p_c * t_c).sum(dim=(1, 2))
            union = p_c.sum(dim=(1, 2)) + t_c.sum(dim=(1, 2))
            loss_c = 1.0 - ((2.0 * intersection + self.smooth) / (union + self.smooth)).mean()
            wc = self.class_weights[c].item() if self.class_weights is not None else 1.0
            dice_loss += loss_c * wc
            total_w += wc
        dice_loss /= total_w

        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


def calculate_metrics(outputs, targets):
    """Batch-level mIoU and per-class IoU from raw logits (used during validation)."""
    preds = torch.argmax(outputs, dim=1)
    tp, fp, fn, tn = get_stats(preds, targets, mode='multiclass', num_classes=config.NUM_CLASSES)
    miou = iou_score(tp, fp, fn, tn, reduction='macro')
    per_class = iou_score(tp, fp, fn, tn, reduction='none').mean(dim=0)
    return miou, per_class


def per_image_iou(pred, mask, num_classes):
    """Return (mIoU over classes present in pred or GT, per-class IoU with NaN for absent)."""
    ious = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        p = (pred == c)
        g = (mask == c)
        if not g.any() and not p.any():
            continue
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        ious[c] = inter / union if union > 0 else 0.0
    return float(np.nanmean(ious)), ious


def compute_test_metrics(all_targets, all_preds):
    """Per-class IoU, mIoU and confusion matrix from flat numpy arrays."""
    cm = sk_confusion_matrix(all_targets, all_preds, labels=range(config.NUM_CLASSES))
    ious = []
    for i in range(config.NUM_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        ious.append(float(tp) / (tp + fp + fn + 1e-6))
    return np.array(ious), float(np.mean(ious)), cm
