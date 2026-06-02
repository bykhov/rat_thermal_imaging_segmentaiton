import torch
import torch.nn as nn
import numpy as np
import cv2
import segmentation_models_pytorch as smp


class RatThermalSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=4,
        )

    def forward(self, x):
        return self.base_model(x)

    def _keep_largest_blob(self, mask_pred):
        binary = (mask_pred > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return mask_pred
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned = np.zeros_like(mask_pred)
        cleaned[labels == largest] = mask_pred[labels == largest]
        return cleaned

    def predict_and_clean(self, x):
        self.eval()
        with torch.no_grad():
            preds = torch.argmax(self.forward(x), dim=1).cpu().numpy()
        return np.array([self._keep_largest_blob(preds[i]) for i in range(preds.shape[0])])
