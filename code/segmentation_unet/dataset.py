import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    stream = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(stream, flags)


class RatThermalDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.csv_paths = self.df['csv_abspath'].values
        self.mask_paths = self.df['mask_abspath'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mask = imread_unicode(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)
        if mask is None:
            mask = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        try:
            temp_df = pd.read_csv(self.csv_paths[idx], header=None, skiprows=2)
            temp = temp_df.iloc[:, 1:].values.astype(np.float32)
        except Exception:
            temp = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.float32)

        # Standardize orientation: rotate each array independently if landscape
        if temp.shape[0] < temp.shape[1]:
            temp = cv2.rotate(temp, cv2.ROTATE_90_CLOCKWISE)
        if mask.shape[0] < mask.shape[1]:
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        # If shapes still differ, resize mask to match temp (nearest-neighbour to preserve labels)
        if mask.shape != temp.shape:
            mask = cv2.resize(mask, (temp.shape[1], temp.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        image = np.clip((temp - 20.0) / (40.0 - 20.0), 0.0, 1.0)
        image = np.stack([image, image, image], axis=-1).astype(np.float32)

        if self.transforms:
            aug = self.transforms(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']

        return image, mask.long()


def get_train_transforms():
    return A.Compose([
        A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=(-0.1, 0.1), scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH, interpolation=cv2.INTER_NEAREST),
        ToTensorV2(),
    ])
