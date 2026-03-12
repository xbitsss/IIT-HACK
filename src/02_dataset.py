"""
02_dataset.py
─────────────
PyTorch Dataset + augmentation pipeline for geospatial segmentation tiles.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROCESSED_DIR, TILE_SIZE, VAL_SPLIT,
    RANDOM_SEED, AUGMENT_TRAIN, BATCH_SIZE
)


# ── Augmentation Pipelines ───────────────────────────────────────────────────

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
    ])

def get_val_transforms():
    return None   # no augmentation for validation


# ── Dataset ──────────────────────────────────────────────────────────────────

class GeoSegDataset(Dataset):
    """
    Loads pre-tiled .npy image and mask arrays.
    Handles arbitrary number of input bands.
    """

    def __init__(self, tile_ids: list[str], proc_dir: Path, transform=None):
        self.tile_ids  = tile_ids
        self.img_dir   = proc_dir / "images"
        self.msk_dir   = proc_dir / "masks"
        self.transform = transform

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tid = self.tile_ids[idx]

        image = np.load(self.img_dir / f"{tid}.npy")   # (C, H, W) float32
        mask  = np.load(self.msk_dir / f"{tid}.npy")   # (H, W) uint8

        # Albumentations expects (H, W, C)
        image_hwc = np.transpose(image, (1, 2, 0))

        if self.transform:
            augmented = self.transform(image=image_hwc, mask=mask)
            image_hwc = augmented["image"]
            mask      = augmented["mask"]

        # Back to (C, H, W) tensor
        image_tensor = torch.from_numpy(np.transpose(image_hwc, (2, 0, 1))).float()
        mask_tensor  = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor


# ── Factory Functions ─────────────────────────────────────────────────────────

def build_dataloaders(proc_dir: Path = None):
    """
    Reads tile metadata, splits train/val, returns DataLoaders.
    Uses spatial-aware split: tiles from the same source TIFF stay together.
    """
    if proc_dir is None:
        proc_dir = Path(DATA_PROCESSED_DIR)

    meta_path = proc_dir / "tiles_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run 01_preprocess.py first. Expected: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    tiles = meta["tiles"]
    print(f"Total tiles: {len(tiles)}")

    # Spatial split: group by source TIFF, split at source level
    sources = list({t["source"] for t in tiles})
    if len(sources) == 1:
        # Only one TIFF — fall back to random tile split
        print("  [INFO] Single source TIFF — using random tile split")
        all_ids = [t["id"] for t in tiles]
        train_ids, val_ids = train_test_split(
            all_ids, test_size=VAL_SPLIT, random_state=RANDOM_SEED
        )
    else:
        # Multiple TIFFs — hold out entire TIFFs for validation
        train_sources, val_sources = train_test_split(
            sources, test_size=VAL_SPLIT, random_state=RANDOM_SEED
        )
        train_ids = [t["id"] for t in tiles if t["source"] in train_sources]
        val_ids   = [t["id"] for t in tiles if t["source"] in val_sources]
        print(f"  Train TIFFs: {train_sources}")
        print(f"  Val TIFFs:   {val_sources}")

    print(f"  Train tiles: {len(train_ids)},  Val tiles: {len(val_ids)}")

    train_ds = GeoSegDataset(
        train_ids, proc_dir,
        transform=get_train_transforms() if AUGMENT_TRAIN else None
    )
    val_ds = GeoSegDataset(val_ids, proc_dir, transform=None)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, meta["num_bands"]


if __name__ == "__main__":
    train_loader, val_loader, num_bands = build_dataloaders()
    batch = next(iter(train_loader))
    imgs, masks = batch
    print(f"Image batch: {imgs.shape}, dtype: {imgs.dtype}")
    print(f"Mask  batch: {masks.shape}, dtype: {masks.dtype}")
    print(f"Unique mask values: {masks.unique()}")
