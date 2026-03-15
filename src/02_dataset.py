"""
02_dataset.py — PyTorch Dataset + augmentation pipeline.
Train/val split is based on TIFF file size — largest TIFFs go to train,
smallest go to val. This ensures val has representative but smaller coverage.
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
    DATA_PROCESSED_DIR, DATA_RAW_DIR, VAL_SPLIT,
    RANDOM_SEED, AUGMENT_TRAIN, BATCH_SIZE
)


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-15, 15), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32),
                        hole_width_range=(16, 32), p=0.2),
    ])


class GeoSegDataset(Dataset):
    def __init__(self, tile_ids, proc_dir, transform=None):
        self.tile_ids  = tile_ids
        self.img_dir   = proc_dir / "images"
        self.msk_dir   = proc_dir / "masks"
        self.transform = transform

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tid   = self.tile_ids[idx]
        image = np.load(self.img_dir / f"{tid}.npy")   # (C, H, W)
        mask  = np.load(self.msk_dir / f"{tid}.npy")   # (H, W)
        image_hwc = np.transpose(image, (1, 2, 0))
        if self.transform:
            aug       = self.transform(image=image_hwc, mask=mask)
            image_hwc = aug["image"]
            mask      = aug["mask"]
        image_tensor = torch.from_numpy(np.transpose(image_hwc, (2, 0, 1))).float()
        mask_tensor  = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor


def get_tiff_sizes(raw_dir):
    """
    Returns a dict of {tiff_filename: file_size_bytes}.
    Used to split train/val by size.
    """
    raw_dir = Path(raw_dir)
    sizes   = {}
    for tif_path in list(raw_dir.glob("**/*.tif")) + list(raw_dir.glob("**/*.tiff")):
        sizes[tif_path.name] = tif_path.stat().st_size
    return sizes


def split_by_size(sources, raw_dir, val_split):
    """
    Splits source TIFFs into train/val based on file size.
    Smallest TIFFs go to val until val_split fraction is reached by tile count.
    Falls back to random split if sizes can't be determined.
    """
    try:
        sizes = get_tiff_sizes(raw_dir)
        # Sort by size ascending — smallest first
        sorted_sources = sorted(sources, key=lambda s: sizes.get(s, 0))

        total    = len(sorted_sources)
        n_val    = max(1, round(total * val_split))
        val_sources   = sorted_sources[:n_val]    # smallest → val
        train_sources = sorted_sources[n_val:]    # largest  → train

        print(f"  Size-based split:")
        print(f"    Train ({len(train_sources)} TIFFs — largest):")
        for s in train_sources:
            mb = sizes.get(s, 0) / 1024**2
            print(f"      {s} ({mb:.0f} MB)")
        print(f"    Val ({len(val_sources)} TIFFs — smallest):")
        for s in val_sources:
            mb = sizes.get(s, 0) / 1024**2
            print(f"      {s} ({mb:.0f} MB)")

        return train_sources, val_sources

    except Exception as e:
        print(f"  [WARN] Size-based split failed ({e}) — falling back to random split")
        train_sources, val_sources = train_test_split(
            sources, test_size=val_split, random_state=RANDOM_SEED
        )
        return train_sources, val_sources


def build_dataloaders(proc_dir=None):
    if proc_dir is None:
        proc_dir = Path(DATA_PROCESSED_DIR)
    raw_dir = Path(DATA_RAW_DIR)

    meta_path = proc_dir / "tiles_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run 01_preprocess.py first. Expected: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    tiles   = meta["tiles"]
    print(f"Total tiles: {len(tiles)}")
    sources = list({t["source"] for t in tiles})

    if len(sources) == 1:
        print("  [INFO] Only one source TIFF — using random 80/20 tile split")
        all_ids = [t["id"] for t in tiles]
        train_ids, val_ids = train_test_split(
            all_ids, test_size=VAL_SPLIT, random_state=RANDOM_SEED
        )
    else:
        train_sources, val_sources = split_by_size(sources, raw_dir, VAL_SPLIT)
        train_ids = [t["id"] for t in tiles if t["source"] in train_sources]
        val_ids   = [t["id"] for t in tiles if t["source"] in val_sources]

    print(f"  Train tiles: {len(train_ids)},  Val tiles: {len(val_ids)}")

    train_ds = GeoSegDataset(train_ids, proc_dir,
                             transform=get_train_transforms() if AUGMENT_TRAIN else None)
    val_ds   = GeoSegDataset(val_ids, proc_dir, transform=None)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, val_loader, meta["num_bands"]