"""
02_dataset.py — PyTorch Dataset + augmentation + class-balanced sampling.

Key additions vs original:
  • WeightedRandomSampler: tiles containing rare classes (road, water) are
    sampled more often.  Weights are derived from per-tile class_ids saved
    during preprocessing — no extra mask I/O at init time.
  • Stronger augmentation: elastic deformation, grid distortion, optical
    distortion, CLAHE, channel shuffle — these are well-validated for
    geospatial / aerial-image segmentation.
  • Val split remains file-size-based (larger TIFFs → train, smaller → val)
    to maximise train data while keeping val representative.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from pathlib import Path
import albumentations as A
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROCESSED_DIR, DATA_RAW_DIR, VAL_SPLIT,
    RANDOM_SEED, AUGMENT_TRAIN, BATCH_SIZE,
    USE_WEIGHTED_SAMPLER, SAMPLER_CLASS_WEIGHTS,
    NUM_CLASSES, REPLAY_RATIO,
)


# ── Augmentation ──────────────────────────────────────────────────────────────

def get_train_transforms():
    """
    Strong augmentation pipeline for aerial / satellite image segmentation.
    Each transform is individually tuned to avoid destroying geospatial signals:
    - No normalisation here (done in preprocessing)
    - Elastic / grid / optical distortions mimic real-world warp variation
    - CLAHE improves model robustness to contrast differences across scenes
    - CoarseDropout simulates clouds, buildings blocking parts of the image
    """
    return A.Compose([
        # ── Geometry ───────────────────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=0.06,
            rotate=(-20, 20),
            shear=(-5, 5),
            p=0.5,
        ),
        # Elastic / grid distortions — simulate sensor/terrain warp
        A.ElasticTransform(alpha=60, sigma=6, p=0.25),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.25),
        A.OpticalDistortion(distort_limit=0.15, shift_limit=0.1, p=0.2),

        # ── Radiometric ────────────────────────────────────────────────────────
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        # CLAHE: equalises local contrast — helps road edges in low-contrast areas
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),

        # ── Regularisation / occlusion ─────────────────────────────────────────
        # CoarseDropout: simulates missing data (clouds, sensor gaps, occlusion)
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(16, 48),
            hole_width_range=(16, 48),
            p=0.3,
        ),
        # RandomShadow: simulates cloud/building shadows — common in aerial data
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3), p=0.2),
    ])


def get_val_transforms():
    """No augmentation for validation — deterministic evaluation."""
    return None


# ── Dataset ───────────────────────────────────────────────────────────────────

class GeoSegDataset(Dataset):
    """
    img_dir / msk_dir: used when tile files are in a single directory (current shard).
    img_map / msk_map: used when tile files are scattered across shard subdirs (replay).
    Pass exactly one of the two forms.
    """
    def __init__(self, tile_ids: list, proc_dir: Path = None,
                 transform=None,
                 img_map: dict = None, msk_map: dict = None):
        self.tile_ids  = tile_ids
        self.img_dir   = proc_dir / "images" if proc_dir else None
        self.msk_dir   = proc_dir / "masks"  if proc_dir else None
        self.img_map   = img_map   # {tile_id: Path}  — for replay tiles
        self.msk_map   = msk_map
        self.transform = transform

    def _img_path(self, tid):
        if self.img_map and tid in self.img_map:
            return self.img_map[tid]
        return self.img_dir / f"{tid}.npy"

    def _msk_path(self, tid):
        if self.msk_map and tid in self.msk_map:
            return self.msk_map[tid]
        return self.msk_dir / f"{tid}.npy"

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tid   = self.tile_ids[idx]
        image = np.load(self._img_path(tid))   # (C, H, W) float32
        mask  = np.load(self._msk_path(tid))   # (H, W) uint8

        image_hwc = np.transpose(image, (1, 2, 0))   # (H, W, C)
        if self.transform:
            n_channels = image_hwc.shape[2]
            # Albumentations radiometric transforms (HueSaturationValue,
            # RandomShadow, CLAHE) only accept 1- or 3-channel images.
            # When >3 bands, augment first 3 and carry extras unchanged.
            if n_channels > 3:
                rgb_hwc   = image_hwc[:, :, :3]
                extra_hwc = image_hwc[:, :, 3:]   # (H, W, C-3)
                aug       = self.transform(image=rgb_hwc, mask=mask)
                image_hwc = np.concatenate([aug["image"], extra_hwc], axis=2)
            else:
                aug       = self.transform(image=image_hwc, mask=mask)
                image_hwc = aug["image"]
            mask = aug["mask"]

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(image_hwc, (2, 0, 1)))
        ).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return image_tensor, mask_tensor


# ── Class-balanced sampling ───────────────────────────────────────────────────

def compute_sample_weights(tile_meta_list: list) -> list:
    """
    Assign a sampling weight to each tile based on which classes it contains.
    A tile's weight = max weight of its contained classes.
    Tiles with only background get the lowest weight.
    Tiles with rare classes (road, water) get boosted.

    Weights come from SAMPLER_CLASS_WEIGHTS in config — tune there.
    """
    weights = []
    for tile in tile_meta_list:
        class_ids = tile.get("class_ids", [0])
        w = max(SAMPLER_CLASS_WEIGHTS.get(cid, 1.0) for cid in class_ids)
        weights.append(w)
    return weights


# ── Train / Val split ─────────────────────────────────────────────────────────

def get_tiff_sizes(raw_dir: Path) -> dict:
    sizes = {}
    for tif_path in list(raw_dir.glob("**/*.tif")) + list(raw_dir.glob("**/*.tiff")):
        sizes[tif_path.name] = tif_path.stat().st_size
    return sizes


def split_by_size(sources: list, raw_dir: Path, val_split: float):
    """
    Put the *smallest* TIFFs in val, largest in train.
    Rationale: smaller TIFFs are faster to evaluate; larger ones give the model
    more diverse training tiles.  Geographic split avoids data leakage.
    """
    try:
        sizes = get_tiff_sizes(raw_dir)
        sorted_sources = sorted(sources, key=lambda s: sizes.get(s, 0))

        total         = len(sorted_sources)
        n_val         = max(1, round(total * val_split))
        val_sources   = sorted_sources[:n_val]
        train_sources = sorted_sources[n_val:]

        print(f"  Size-based geographic split:")
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
        print(f"  [WARN] Size-based split failed ({e}) — falling back to random")
        train_sources, val_sources = train_test_split(
            sources, test_size=val_split, random_state=RANDOM_SEED
        )
        return train_sources, val_sources


# ── DataLoaders ───────────────────────────────────────────────────────────────

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
    print(f"Total tiles loaded from metadata: {len(tiles)}")

    sources = list({t["source"] for t in tiles})

    if len(sources) == 1:
        print("  [INFO] Only one source TIFF — using random 80/20 tile split")
        all_ids  = [t["id"]   for t in tiles]
        all_meta = tiles
        train_ids, val_ids, train_meta, _ = _random_split(all_ids, all_meta)
    else:
        train_sources, val_sources = split_by_size(sources, raw_dir, VAL_SPLIT)
        train_meta = [t for t in tiles if t["source"] in train_sources]
        val_meta   = [t for t in tiles if t["source"] in val_sources]
        train_ids  = [t["id"] for t in train_meta]
        val_ids    = [t["id"] for t in val_meta]

    print(f"  Current shard — Train: {len(train_ids)}, Val: {len(val_ids)}")

    # ── Load replay buffer from previous shards ───────────────────────────────
    from replay_buffer import load_all_replay, replay_exists
    replay_tiles, replay_img_map, replay_msk_map = [], {}, {}
    if replay_exists():
        replay_tiles, replay_img_map, replay_msk_map = load_all_replay()

    # ── Build datasets ────────────────────────────────────────────────────────
    train_transform = get_train_transforms() if AUGMENT_TRAIN else None

    current_ds = GeoSegDataset(train_ids, proc_dir, transform=train_transform)

    if replay_tiles:
        # Replay tiles get the SAME augmentation — they need to stay hard examples
        replay_ids = [t["id"] for t in replay_tiles]
        replay_ds  = GeoSegDataset(
            replay_ids, proc_dir=None,
            transform=train_transform,
            img_map=replay_img_map, msk_map=replay_msk_map,
        )

        # How many replay tiles to include: REPLAY_RATIO fraction of total
        n_current = len(train_ids)
        n_replay  = len(replay_ids)
        # Weight each replay tile so it appears REPLAY_RATIO fraction of the time
        # relative to current shard tiles
        replay_scale = (n_current * REPLAY_RATIO) / max(n_replay * (1 - REPLAY_RATIO), 1)
        replay_scale = min(replay_scale, 5.0)   # cap boost to avoid drowning current data

        print(f"  Replay mix — {n_replay} replay tiles, scale={replay_scale:.2f}x "
              f"(target {REPLAY_RATIO*100:.0f}% of batches from replay)")

        # Build combined sample weights: current shard = 1.0 × class weight,
        # replay tiles = replay_scale × class weight
        current_weights = compute_sample_weights(train_meta)
        replay_weights  = [
            replay_scale * SAMPLER_CLASS_WEIGHTS.get(
                max(t.get("class_ids", [0]),
                    key=lambda c: SAMPLER_CLASS_WEIGHTS.get(c, 0.0)), 1.0
            )
            for t in replay_tiles
        ]
        all_weights = current_weights + replay_weights

        combined_ds = ConcatDataset([current_ds, replay_ds])
        sampler = WeightedRandomSampler(
            weights=all_weights,
            num_samples=len(all_weights),
            replacement=True,
        )
        print(f"  WeightedRandomSampler over {len(combined_ds)} tiles "
              f"(current + replay)")
        train_loader = DataLoader(
            combined_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True,
        )

    elif USE_WEIGHTED_SAMPLER and train_meta:
        sample_weights = compute_sample_weights(train_meta)
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        print(f"  WeightedRandomSampler (no replay yet)")
        train_loader = DataLoader(
            current_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            current_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )

    val_ds = GeoSegDataset(val_ids, proc_dir, transform=None)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    return train_loader, val_loader, meta["num_bands"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_split(all_ids, all_meta):
    indices = list(range(len(all_ids)))
    train_idx, val_idx = train_test_split(
        indices, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )
    return (
        [all_ids[i] for i in train_idx],
        [all_ids[i] for i in val_idx],
        [all_meta[i] for i in train_idx],
        [all_meta[i] for i in val_idx],
    )