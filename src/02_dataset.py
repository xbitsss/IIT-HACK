"""
02_dataset.py — PyTorch Dataset + augmentation + class-balanced sampling.

Replay buffer is included: when training on shard N, tiles from shards
1…N-1 are mixed in via WeightedRandomSampler to prevent catastrophic
forgetting. The mix ratio is REPLAY_RATIO (default 25% of each batch).
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from pathlib import Path
import albumentations as A
import sys

sys.path.insert(0, str(Path(__file__).parent))

# ImageNet mean/std applied AFTER augmentation so albumentations always
# receives clean [0,1] data. Shape (3,1,1) for CHW broadcasting.
_IN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IN_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

def _imagenet_normalize(image_chw: np.ndarray) -> np.ndarray:
    """Apply ImageNet mean/std to the first 3 channels. Band 3 (NIR) left in [0,1]."""
    n = min(3, image_chw.shape[0])
    image_chw = image_chw.copy()
    image_chw[:n] = (image_chw[:n] - _IN_MEAN[:n]) / _IN_STD[:n]
    return image_chw

def _worker_init_fn(worker_id):
    """
    1. SIGINT isolation — Ctrl+C handled only by main process, not workers.
    2. Per-worker per-epoch RNG seeding from torch.initial_seed() so
       albumentations augmentation is genuinely random across epochs.
       Without this all workers inherit the same frozen numpy state from
       the parent process and apply identical augmentations every epoch.
    """
    import signal as _signal
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    import random as _random
    _random.seed(seed)

from config import (
    DATA_PROCESSED_DIR, DATA_RAW_DIR, VAL_SPLIT,
    RANDOM_SEED, AUGMENT_TRAIN, BATCH_SIZE,
    USE_WEIGHTED_SAMPLER, SAMPLER_CLASS_WEIGHTS,
    NUM_CLASSES, REPLAY_RATIO, CLASS_LABELS,
)

# ── Augmentation ──────────────────────────────────────────────────────────────

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        A.Affine(scale=(0.85, 1.15), translate_percent=0.06,
                 rotate=(-20, 20), shear=(-5, 5), p=0.5),
        A.ElasticTransform(alpha=60, sigma=6, p=0.25),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.25),
        A.OpticalDistortion(distort_limit=0.15, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=15, p=0.3),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 6),
                        hole_height_range=(16, 48), hole_width_range=(16, 48), p=0.3),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3), p=0.2),
    ])

# ── Dataset ───────────────────────────────────────────────────────────────────

class GeoSegDataset(Dataset):
    """
    proc_dir     : used for current-shard tiles (images/ and masks/ subdirs).
    img_map / msk_map : {tile_id: Path} — used for replay tiles which live in
                   shard subdirs under data/replay/.
    Pass exactly one of (proc_dir) or (img_map + msk_map).
    """
    def __init__(self, tile_ids, proc_dir=None, transform=None,
                 img_map=None, msk_map=None):
        self.tile_ids = tile_ids
        self.img_dir  = proc_dir / "images" if proc_dir else None
        self.msk_dir  = proc_dir / "masks"  if proc_dir else None
        self.img_map  = img_map
        self.msk_map  = msk_map
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
        image = np.load(self._img_path(tid))
        mask  = np.load(self._msk_path(tid))

        image_hwc = np.transpose(image, (1, 2, 0))
        if self.transform:
            n = image_hwc.shape[2]
            if n > 3:
                aug       = self.transform(image=image_hwc[:, :, :3], mask=mask)
                image_hwc = np.concatenate([aug["image"], image_hwc[:, :, 3:]], axis=2)
            else:
                aug       = self.transform(image=image_hwc, mask=mask)
                image_hwc = aug["image"]
            mask = aug["mask"]

        # ImageNet normalization AFTER augmentation so albumentations
        # always receives clean [0, 1] inputs.
        image_chw = np.transpose(image_hwc, (2, 0, 1))
        image_chw = _imagenet_normalize(image_chw)

        return (
            torch.from_numpy(np.ascontiguousarray(image_chw)).float(),
            torch.from_numpy(np.ascontiguousarray(mask)).long(),
        )

# ── Class-balanced sampling ───────────────────────────────────────────────────

def compute_sample_weights(tile_meta_list):
    return [
        max(SAMPLER_CLASS_WEIGHTS.get(cid, 1.0)
            for cid in t.get("class_ids", [0]))
        for t in tile_meta_list
    ]

# ── Train / val split ─────────────────────────────────────────────────────────

def split_tiles(tiles):
    """
    Stratified TIFF-level split: every TIFF contributes VAL_SPLIT fraction
    of its tiles to val, so the val class distribution mirrors the whole
    dataset — no single TIFF's class imbalance (e.g. dense built-up) can
    deflate or inflate the mIoU signal.

    Spatial leakage is mitigated by SYSTEMATIC sampling (every Nth tile)
    within each TIFF rather than random — tiles are stored in row-major
    order by 01_preprocess.py, so every-Nth spacing spreads val tiles
    across the full image rather than clustering neighbours together.

    Falls back to the old random tile split only when there is exactly
    1 source TIFF (nothing to stratify over).
    """
    from collections import defaultdict

    tiff_buckets = defaultdict(list)
    for t in tiles:
        stem = "_".join(t["id"].split("_")[:-2])
        tiff_buckets[stem].append(t)

    tiff_keys = sorted(tiff_buckets.keys())
    n_tiffs   = len(tiff_keys)

    # ── Fallback: only 1 TIFF ────────────────────────────────────────────
    if n_tiffs < 2:
        print(
            " [WARN] Only 1 source TIFF — falling back to random tile split. "
            "Val mIoU will be optimistic due to tile overlap leakage.",
            flush=True,
        )
        import random as _rnd
        idx = list(range(len(tiles)))
        rng = _rnd.Random(RANDOM_SEED)
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * VAL_SPLIT))
        va_idx, tr_idx = idx[:cut], idx[cut:]
        return [tiles[i] for i in tr_idx], [tiles[i] for i in va_idx]

    # ── Stratified split: each TIFF donates VAL_SPLIT% of its tiles ──────
    train_meta, val_meta = [], []
    for key in tiff_keys:
        bucket = tiff_buckets[key]
        n      = len(bucket)
        n_val  = max(1, int(round(n * VAL_SPLIT)))

        # Systematic (every-Nth) sampling within the TIFF.
        # Tiles are in row-major order from 01_preprocess.py, so picking
        # every step-th tile spaces val tiles across the full image rather
        # than clustering them in one corner (which random sampling risks).
        step       = max(1, n // n_val)
        val_indices = set(list(range(0, n, step))[:n_val])

        for i, tile in enumerate(bucket):
            (val_meta if i in val_indices else train_meta).append(tile)

    val_frac = len(val_meta) / max(len(tiles), 1)
    print(
        f" Stratified split across {n_tiffs} TIFFs: "
        f"train={len(train_meta)} tiles  val={len(val_meta)} tiles "
        f"({val_frac * 100:.1f}% val)",
        flush=True,
    )
    print(f" TIFFs contributing to val: {tiff_keys}", flush=True)
    return train_meta, val_meta

# ── DataLoaders ───────────────────────────────────────────────────────────────

def build_dataloaders(proc_dir=None):
    if proc_dir is None:
        proc_dir = Path(DATA_PROCESSED_DIR)

    meta_path = proc_dir / "tiles_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run 01_preprocess.py first. Expected: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    tiles = meta["tiles"]
    print(f"Total tiles this shard: {len(tiles)}", flush=True)

    train_meta, val_meta = split_tiles(tiles)
    train_ids = [t["id"] for t in train_meta]
    val_ids   = [t["id"] for t in val_meta]
    print(f"  Train: {len(train_ids)}  Val: {len(val_ids)}", flush=True)

    # ── Val class distribution check ─────────────────────────────────────
    from collections import Counter
    val_cls = Counter(cid for t in val_meta for cid in t.get("class_ids", [0]))
    print(" Val class coverage:", flush=True)
    for cid in range(len(CLASS_LABELS)):
        print(f"   {CLASS_LABELS[cid]:12s}: {val_cls.get(cid, 0)} tiles", flush=True)

    # ── Replay tiles from previous shards ────────────────────────────────
    from replay_buffer import load_all_replay, replay_exists
    replay_tiles, replay_img_map, replay_msk_map = [], {}, {}
    if replay_exists():
        replay_tiles, replay_img_map, replay_msk_map = load_all_replay()

    train_transform = get_train_transforms() if AUGMENT_TRAIN else None
    current_ds = GeoSegDataset(train_ids, proc_dir, transform=train_transform)

    if replay_tiles:
        replay_ids = [t["id"] for t in replay_tiles]
        replay_ds  = GeoSegDataset(
            replay_ids, proc_dir=None, transform=train_transform,
            img_map=replay_img_map, msk_map=replay_msk_map,
        )

        n_current    = len(train_ids)
        n_replay     = len(replay_ids)
        replay_scale = min(
            (n_current * REPLAY_RATIO) / max(n_replay * (1 - REPLAY_RATIO), 1),
            5.0,
        )
        print(f" Replay: {n_replay} tiles scale={replay_scale:.2f}× "
              f"(target {REPLAY_RATIO*100:.0f}% of batches from replay)", flush=True)

        current_weights = compute_sample_weights(train_meta)
        replay_weights  = [
            replay_scale * SAMPLER_CLASS_WEIGHTS.get(
                max(t.get("class_ids", [0]),
                    key=lambda c: SAMPLER_CLASS_WEIGHTS.get(c, 0.0)), 1.0)
            for t in replay_tiles
        ]
        all_weights  = current_weights + replay_weights
        combined_ds  = ConcatDataset([current_ds, replay_ds])

        sampler = WeightedRandomSampler(
            weights=all_weights, num_samples=len(all_weights), replacement=True
        )
        train_loader = DataLoader(
            combined_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True,
            worker_init_fn=_worker_init_fn,
        )

    elif USE_WEIGHTED_SAMPLER and train_meta:
        sampler = WeightedRandomSampler(
            weights=compute_sample_weights(train_meta),
            num_samples=len(train_meta), replacement=True,
        )
        print(" WeightedRandomSampler (no replay yet)", flush=True)
        train_loader = DataLoader(
            current_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True,
            worker_init_fn=_worker_init_fn,
        )
    else:
        train_loader = DataLoader(
            current_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
            worker_init_fn=_worker_init_fn,
        )

    val_loader = DataLoader(
        GeoSegDataset(val_ids, proc_dir),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return train_loader, val_loader, meta["num_bands"]
