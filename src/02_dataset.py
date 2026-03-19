"""
02_dataset.py — PyTorch Dataset + augmentation + class-balanced sampling.

Replay buffer is included: when training on shard N, tiles from shards
1…N-1 are mixed in via WeightedRandomSampler to prevent catastrophic
forgetting.  The mix ratio is REPLAY_RATIO (default 25% of each batch).
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


def _worker_init_fn(worker_id):
    """
    Called in each DataLoader worker process at startup.
    Resets SIGINT to SIG_IGN so Ctrl+C in the terminal is handled only by
    the main training process — not by each of the 4 worker subprocesses.
    Without this, pressing Ctrl+C fires the crash handler once per worker
    plus once in the main process, sending N+1 crash emails.
    """
    import signal as _signal
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
from config import (
    DATA_PROCESSED_DIR, DATA_RAW_DIR, VAL_SPLIT,
    RANDOM_SEED, AUGMENT_TRAIN, BATCH_SIZE,
    USE_WEIGHTED_SAMPLER, SAMPLER_CLASS_WEIGHTS,
    NUM_CLASSES, REPLAY_RATIO,
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
        A.OpticalDistortion(distort_limit=0.15, shift_limit=0.1, p=0.2),
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
    proc_dir  : used for current-shard tiles (images/ and masks/ subdirs).
    img_map / msk_map : {tile_id: Path} — used for replay tiles which live in
                        shard subdirs under data/replay/.
    Pass exactly one of (proc_dir) or (img_map + msk_map).
    """
    def __init__(self, tile_ids, proc_dir=None, transform=None,
                 img_map=None, msk_map=None):
        self.tile_ids  = tile_ids
        self.img_dir   = proc_dir / "images" if proc_dir else None
        self.msk_dir   = proc_dir / "masks"  if proc_dir else None
        self.img_map   = img_map
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

        return (
            torch.from_numpy(np.ascontiguousarray(
                np.transpose(image_hwc, (2, 0, 1)))).float(),
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

def _tiff_sizes(raw_dir: Path) -> dict:
    sizes = {}
    for p in list(raw_dir.glob("**/*.tif")) + list(raw_dir.glob("**/*.tiff")):
        sizes[p.name] = p.stat().st_size
    return sizes


def split_tiles(tiles):
    print("  Stratified random 80/20 split", flush=True)
    idx = list(range(len(tiles)))
    tr, va = train_test_split(idx, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
    return [tiles[i] for i in tr], [tiles[i] for i in va]


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

    # ── Val class distribution check ─────────────────────────────────────────
    from collections import Counter
    from config import CLASS_LABELS as CL
    val_cls = Counter(cid for t in val_meta for cid in t.get("class_ids", [0]))
    print("  Val class coverage:", flush=True)
    for cid in range(len(CL)):
        print(f"    {CL[cid]:12s}: {val_cls.get(cid,0)} tiles", flush=True)

    # ── Replay tiles from previous shards ────────────────────────────────────
    from replay_buffer import load_all_replay, replay_exists
    replay_tiles, replay_img_map, replay_msk_map = [], {}, {}
    if replay_exists():
        replay_tiles, replay_img_map, replay_msk_map = load_all_replay()

    train_transform = get_train_transforms() if AUGMENT_TRAIN else None
    current_ds      = GeoSegDataset(train_ids, proc_dir, transform=train_transform)

    if replay_tiles:
        replay_ids = [t["id"] for t in replay_tiles]
        replay_ds  = GeoSegDataset(
            replay_ids, proc_dir=None, transform=train_transform,
            img_map=replay_img_map, msk_map=replay_msk_map,
        )

        n_current     = len(train_ids)
        n_replay      = len(replay_ids)
        replay_scale  = min(
            (n_current * REPLAY_RATIO) / max(n_replay * (1 - REPLAY_RATIO), 1),
            5.0,
        )
        print(f"  Replay: {n_replay} tiles  scale={replay_scale:.2f}×  "
              f"(target {REPLAY_RATIO*100:.0f}% of batches from replay)", flush=True)

        current_weights = compute_sample_weights(train_meta)
        replay_weights  = [
            replay_scale * SAMPLER_CLASS_WEIGHTS.get(
                max(t.get("class_ids", [0]),
                    key=lambda c: SAMPLER_CLASS_WEIGHTS.get(c, 0.0)), 1.0)
            for t in replay_tiles
        ]
        all_weights = current_weights + replay_weights
        combined_ds = ConcatDataset([current_ds, replay_ds])

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
        print("  WeightedRandomSampler (no replay yet)", flush=True)
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