"""
Dataset + DataLoader builder for the specialist model.

Reuses GeoSegDataset, split_tiles, get_train_transforms, worker_init_fn
from src/02_dataset.py verbatim.  Only differences:
  - reads tiles_meta_specialist.json
  - uses specialist SAMPLER_CLASS_WEIGHTS from config_specialist
"""

import json
import sys
import importlib
from pathlib import Path

# src/ for base modules; specialist/ for config_specialist
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from config_specialist import (
    DATAPROCESSEDDIR, TILES_META_PATH,
    AUGMENTTRAIN, BATCHSIZE, USEWEIGHTEDSAMPLER,
    SAMPLER_CLASS_WEIGHTS, REPLAYRATIO,
)

_base = importlib.import_module("02_dataset")

GeoSegDataset         = _base.GeoSegDataset
get_train_transforms  = _base.get_train_transforms
split_tiles           = _base.split_tiles
worker_init_fn        = _base.worker_init_fn

from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset


def compute_sample_weights(tile_meta_list):
    """Same logic as base, but uses specialist SAMPLER_CLASS_WEIGHTS."""
    return [
        max(SAMPLER_CLASS_WEIGHTS.get(cid, 1.0) for cid in t.get("classids", [0]))
        for t in tile_meta_list
    ]


def build_dataloaders(procdir=None):
    if procdir is None:
        procdir = Path(DATAPROCESSEDDIR)

    metapath = Path(TILES_META_PATH)
    if not metapath.exists():
        raise FileNotFoundError(
            f"Run specialist/01_build_meta.py first. Expected: {metapath}"
        )

    with open(metapath) as f:
        meta = json.load(f)

    tiles = meta["tiles"]
    print(f"[Specialist] Total tiles: {len(tiles)}", flush=True)

    train_meta, val_meta = split_tiles(tiles)
    train_ids = [t["id"] for t in train_meta]
    val_ids   = [t["id"] for t in val_meta]
    print(f"Train {len(train_ids)}  Val {len(val_ids)}", flush=True)

    # ── replay buffer (reuse as-is from base) ─────────────────────────────
    from replay_buffer import load_all_replay, replay_exists
    replay_tiles, replay_imgmap, replay_mskmap = [], {}, {}
    if replay_exists():
        replay_tiles, replay_imgmap, replay_mskmap = load_all_replay()

    train_transform = get_train_transforms() if AUGMENTTRAIN else None
    current_ds      = GeoSegDataset(train_ids, procdir, transform=train_transform)

    if replay_tiles:
        replay_ids = [t["id"] for t in replay_tiles]
        replay_ds  = GeoSegDataset(
            replay_ids, procdir=None, transform=train_transform,
            imgmap=replay_imgmap, mskmap=replay_mskmap,
        )
        n_current, n_replay = len(train_ids), len(replay_ids)
        replay_scale = min(
            n_current * REPLAYRATIO / max(n_replay * (1 - REPLAYRATIO), 1),
            5.0,
        )
        current_weights = compute_sample_weights(train_meta)
        replay_weights  = [
            replay_scale * SAMPLER_CLASS_WEIGHTS.get(
                max(t.get("classids", [0]),
                    key=lambda c: SAMPLER_CLASS_WEIGHTS.get(c, 0.0)), 1.0
            )
            for t in replay_tiles
        ]
        all_weights = current_weights + replay_weights
        combined_ds = ConcatDataset([current_ds, replay_ds])
        sampler = WeightedRandomSampler(
            weights=all_weights, num_samples=len(all_weights), replacement=True
        )
        train_loader = DataLoader(
            combined_ds, batch_size=BATCHSIZE, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True,
            worker_init_fn=worker_init_fn,
        )
    elif USEWEIGHTEDSAMPLER and train_meta:
        sampler = WeightedRandomSampler(
            weights=compute_sample_weights(train_meta),
            num_samples=len(train_meta), replacement=True,
        )
        train_loader = DataLoader(
            current_ds, batch_size=BATCHSIZE, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True,
            worker_init_fn=worker_init_fn,
        )
    else:
        train_loader = DataLoader(
            current_ds, batch_size=BATCHSIZE, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
            worker_init_fn=worker_init_fn,
        )

    val_loader = DataLoader(
        GeoSegDataset(val_ids, procdir),
        batch_size=BATCHSIZE, shuffle=False,
        num_workers=4, pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, meta["num_bands"]
