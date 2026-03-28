"""
Dataset + DataLoader builder for the specialist model.

Reuses GeoSegDataset, split_tiles, get_train_transforms, worker_init_fn
from src/02_dataset.py verbatim.  Only differences:
  - reads tiles_meta_specialist.json
  - uses specialist SAMPLER_CLASS_WEIGHTS from config_specialist

Specialist-only mode:
  When running --specialist-only the current processed/ directory may be
  empty (it was wiped after generalist training).  This module partitions
  tile IDs by where their .npy files actually live:
    • procdir tiles  — images exist under data/processed/images/
    • replay tiles   — images found via the replay-buffer img_map
  IDs whose images exist in neither location are skipped with a warning.
"""

import json
import sys
import importlib
from pathlib import Path

# src/ for base modules; specialist/ for config_specialist
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# ── BUG FIX: all names corrected to snake_case matching config.py convention
from config_specialist import (
    DATA_PROCESSED_DIR, TILES_META_PATH,
    AUGMENT_TRAIN, BATCH_SIZE, USE_WEIGHTED_SAMPLER,
    SAMPLER_CLASS_WEIGHTS, REPLAY_RATIO,
)

_base = importlib.import_module("02_dataset")

GeoSegDataset         = _base.GeoSegDataset
get_train_transforms  = _base.get_train_transforms
split_tiles           = _base.split_tiles
# ── BUG FIX: _worker_init_fn was private; 02_dataset.py now exposes
# the public alias `worker_init_fn = _worker_init_fn` for this import.
worker_init_fn        = _base.worker_init_fn

from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset


def compute_sample_weights(tile_meta_list):
    """Same logic as base, but uses specialist SAMPLER_CLASS_WEIGHTS."""
    # BUG FIX: key was "classids" — correct key is "class_ids"
    return [
        max(SAMPLER_CLASS_WEIGHTS.get(cid, 1.0) for cid in t.get("class_ids", [0]))
        for t in tile_meta_list
    ]


def build_dataloaders(procdir=None):
    if procdir is None:
        procdir = Path(DATA_PROCESSED_DIR)

    metapath = Path(TILES_META_PATH)
    if not metapath.exists():
        raise FileNotFoundError(
            f"Run specialist/01_build_specialist_meta.py first. Expected: {metapath}"
        )

    with open(metapath) as f:
        meta = json.load(f)

    tiles = meta["tiles"]
    print(f"[Specialist] Total tiles: {len(tiles)}", flush=True)

    train_meta, val_meta = split_tiles(tiles)
    train_ids = [t["id"] for t in train_meta]
    val_ids   = [t["id"] for t in val_meta]
    print(f"Train {len(train_ids)}  Val {len(val_ids)}", flush=True)

    # ── Always load the replay buffer up-front ─────────────────────────────
    # In specialist-only mode procdir/images/ may be empty — all tile images
    # will be found in the replay buffer.  In the full pipeline (last shard
    # kept alive), some tiles live in procdir and others in replay.
    from replay_buffer import load_all_replay, replay_exists
    replay_tiles, replay_img_map, replay_msk_map = [], {}, {}
    if replay_exists():
        replay_tiles, replay_img_map, replay_msk_map = load_all_replay()

    # ── Partition train tile IDs by where images actually live ─────────────
    img_dir = procdir / "images"
    proc_train_meta,   replay_train_meta,   skipped_meta   = [], [], []
    proc_train_ids,    replay_train_ids,    skipped_ids    = [], [], []

    for t in train_meta:
        tid = t["id"]
        if img_dir.exists() and (img_dir / f"{tid}.npy").exists():
            proc_train_meta.append(t)
            proc_train_ids.append(tid)
        elif tid in replay_img_map:
            replay_train_meta.append(t)
            replay_train_ids.append(tid)
        else:
            skipped_meta.append(t)
            skipped_ids.append(tid)

    if skipped_ids:
        print(f"[WARN] {len(skipped_ids)} train tile(s) not found in procdir or "
              f"replay buffer — skipping.", flush=True)

    print(
        f"  procdir tiles: {len(proc_train_ids)}  "
        f"replay tiles: {len(replay_train_ids)}  "
        f"skipped: {len(skipped_ids)}",
        flush=True,
    )

    # ── Partition val tile IDs the same way ───────────────────────────────
    proc_val_ids,  replay_val_meta,  replay_val_ids  = [], [], []
    for t in val_meta:
        tid = t["id"]
        if img_dir.exists() and (img_dir / f"{tid}.npy").exists():
            proc_val_ids.append(tid)
        elif tid in replay_img_map:
            replay_val_meta.append(t)
            replay_val_ids.append(tid)
        # silently skip val tiles not found anywhere (rare edge case)

    # ── Build datasets ─────────────────────────────────────────────────────
    train_transform = get_train_transforms() if AUGMENT_TRAIN else None

    train_datasets = []
    train_weights_list = []

    if proc_train_ids:
        ds = GeoSegDataset(proc_train_ids, procdir, transform=train_transform)
        train_datasets.append(ds)
        train_weights_list.extend(compute_sample_weights(proc_train_meta))

    if replay_train_ids:
        # BUG FIX: kwargs were `procdir=` / `imgmap=` / `mskmap=`
        #          Correct: `proc_dir=` / `img_map=` / `msk_map=`
        ds = GeoSegDataset(
            replay_train_ids,
            proc_dir=None,
            transform=train_transform,
            img_map=replay_img_map,
            msk_map=replay_msk_map,
        )
        train_datasets.append(ds)
        train_weights_list.extend(compute_sample_weights(relay_train_meta := replay_train_meta))

    if not train_datasets:
        raise RuntimeError(
            "No training tiles found in procdir or replay buffer. "
            "Run generalist shards first, or ensure --specialist-only is used "
            "after replay data has been saved."
        )

    # ── Optional extra replay mix (continual-learning context tiles) ───────
    # Primary train IDs already come from the specialist meta (minor+major).
    # Mix in the remaining replay tiles at REPLAY_RATIO weight so the model
    # keeps seeing varied non-specialist context without forgetting.
    primary_ids = set(proc_train_ids + replay_train_ids)
    extra_replay_tiles = [t for t in replay_tiles if t["id"] not in primary_ids]

    if extra_replay_tiles:
        n_primary  = len(primary_ids)
        n_extra    = len(extra_replay_tiles)
        # Scale extra tiles so they constitute ~REPLAY_RATIO of total batches
        replay_scale = min(
            n_primary * REPLAY_RATIO / max(n_extra * (1 - REPLAY_RATIO), 1),
            5.0,
        )
        extra_ids = [t["id"] for t in extra_replay_tiles]
        extra_ds  = GeoSegDataset(
            extra_ids,
            proc_dir=None,
            transform=train_transform,
            img_map=replay_img_map,
            msk_map=replay_msk_map,
        )
        train_datasets.append(extra_ds)
        # BUG FIX: key was "classids" — correct key is "class_ids"
        extra_weights = [
            replay_scale * SAMPLER_CLASS_WEIGHTS.get(
                max(t.get("class_ids", [0]),
                    key=lambda c: SAMPLER_CLASS_WEIGHTS.get(c, 0.0)), 1.0
            )
            for t in extra_replay_tiles
        ]
        train_weights_list.extend(extra_weights)
        print(
            f"  Extra replay context: {n_extra} tiles  "
            f"(scale={replay_scale:.2f}×, ~{REPLAY_RATIO*100:.0f}% of batches)",
            flush=True,
        )

    # ── Build combined train dataset + sampler ─────────────────────────────
    if len(train_datasets) == 1:
        combined_train_ds = train_datasets[0]
    else:
        combined_train_ds = ConcatDataset(train_datasets)

    if USE_WEIGHTED_SAMPLER and train_weights_list:
        sampler = WeightedRandomSampler(
            weights=train_weights_list,
            num_samples=len(train_weights_list),
            replacement=True,
        )
        train_loader = DataLoader(
            combined_train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True,
            worker_init_fn=worker_init_fn,
        )
    else:
        train_loader = DataLoader(
            combined_train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
            worker_init_fn=worker_init_fn,
        )

    # ── Val dataset ────────────────────────────────────────────────────────
    val_datasets = []
    if proc_val_ids:
        val_datasets.append(GeoSegDataset(proc_val_ids, procdir))
    if replay_val_ids:
        val_datasets.append(GeoSegDataset(
            replay_val_ids, proc_dir=None,
            img_map=replay_img_map, msk_map=replay_msk_map,
        ))

    if not val_datasets:
        # Edge case: no val tiles found anywhere — fall back to a tiny subset
        # of train tiles so training can at least report a val loss.
        print("[WARN] No val tiles found — using first 10% of train tiles as val proxy.",
              flush=True)
        fallback_ids = (proc_train_ids + replay_train_ids)[:max(1, len(proc_train_ids + replay_train_ids) // 10)]
        val_datasets.append(GeoSegDataset(
            fallback_ids, procdir if proc_train_ids else None,
            img_map=replay_img_map if replay_train_ids else None,
            msk_map=replay_msk_map if replay_train_ids else None,
        ))

    val_combined = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    val_loader = DataLoader(
        val_combined, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, meta["num_bands"]