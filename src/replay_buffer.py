"""
replay_buffer.py — Continual-learning replay buffer for incremental training.

WHY THIS EXISTS
───────────────
The dataset is split into 4 shards (CG_1 … CG_4), each preprocessed and
trained on individually to fit within disk constraints.  Without replay,
training on CG_2 will cause the model to catastrophically forget what it
learned from CG_1.  This is a well-studied problem in continual learning.

HOW IT WORKS
────────────
After preprocessing each shard, `save_replay_from_shard()` is called.
It selects REPLAY_TILES_PER_SHARD tiles — stratified by class to ensure
rare classes (road, water) are represented — and copies them into a
persistent `replay/` directory that is NEVER wiped between shards.

When building the training dataloader, `build_dataloaders()` in
02_dataset.py automatically detects the replay directory and mixes
replay tiles into the current shard's training set.

The mix ratio is controlled by REPLAY_RATIO in config.py:
  - 0.3 means 30% of each batch comes from replay tiles
  - Replay tiles receive the same augmentation as current tiles
  - Their sample weights are boosted to ensure they're seen regularly

DISK COST
─────────
With REPLAY_TILES_PER_SHARD=300 and 512×512 tiles at 4 bands float32:
  300 tiles × (4 × 512 × 512 × 4 bytes image + 512 × 512 × 1 byte mask)
  ≈ 300 × (4 MB + 0.25 MB) ≈ 1.3 GB across all 4 shards (4 × 300 = 1200 tiles)
Adjust REPLAY_TILES_PER_SHARD down if disk is very tight.

STRATEGY: Herding / class-stratified reservoir sampling
──────────────────────────────────────────────────────
We don't just pick tiles randomly.  We:
  1. Bucket tiles by their rarest contained class.
  2. Fill each class bucket proportionally to its inverse frequency.
  3. Within each bucket, pick tiles that have the HIGHEST coverage ratio
     (most annotation density) — these are the most information-rich tiles.

This ensures the replay buffer is compact but maximally representative.
"""

import json
import shutil
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROCESSED_DIR, CHECKPOINT_DIR,
    NUM_CLASSES, CLASS_LABELS, RANDOM_SEED,
    REPLAY_TILES_PER_SHARD, REPLAY_DIR,
    SAMPLER_CLASS_WEIGHTS,
)


# ── Public API ────────────────────────────────────────────────────────────────

def save_replay_from_shard(shard_name: str, proc_dir: Path = None):
    """
    Called after preprocessing a shard (before training begins).
    Selects the best REPLAY_TILES_PER_SHARD tiles and copies them to
    REPLAY_DIR / shard_name / {images,masks}/.

    Existing replay for this shard is replaced (idempotent — safe to re-run).
    """
    if proc_dir is None:
        proc_dir = Path(DATA_PROCESSED_DIR)

    replay_dir = Path(REPLAY_DIR)
    shard_dir  = replay_dir / shard_name
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    (shard_dir / "images").mkdir(parents=True)
    (shard_dir / "masks").mkdir(parents=True)

    meta_path = proc_dir / "tiles_meta.json"
    if not meta_path.exists():
        print(f"[REPLAY] No tiles_meta.json — skipping replay save for {shard_name}")
        return []

    with open(meta_path) as f:
        meta = json.load(f)

    tiles = meta["tiles"]
    print(f"[REPLAY] Selecting {REPLAY_TILES_PER_SHARD} replay tiles from "
          f"{len(tiles)} tiles in {shard_name}...", flush=True)

    selected = _stratified_select(tiles, REPLAY_TILES_PER_SHARD)

    # Copy image + mask files
    img_dir = proc_dir / "images"
    msk_dir = proc_dir / "masks"
    copied  = []
    for tile in selected:
        tid = tile["id"]
        src_img = img_dir / f"{tid}.npy"
        src_msk = msk_dir / f"{tid}.npy"
        if not src_img.exists() or not src_msk.exists():
            continue
        shutil.copy2(src_img, shard_dir / "images" / f"{tid}.npy")
        shutil.copy2(src_msk, shard_dir / "masks"  / f"{tid}.npy")
        copied.append(tile)

    # Write shard replay manifest
    manifest_path = shard_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"shard": shard_name, "tiles": copied}, f, indent=2)

    # Print class distribution of saved tiles
    _print_class_dist(copied, shard_name)

    print(f"[REPLAY] Saved {len(copied)} tiles to {shard_dir}", flush=True)
    return copied


def load_all_replay(replay_dir: Path = None):
    """
    Returns (tile_meta_list, img_dir_map, msk_dir_map) where *_dir_map maps
    tile_id → actual .npy path (needed because replay tiles live in shard
    subdirs, not the current proc_dir).

    img_dir_map / msk_dir_map are dicts: {tile_id: Path_to_npy_file}
    """
    if replay_dir is None:
        replay_dir = Path(REPLAY_DIR)

    if not replay_dir.exists():
        return [], {}, {}

    all_tiles = []
    img_map   = {}
    msk_map   = {}

    for shard_dir in sorted(replay_dir.iterdir()):
        manifest_path = shard_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        for tile in manifest["tiles"]:
            tid = tile["id"]
            img_p = shard_dir / "images" / f"{tid}.npy"
            msk_p = shard_dir / "masks"  / f"{tid}.npy"
            if img_p.exists() and msk_p.exists():
                all_tiles.append(tile)
                img_map[tid] = img_p
                msk_map[tid] = msk_p

    print(f"[REPLAY] Loaded {len(all_tiles)} replay tiles from "
          f"{sum(1 for d in replay_dir.iterdir() if d.is_dir())} shards", flush=True)
    if all_tiles:
        _print_class_dist(all_tiles, "ALL REPLAY")
    return all_tiles, img_map, msk_map


def replay_exists() -> bool:
    replay_dir = Path(REPLAY_DIR)
    if not replay_dir.exists():
        return False
    return any((d / "manifest.json").exists() for d in replay_dir.iterdir()
               if d.is_dir())


# ── Selection strategy ────────────────────────────────────────────────────────

def _stratified_select(tiles: list, n: int) -> list:
    """
    Stratify tiles by their rarest class.  Within each stratum, prefer tiles
    with the highest coverage_ratio (most annotation density).

    Class budget is inversely proportional to class frequency, so rare classes
    like road get more replay slots than abundant background-heavy tiles.
    """
    rng = random.Random(RANDOM_SEED)

    # Bucket tiles by rarest (highest-weight) class
    buckets = defaultdict(list)
    for tile in tiles:
        class_ids   = tile.get("class_ids", [0])
        rarest_cls  = max(class_ids, key=lambda c: SAMPLER_CLASS_WEIGHTS.get(c, 0.0))
        buckets[rarest_cls].append(tile)

    # Compute budget per class (weighted by SAMPLER_CLASS_WEIGHTS)
    total_weight = sum(SAMPLER_CLASS_WEIGHTS.get(c, 1.0) for c in buckets)
    budgets = {
        c: max(1, round(n * SAMPLER_CLASS_WEIGHTS.get(c, 1.0) / total_weight))
        for c in buckets
    }
    # Trim to exactly n
    while sum(budgets.values()) > n:
        biggest = max(budgets, key=budgets.get)
        budgets[biggest] -= 1

    selected = []
    for cls_id, budget in budgets.items():
        pool = buckets[cls_id]
        # Sort by coverage_ratio descending — highest annotation density first
        pool.sort(key=lambda t: t.get("coverage_ratio", 0.0), reverse=True)
        chosen = pool[:budget]   # take top-coverage tiles up to budget
        selected.extend(chosen)
        label = CLASS_LABELS[cls_id] if cls_id < len(CLASS_LABELS) else f"cls{cls_id}"
        print(f"  [REPLAY]   {label:12s}: {len(chosen):4d} tiles "
              f"(pool={len(pool):5d}, budget={budget})", flush=True)

    rng.shuffle(selected)
    return selected[:n]


def _print_class_dist(tiles: list, label: str):
    from collections import Counter
    ctr = Counter()
    for t in tiles:
        for cid in t.get("class_ids", [0]):
            ctr[cid] += 1
    parts = []
    for cid in sorted(ctr):
        lbl = CLASS_LABELS[cid] if cid < len(CLASS_LABELS) else f"cls{cid}"
        parts.append(f"{lbl}={ctr[cid]}")
    print(f"  [REPLAY] {label} class coverage: {', '.join(parts)}", flush=True)
