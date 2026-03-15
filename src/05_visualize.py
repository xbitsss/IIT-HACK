"""
05_visualize.py — Visualize random tiles from the processed dataset.
Shows image (RGB) + mask side by side for N random tiles.
Saves a grid PNG to outputs/dataset_samples.png

Usage:
    python src/05_visualize.py              # 10 random tiles
    python src/05_visualize.py --n 20       # 20 random tiles
    python src/05_visualize.py --n 10 --split train   # from train split only
    python src/05_visualize.py --show-class road       # only tiles containing road
"""

import sys
import json
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROCESSED_DIR, OUTPUT_DIR, CLASS_LABELS, CLASS_COLORS,
    NUM_CLASSES, VAL_SPLIT, RANDOM_SEED, CLASSES
)

# ── Color map ─────────────────────────────────────────────────────────────────

def build_colormap():
    """Build a (NUM_CLASSES, 3) RGB array for colorizing masks."""
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for class_id, rgb in CLASS_COLORS.items():
        cmap[class_id] = rgb
    return cmap

COLORMAP = build_colormap()


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert (H, W) label mask → (H, W, 3) RGB image."""
    return COLORMAP[mask.astype(np.uint8)]


# ── Tile loading ──────────────────────────────────────────────────────────────

def load_tile(tile_id: str, proc_dir: Path):
    image = np.load(proc_dir / "images" / f"{tile_id}.npy")  # (C, H, W) float32
    mask  = np.load(proc_dir / "masks"  / f"{tile_id}.npy")  # (H, W) uint8
    return image, mask


def get_rgb(image: np.ndarray) -> np.ndarray:
    """Extract RGB from (C, H, W) — uses first 3 bands."""
    rgb = image[:3]  # (3, H, W)
    rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    return rgb


def class_distribution(mask: np.ndarray) -> str:
    unique, counts = np.unique(mask, return_counts=True)
    total = mask.size
    parts = []
    for u, c in zip(unique, counts):
        label = CLASS_LABELS[int(u)] if int(u) < len(CLASS_LABELS) else f"cls{u}"
        parts.append(f"{label[:3]}:{c/total*100:.0f}%")
    return " ".join(parts)


# ── Filter tiles ──────────────────────────────────────────────────────────────

def filter_tiles(tiles, show_class, proc_dir, sample_size=200):
    """Filter tiles that contain a specific class. Samples for speed."""
    if show_class is None:
        return tiles

    class_id = CLASSES.get(show_class)
    if class_id is None:
        print(f"[WARN] Unknown class '{show_class}'. Available: {list(CLASSES.keys())}")
        return tiles

    # Sample a subset to avoid scanning all tiles
    sample = random.sample(tiles, min(sample_size, len(tiles)))
    filtered = []
    for t in sample:
        mask = np.load(proc_dir / "masks" / f"{t['id']}.npy")
        if class_id in mask:
            filtered.append(t)

    print(f"  Found {len(filtered)} tiles containing '{show_class}' (scanned {len(sample)})")
    return filtered if filtered else tiles


# ── Main visualization ────────────────────────────────────────────────────────

def visualize(n=10, split=None, show_class=None, seed=None):
    proc_dir   = Path(DATA_PROCESSED_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = proc_dir / "tiles_meta.json"
    if not meta_path.exists():
        print(f"[ERROR] No tiles_meta.json found. Run 01_preprocess.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    tiles   = meta["tiles"]
    sources = list({t["source"] for t in tiles})
    print(f"Total tiles: {len(tiles)} from {len(sources)} TIFFs")

    # Split filtering
    if split == "val":
        from sklearn.model_selection import train_test_split
        _, val_sources = train_test_split(sources, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
        tiles = [t for t in tiles if t["source"] in val_sources]
        print(f"  Val split: {len(tiles)} tiles")
    elif split == "train":
        from sklearn.model_selection import train_test_split
        train_sources, _ = train_test_split(sources, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
        tiles = [t for t in tiles if t["source"] in train_sources]
        print(f"  Train split: {len(tiles)} tiles")

    # Class filtering
    tiles = filter_tiles(tiles, show_class, proc_dir)

    if not tiles:
        print("[ERROR] No tiles to show.")
        sys.exit(1)

    # Random sample
    rng = random.Random(seed or RANDOM_SEED)
    selected = rng.sample(tiles, min(n, len(tiles)))
    print(f"  Showing {len(selected)} random tiles")

    # Build figure
    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 5 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for i, tile_meta in enumerate(selected):
        tid   = tile_meta["id"]
        image, mask = load_tile(tid, proc_dir)
        rgb         = get_rgb(image)
        color_mask  = colorize_mask(mask)
        dist        = class_distribution(mask)

        ax_rgb, ax_mask, ax_overlay = axes[i]

        # RGB image
        ax_rgb.imshow(rgb)
        ax_rgb.set_title(f"RGB  |  {tid.split('_r')[0][:20]}", fontsize=8)
        ax_rgb.axis("off")

        # Mask
        ax_mask.imshow(color_mask)
        ax_mask.set_title(f"Mask  |  {dist}", fontsize=8)
        ax_mask.axis("off")

        # Overlay: RGB + semi-transparent mask
        overlay = rgb.copy().astype(np.float32)
        mask_rgb = color_mask.astype(np.float32)
        # Only overlay non-background pixels
        bg_mask = mask == 0
        alpha   = np.where(bg_mask, 0.0, 0.45)[:, :, np.newaxis]
        overlay = (overlay * (1 - alpha) + mask_rgb * alpha).clip(0, 255).astype(np.uint8)
        ax_overlay.imshow(overlay)
        ax_overlay.set_title("Overlay", fontsize=8)
        ax_overlay.axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255, label=CLASS_LABELS[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=NUM_CLASSES,
               fontsize=10, bbox_to_anchor=(0.5, 0.0))

    title = f"Dataset Samples — {len(selected)} tiles"
    if show_class:
        title += f" (class: {show_class})"
    if split:
        title += f" ({split} split)"
    fig.suptitle(title, fontsize=14, y=1.01)

    plt.tight_layout()
    out_path = output_dir / "dataset_samples.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize dataset tiles")
    parser.add_argument("--n",           type=int,   default=10,   help="Number of tiles to show")
    parser.add_argument("--split",       type=str,   default=None, choices=["train", "val"],
                        help="Show only train or val tiles")
    parser.add_argument("--show-class",  type=str,   default=None,
                        help="Only show tiles containing this class (e.g. road, waterbody)")
    parser.add_argument("--seed",        type=int,   default=None, help="Random seed")
    args = parser.parse_args()

    visualize(n=args.n, split=args.split, show_class=args.show_class, seed=args.seed)