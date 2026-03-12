"""
04_inference.py
───────────────
Runs the trained model on new GeoTIFF files using sliding-window inference.

Features:
- Sliding window with overlap + Gaussian blending (no hard tile edges)
- Preserves original CRS, transform, and spatial metadata
- Outputs both:
    1. Class mask GeoTIFF (uint8, values = class IDs)
    2. Colorized RGB visualization PNG
- Optional prompt-based override: click a point → force its class

Usage:
    # Basic inference
    python src/04_inference.py --input data/raw/test.tif --output outputs/test_mask.tif

    # With visualization
    python src/04_inference.py --input data/raw/test.tif --output outputs/test_mask.tif --visualize

    # With point prompt override
    python src/04_inference.py --input test.tif --output mask.tif --prompt 12.34,56.78,road
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.transform import rowcol
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerConfig

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODEL_NAME, NUM_CLASSES, CLASS_LABELS, CLASS_COLORS, CLASSES,
    TILE_SIZE, INFERENCE_OVERLAP, INFERENCE_BATCH,
    BAND_INDICES, CHECKPOINT_DIR, OUTPUT_DIR
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model Loader ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path) -> tuple:
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    num_bands = ckpt["num_bands"]

    cfg = SegformerConfig.from_pretrained(MODEL_NAME)
    cfg.num_labels   = NUM_CLASSES
    cfg.id2label     = {i: l for i, l in enumerate(CLASS_LABELS)}
    cfg.label2id     = {l: i for i, l in enumerate(CLASS_LABELS)}
    cfg.num_channels = num_bands

    model = SegformerForSemanticSegmentation(cfg)

    # Patch embedding if needed
    if num_bands != 3:
        import torch.nn as nn
        old_embed = model.segformer.encoder.patch_embeddings[0].proj
        new_embed = nn.Conv2d(
            num_bands, old_embed.out_channels,
            kernel_size=old_embed.kernel_size, stride=old_embed.stride,
            padding=old_embed.padding, bias=old_embed.bias is not None,
        )
        model.segformer.encoder.patch_embeddings[0].proj = new_embed

    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    print(f"  Loaded — epoch {ckpt['epoch']}, val_mIoU={ckpt['val_miou']:.4f}, bands={num_bands}")
    return model, num_bands


# ── Gaussian Blending Window ──────────────────────────────────────────────────

def gaussian_window(size: int, sigma_ratio: float = 4.0) -> np.ndarray:
    """
    Creates a 2D Gaussian weight window.
    Tiles blended with this window produce smooth stitches — no hard edges.
    """
    sigma = size / sigma_ratio
    coords = np.arange(size) - size // 2
    g1d    = np.exp(-(coords**2) / (2 * sigma**2))
    g2d    = np.outer(g1d, g1d)
    return (g2d / g2d.max()).astype(np.float32)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def normalize_bands(bands: np.ndarray) -> np.ndarray:
    bands = bands.astype(np.float32)
    for i in range(bands.shape[0]):
        b = bands[i]
        bmin, bmax = b.min(), b.max()
        if bmax > bmin:
            bands[i] = (b - bmin) / (bmax - bmin)
        else:
            bands[i] = 0.0
    return bands


# ── Sliding Window Inference ──────────────────────────────────────────────────

def sliding_window_predict(model, image: np.ndarray) -> np.ndarray:
    """
    image: (C, H, W) float32 normalized
    Returns: (H, W) uint8 class mask
    """
    C, H, W  = image.shape
    stride   = TILE_SIZE - INFERENCE_OVERLAP
    win      = gaussian_window(TILE_SIZE)

    # Accumulators
    logit_sum = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    # Collect all tiles
    tiles, positions = [], []
    for r in range(0, max(H - TILE_SIZE + 1, 1), stride):
        for c in range(0, max(W - TILE_SIZE + 1, 1), stride):
            r2 = min(r + TILE_SIZE, H)
            c2 = min(c + TILE_SIZE, W)
            r1 = r2 - TILE_SIZE
            c1 = c2 - TILE_SIZE

            tile = image[:, r1:r2, c1:c2]
            tiles.append(tile)
            positions.append((r1, r2, c1, c2))

    # Batch inference
    batch_size = INFERENCE_BATCH
    for i in tqdm(range(0, len(tiles), batch_size), desc="  Inference batches"):
        batch_tiles = tiles[i:i+batch_size]
        batch_pos   = positions[i:i+batch_size]

        tensor = torch.from_numpy(np.stack(batch_tiles)).to(DEVICE)

        with torch.no_grad():
            out = model(pixel_values=tensor)
            logits = F.interpolate(
                out.logits, size=(TILE_SIZE, TILE_SIZE),
                mode="bilinear", align_corners=False
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()   # (B, C, H, W)

        for j, (r1, r2, c1, c2) in enumerate(batch_pos):
            logit_sum[:, r1:r2, c1:c2] += probs[j] * win[np.newaxis]
            weight_sum[r1:r2, c1:c2]   += win

    # Normalize and argmax
    weight_sum = np.maximum(weight_sum, 1e-6)
    prob_map   = logit_sum / weight_sum[np.newaxis]
    class_mask = prob_map.argmax(axis=0).astype(np.uint8)

    return class_mask


# ── Point Prompt Override ─────────────────────────────────────────────────────

def apply_point_prompts(mask: np.ndarray, prompts: list, tif: rasterio.DatasetReader) -> np.ndarray:
    """
    prompts: list of (lon, lat, class_name) tuples
    Overrides a small region around each prompt point with the specified class.
    """
    mask = mask.copy()
    for lon, lat, class_name in prompts:
        if class_name not in CLASSES:
            print(f"  [WARN] Unknown class '{class_name}' in prompt — skipping")
            continue
        row, col = rowcol(tif.transform, lon, lat)
        row, col = int(row), int(col)
        class_id = CLASSES[class_name]
        r_size   = 5   # override radius in pixels
        r1 = max(0, row - r_size); r2 = min(mask.shape[0], row + r_size)
        c1 = max(0, col - r_size); c2 = min(mask.shape[1], col + r_size)
        mask[r1:r2, c1:c2] = class_id
        print(f"  Prompt: ({lon},{lat}) → class '{class_name}' (id={class_id}) at pixel ({row},{col})")
    return mask


# ── Colorized Visualization ───────────────────────────────────────────────────

def save_visualization(mask: np.ndarray, rgb_image: np.ndarray, output_path: Path):
    """Saves a side-by-side comparison: RGB image vs colorized prediction mask."""
    H, W = mask.shape

    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for class_id, bgr in CLASS_COLORS.items():
        rgb = bgr[::-1]   # convert BGR → RGB for matplotlib
        color_mask[mask == class_id] = rgb

    # Prepare RGB display (clip to 3 bands)
    if rgb_image.shape[0] >= 3:
        rgb_display = np.transpose(rgb_image[:3], (1, 2, 0))
    else:
        rgb_display = np.transpose(np.stack([rgb_image[0]]*3), (1, 2, 0))
    rgb_display = (rgb_display * 255).clip(0, 255).astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(rgb_display)
    ax1.set_title("Input (RGB bands)", fontsize=12)
    ax1.axis("off")

    ax2.imshow(color_mask)
    ax2.set_title("Predicted Mask", fontsize=12)
    ax2.axis("off")

    legend_elements = [
        Patch(facecolor=np.array(CLASS_COLORS[i][::-1])/255, label=CLASS_LABELS[i])
        for i in range(NUM_CLASSES)
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    viz_path = output_path.with_suffix(".viz.png")
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved: {viz_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def predict(input_tif: str, output_tif: str, visualize: bool = False, prompts: list = None):
    input_path  = Path(input_tif)
    output_path = Path(output_tif)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(CHECKPOINT_DIR) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Run 03_train.py first.")

    model, num_bands = load_model(ckpt_path)

    print(f"\nRunning inference on: {input_path.name}")
    with rasterio.open(input_path) as tif:
        print(f"  Size: {tif.width}×{tif.height}, Bands: {tif.count}, CRS: {tif.crs}")

        band_idx = BAND_INDICES if BAND_INDICES else list(range(1, tif.count + 1))
        # Use only as many bands as the model expects
        band_idx = band_idx[:num_bands]
        bands = tif.read(band_idx).astype(np.float32)
        bands = normalize_bands(bands)

        # Run sliding window
        print(f"  Sliding window inference (tile={TILE_SIZE}, overlap={INFERENCE_OVERLAP})...")
        class_mask = sliding_window_predict(model, bands)

        # Apply point prompts if provided
        if prompts:
            class_mask = apply_point_prompts(class_mask, prompts, tif)

        # Save GeoTIFF mask (preserves CRS + spatial transform)
        profile = tif.profile.copy()
        profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=255,
            compress="lzw"
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(class_mask[np.newaxis], 1)
            # Write class descriptions as band metadata
            dst.update_tags(1, **{
                f"class_{i}": label for i, label in enumerate(CLASS_LABELS)
            })

        print(f"  ✓ Mask saved: {output_path}")

        # Distribution
        unique, counts = np.unique(class_mask, return_counts=True)
        total = class_mask.size
        print("  Class distribution:")
        for u, c in zip(unique, counts):
            label = CLASS_LABELS[int(u)] if int(u) < len(CLASS_LABELS) else "unknown"
            print(f"    {label}: {c/total*100:.1f}%")

        if visualize:
            save_visualization(class_mask, bands, output_path)

    return class_mask


def parse_prompts(prompt_strs: list) -> list:
    """Parse 'lon,lat,class' strings into tuples."""
    prompts = []
    for s in prompt_strs:
        parts = s.split(",")
        if len(parts) != 3:
            print(f"  [WARN] Invalid prompt format '{s}' — expected 'lon,lat,classname'")
            continue
        prompts.append((float(parts[0]), float(parts[1]), parts[2].strip()))
    return prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoSeg inference")
    parser.add_argument("--input",      required=True,  help="Input GeoTIFF path")
    parser.add_argument("--output",     required=True,  help="Output mask GeoTIFF path")
    parser.add_argument("--visualize",  action="store_true", help="Save colorized viz PNG")
    parser.add_argument("--prompt",     nargs="*", default=[],
                        help="Point prompts as 'lon,lat,classname'. Can repeat.")
    args = parser.parse_args()

    prompts = parse_prompts(args.prompt) if args.prompt else None
    predict(args.input, args.output, visualize=args.visualize, prompts=prompts)
