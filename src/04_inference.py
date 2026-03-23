"""
04_inference.py
───────────────
Sliding-window inference with Gaussian blending + Test-Time Augmentation (TTA).

TTA averages predictions from 8 augmented versions of each tile:
  original, H-flip, V-flip, HV-flip, 90°, 90°+H, 270°, 180°
This typically adds +1-3% mIoU over single-pass inference at the cost of ~8×
compute.  Disable with --no-tta for speed.

Usage:
    python src/04_inference.py --input data/raw/test.tif --output outputs/mask.tif
    python src/04_inference.py --input test.tif --output mask.tif --visualize
    python src/04_inference.py --input test.tif --output mask.tif --no-tta
    python src/04_inference.py --input test.tif --output mask.tif \\
        --prompt 12.34,56.78,road --prompt 12.40,56.80,waterbody
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
    BAND_INDICES, CHECKPOINT_DIR, OUTPUT_DIR, USE_TTA,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt      = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    num_bands = ckpt["num_bands"]

    cfg = SegformerConfig.from_pretrained(MODEL_NAME)
    cfg.num_labels   = NUM_CLASSES
    cfg.id2label     = {i: l for i, l in enumerate(CLASS_LABELS)}
    cfg.label2id     = {l: i for i, l in enumerate(CLASS_LABELS)}
    cfg.num_channels = num_bands

    model = SegformerForSemanticSegmentation(cfg)

    if num_bands != 3:
        import torch.nn as nn
        old = model.segformer.encoder.patch_embeddings[0].proj
        new = nn.Conv2d(num_bands, old.out_channels,
                        kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, bias=old.bias is not None)
        model.segformer.encoder.patch_embeddings[0].proj = new

    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    print(f"  Loaded — epoch={ckpt['epoch']}  val_mIoU={ckpt['val_miou']:.4f}  bands={num_bands}")

    if "per_class_iou" in ckpt:
        pc = ckpt["per_class_iou"]
        print("  Checkpoint per-class IoU:")
        for cid, iou in pc.items():
            label = CLASS_LABELS[int(cid)] if int(cid) < len(CLASS_LABELS) else f"cls{cid}"
            print(f"    {label}: {iou:.4f}")

    return model, num_bands


# ── Gaussian blending window ──────────────────────────────────────────────────

def gaussian_window(size: int, sigma_ratio: float = 4.0) -> np.ndarray:
    sigma  = size / sigma_ratio
    coords = np.arange(size) - size // 2
    g1d    = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g2d    = np.outer(g1d, g1d)
    return (g2d / g2d.max()).astype(np.float32)


# ── Preprocessing ─────────────────────────────────────────────────────────────
# Must match 01_preprocess.py (p2/p98 clip) + 02_dataset.py (ImageNet mean/std).

_IN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IN_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def normalize_bands(bands: np.ndarray) -> np.ndarray:
    """Stage 1: p2/p98 clip → [0,1].  Stage 2: ImageNet mean/std on bands 0-2."""
    bands = bands.astype(np.float32)
    for i in range(bands.shape[0]):
        lo, hi   = np.percentile(bands[i], 2), np.percentile(bands[i], 98)
        bands[i] = np.clip((bands[i] - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    n = min(3, bands.shape[0])
    bands[:n] = (bands[:n] - _IN_MEAN[:n]) / _IN_STD[:n]
    return bands


# ── TTA augment / de-augment pairs ───────────────────────────────────────────

# Each entry: (augment_fn, deaugment_fn)
# augment applies to (C, H, W) image arrays
# deaugment applies to (C, H, W) probability arrays
_TTA_OPS = [
    # identity
    (lambda x: x,
     lambda x: x),
    # horizontal flip
    (lambda x: x[:, :, ::-1].copy(),
     lambda x: x[:, :, ::-1].copy()),
    # vertical flip
    (lambda x: x[:, ::-1, :].copy(),
     lambda x: x[:, ::-1, :].copy()),
    # H + V flip
    (lambda x: x[:, ::-1, ::-1].copy(),
     lambda x: x[:, ::-1, ::-1].copy()),
    # 90° CCW  → deaugment = 90° CW
    (lambda x: np.rot90(x, k=1, axes=(1, 2)).copy(),
     lambda x: np.rot90(x, k=-1, axes=(1, 2)).copy()),
    # 90° CCW + H-flip
    (lambda x: np.rot90(x, k=1, axes=(1, 2))[:, :, ::-1].copy(),
     lambda x: np.rot90(x[:, :, ::-1], k=-1, axes=(1, 2)).copy()),
    # 180°
    (lambda x: np.rot90(x, k=2, axes=(1, 2)).copy(),
     lambda x: np.rot90(x, k=-2, axes=(1, 2)).copy()),
    # 270° CCW  → deaugment = 270° CW
    (lambda x: np.rot90(x, k=3, axes=(1, 2)).copy(),
     lambda x: np.rot90(x, k=-3, axes=(1, 2)).copy()),
]


def _batch_infer(model, tiles: list) -> np.ndarray:
    """Run model on a list of (C, H, W) numpy tiles; returns (B, NUM_CLASSES, H, W) probabilities."""
    tensor = torch.from_numpy(np.stack(tiles)).to(DEVICE)
    with torch.no_grad():
        out    = model(pixel_values=tensor)
        logits = F.interpolate(out.logits, size=(TILE_SIZE, TILE_SIZE),
                               mode="bilinear", align_corners=False)
        return torch.softmax(logits, dim=1).cpu().numpy()


# ── Sliding window inference ──────────────────────────────────────────────────

def sliding_window_predict(model, image: np.ndarray, use_tta: bool = True) -> np.ndarray:
    """
    image: (C, H, W) float32 normalised
    Returns: (H, W) uint8 class mask
    """
    C, H, W  = image.shape
    stride   = TILE_SIZE - INFERENCE_OVERLAP
    win      = gaussian_window(TILE_SIZE)

    logit_sum  = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    # Build tile list
    tiles, positions = [], []
    for r in range(0, max(H - TILE_SIZE + 1, 1), stride):
        for c in range(0, max(W - TILE_SIZE + 1, 1), stride):
            r2 = min(r + TILE_SIZE, H); r1 = r2 - TILE_SIZE
            c2 = min(c + TILE_SIZE, W); c1 = c2 - TILE_SIZE
            tiles.append(image[:, r1:r2, c1:c2])
            positions.append((r1, r2, c1, c2))

    tta_ops = _TTA_OPS if use_tta else [_TTA_OPS[0]]

    # For each TTA variant, run batched inference
    for aug_fn, deaug_fn in tqdm(tta_ops,
                                  desc=f"  TTA ({len(tta_ops)} variants)",
                                  disable=(not use_tta)):
        aug_tiles = [aug_fn(t) for t in tiles]

        for i in tqdm(range(0, len(aug_tiles), INFERENCE_BATCH),
                      desc="    Batches", leave=False):
            batch_tiles = aug_tiles[i:i + INFERENCE_BATCH]
            batch_pos   = positions[i:i + INFERENCE_BATCH]
            probs       = _batch_infer(model, batch_tiles)   # (B, C, H, W)

            for j, (r1, r2, c1, c2) in enumerate(batch_pos):
                deaug_probs = deaug_fn(probs[j])             # (C, H, W)
                logit_sum[:, r1:r2, c1:c2] += deaug_probs * win[np.newaxis]
                weight_sum[r1:r2, c1:c2]   += win

    weight_sum = np.maximum(weight_sum, 1e-6)
    prob_map   = logit_sum / weight_sum[np.newaxis]
    return prob_map.argmax(axis=0).astype(np.uint8)


# ── Point prompt override ─────────────────────────────────────────────────────

def apply_point_prompts(mask, prompts, tif):
    mask = mask.copy()
    for lon, lat, class_name in prompts:
        if class_name not in CLASSES:
            print(f"  [WARN] Unknown class '{class_name}' — skipping")
            continue
        row, col     = rowcol(tif.transform, lon, lat)
        row, col     = int(row), int(col)
        class_id     = CLASSES[class_name]
        r_size       = 5
        r1 = max(0, row - r_size); r2 = min(mask.shape[0], row + r_size)
        c1 = max(0, col - r_size); c2 = min(mask.shape[1], col + r_size)
        mask[r1:r2, c1:c2] = class_id
        print(f"  Prompt ({lon},{lat}) → '{class_name}' id={class_id} px=({row},{col})")
    return mask


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_visualization(mask, rgb_image, output_path):
    H, W = mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for class_id, bgr in CLASS_COLORS.items():
        color_mask[mask == class_id] = bgr[::-1]   # BGR → RGB

    if rgb_image.shape[0] >= 3:
        rgb_disp = np.transpose(rgb_image[:3], (1, 2, 0))
    else:
        rgb_disp = np.transpose(np.stack([rgb_image[0]] * 3), (1, 2, 0))
    rgb_disp = (rgb_disp * 255).clip(0, 255).astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(rgb_disp);    ax1.set_title("Input (RGB bands)"); ax1.axis("off")
    ax2.imshow(color_mask);  ax2.set_title("Predicted Mask");    ax2.axis("off")

    legend = [Patch(facecolor=np.array(CLASS_COLORS[i][::-1]) / 255, label=CLASS_LABELS[i])
              for i in range(NUM_CLASSES)]
    ax2.legend(handles=legend, loc="lower right", fontsize=9)
    plt.tight_layout()
    viz_path = output_path.with_suffix(".viz.png")
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved: {viz_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def predict(input_tif, output_tif, visualize=False, prompts=None, use_tta=None):
    if use_tta is None:
        use_tta = USE_TTA

    input_path  = Path(input_tif)
    output_path = Path(output_tif)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(CHECKPOINT_DIR) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Run 03_train.py first.")

    model, num_bands = load_model(ckpt_path)

    print(f"\nRunning inference on: {input_path.name}  (TTA={use_tta})")
    with rasterio.open(input_path) as tif:
        print(f"  Size: {tif.width}×{tif.height}  Bands: {tif.count}  CRS: {tif.crs}")

        band_idx = (BAND_INDICES if BAND_INDICES else list(range(1, tif.count + 1)))[:num_bands]
        bands    = normalize_bands(tif.read(band_idx).astype(np.float32))

        print(f"  Sliding window (tile={TILE_SIZE}, overlap={INFERENCE_OVERLAP})...")
        class_mask = sliding_window_predict(model, bands, use_tta=use_tta)

        if prompts:
            class_mask = apply_point_prompts(class_mask, prompts, tif)

        # Save georeferenced mask
        profile = tif.profile.copy()
        profile.update(count=1, dtype=rasterio.uint8, nodata=255, compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(class_mask[np.newaxis], 1)
            dst.update_tags(1, **{f"class_{i}": l for i, l in enumerate(CLASS_LABELS)})

        print(f"  ✓ Mask saved: {output_path}")

        unique, counts = np.unique(class_mask, return_counts=True)
        total          = class_mask.size
        print("  Class distribution:")
        for u, c in zip(unique, counts):
            label = CLASS_LABELS[int(u)] if int(u) < len(CLASS_LABELS) else "unknown"
            print(f"    {label}: {c/total*100:.1f}%")

        if visualize:
            save_visualization(class_mask, bands, output_path)

    return class_mask


def parse_prompts(prompt_strs):
    prompts = []
    for s in prompt_strs:
        parts = s.split(",")
        if len(parts) != 3:
            print(f"  [WARN] Invalid prompt '{s}' — expected 'lon,lat,classname'")
            continue
        prompts.append((float(parts[0]), float(parts[1]), parts[2].strip()))
    return prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoSeg inference")
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--no-tta",    action="store_true", help="Disable TTA for speed")
    parser.add_argument("--prompt",    nargs="*", default=[])
    args = parser.parse_args()

    prompts = parse_prompts(args.prompt) if args.prompt else None
    predict(args.input, args.output,
            visualize=args.visualize,
            prompts=prompts,
            use_tta=not args.no_tta)