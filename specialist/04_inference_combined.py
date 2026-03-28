"""
Combined inference: generalist model + specialist overlay.

The generalist predicts all 7 classes.
The specialist (trained on minor-class-focused data) overrides
Bridge / Railway / Utility pixels wherever its confidence exceeds
the per-class threshold defined in config_specialist.

Usage:
    python specialist/04_inference_combined.py --input data/raw/test.tif --output outputs/combined_mask.tif
    python specialist/04_inference_combined.py --input test.tif --output mask.tif --visualize
    python specialist/04_inference_combined.py --input test.tif --output mask.tif --no-tta
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch.nn as nn

# ── paths ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from config_specialist import (
    NUM_CLASSES, CLASS_LABELS, CLASS_COLORS, CLASSES,
    TILE_SIZE, INFERENCE_OVERLAP, INFERENCE_BATCH,
    BAND_INDICES, USE_TTA,
    MINOR_CLASSES, CLASS_THRESHOLDS,
    CHECKPOINT_DIR as SPECIALIST_CKPT_DIR,
)
# generalist checkpoint dir comes from the main config
import importlib as _il
_main_cfg = _il.import_module("config")
GENERALIST_CKPT_DIR = _main_cfg.CHECKPOINT_DIR
GENERALIST_MODEL    = _main_cfg.MODEL_NAME     # "nvidia/mit-b5"
SPECIALIST_MODEL    = "nvidia/mit-b2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── normalisation (must match 01_preprocess → 02_dataset) ─────────────────
IN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IN_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def normalize_bands(bands: np.ndarray) -> np.ndarray:
    bands = bands.astype(np.float32)
    for i in range(bands.shape[0]):
        lo, hi = np.percentile(bands[i], 2), np.percentile(bands[i], 98)
        bands[i] = np.clip((bands[i] - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    n = min(3, bands.shape[0])
    bands[:n] = (bands[:n] - IN_MEAN[:n]) / IN_STD[:n]
    return bands


# ── model loading ──────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, backbone_name: str):
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    ckpt      = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    num_bands = ckpt["num_bands"]

    # BUG FIX: 03_train_specialist.py saves "model_name" in the checkpoint.
    # Prefer that over the caller-supplied name so the correct SegFormer
    # variant is always reconstructed, even if MODEL_NAME config changes.
    backbone_name = ckpt.get("model_name", backbone_name)

    cfg              = SegformerConfig.from_pretrained(backbone_name)
    cfg.num_labels   = NUM_CLASSES
    cfg.id2label     = {i: l for i, l in enumerate(CLASS_LABELS)}
    cfg.label2id     = {l: i for i, l in enumerate(CLASS_LABELS)}
    cfg.num_channels = num_bands

    model = SegformerForSemanticSegmentation(cfg)
    if num_bands != 3:
        old = model.segformer.encoder.patch_embeddings[0].proj
        new = nn.Conv2d(
            num_bands, old.out_channels,
            kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, bias=old.bias is not None,
        )
        model.segformer.encoder.patch_embeddings[0].proj = new

    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    print(
        f"  backbone={backbone_name}  epoch={ckpt.get('epoch','?')}  "
        f"val_mIoU={ckpt.get('val_miou', 0):.4f}  bands={num_bands}",
        flush=True,
    )
    return model, num_bands


# ── Gaussian blending window ───────────────────────────────────────────────
def gaussian_window(size: int, sigma_ratio: float = 4.0) -> np.ndarray:
    sigma  = size / sigma_ratio
    coords = np.arange(size) - size / 2
    g1d    = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g2d    = np.outer(g1d, g1d)
    return (g2d / g2d.max()).astype(np.float32)


# ── TTA ops ────────────────────────────────────────────────────────────────
TTA_OPS = [
    (lambda x: x,                          lambda x: x),                          # identity
    (lambda x: x[:, :, ::-1].copy(),       lambda x: x[:, :, ::-1].copy()),       # H-flip
    (lambda x: x[:, ::-1, :].copy(),       lambda x: x[:, ::-1, :].copy()),       # V-flip
    (lambda x: x[:, ::-1, ::-1].copy(),    lambda x: x[:, ::-1, ::-1].copy()),    # HV-flip
    (lambda x: np.rot90(x, k=1, axes=(1,2)).copy(), lambda x: np.rot90(x, k=-1, axes=(1,2)).copy()),  # 90°
    (lambda x: np.rot90(x, k=2, axes=(1,2)).copy(), lambda x: np.rot90(x, k=-2, axes=(1,2)).copy()),  # 180°
    (lambda x: np.rot90(x, k=3, axes=(1,2)).copy(), lambda x: np.rot90(x, k=-3, axes=(1,2)).copy()),  # 270°
    (lambda x: np.rot90(x[:, :, ::-1].copy(), k=1, axes=(1,2)).copy(),
     lambda x: np.rot90(x, k=-1, axes=(1,2))[:, :, ::-1].copy()),                 # 90° + H-flip
]


# ── batched inference ──────────────────────────────────────────────────────
def batch_infer(model, tiles: list) -> np.ndarray:
    tensor = torch.from_numpy(np.stack(tiles)).to(DEVICE)
    with torch.no_grad():
        out    = model(pixel_values=tensor)
        logits = F.interpolate(
            out.logits, size=(TILE_SIZE, TILE_SIZE),
            mode="bilinear", align_corners=False,
        )
    return torch.softmax(logits, dim=1).cpu().numpy()   # [B, C, H, W]


# ── sliding window → raw probability map ──────────────────────────────────
def sliding_window_probs(model, image: np.ndarray, use_tta: bool = True) -> np.ndarray:
    """
    Returns [NUM_CLASSES, H, W] float32 probability map (NOT argmax).
    This is what both models expose so the overlay can compare confidences.
    """
    C, H, W  = image.shape
    stride   = TILE_SIZE - INFERENCE_OVERLAP
    win      = gaussian_window(TILE_SIZE)
    logit_sum  = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W),             dtype=np.float32)

    tiles, positions = [], []
    for r in range(0, max(H - TILE_SIZE + 1, 1), stride):
        for c in range(0, max(W - TILE_SIZE + 1, 1), stride):
            r2 = min(r + TILE_SIZE, H); r1 = r2 - TILE_SIZE
            c2 = min(c + TILE_SIZE, W); c1 = c2 - TILE_SIZE
            tiles.append(image[:, r1:r2, c1:c2])
            positions.append((r1, r2, c1, c2))

    tta_ops = TTA_OPS if use_tta else TTA_OPS[:1]

    for aug_fn, deaug_fn in tqdm(tta_ops, desc=f"  TTA {len(tta_ops)} variants", disable=not use_tta):
        aug_tiles = [aug_fn(t) for t in tiles]
        for i in tqdm(range(0, len(aug_tiles), INFERENCE_BATCH), desc="  Batches", leave=False):
            batch_tiles = aug_tiles[i: i + INFERENCE_BATCH]
            batch_pos   = positions[i: i + INFERENCE_BATCH]
            probs       = batch_infer(model, batch_tiles)       # [B, C, H, W]
            for j, (r1, r2, c1, c2) in enumerate(batch_pos):
                dp = deaug_fn(probs[j])                         # [C, H, W]
                logit_sum[:, r1:r2, c1:c2]  += dp * win[np.newaxis]
                weight_sum[r1:r2, c1:c2]    += win

    weight_sum = np.maximum(weight_sum, 1e-6)
    return logit_sum / weight_sum[np.newaxis]                    # [C, H, W]


# ── visualisation ──────────────────────────────────────────────────────────
def save_visualization(mask, rgb_image, output_path, title="Combined Prediction"):
    H, W      = mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, bgr in CLASS_COLORS.items():
        color_mask[mask == cls_id] = bgr[::-1]   # BGR → RGB

    if rgb_image.shape[0] >= 3:
        rgb_disp = np.transpose(rgb_image[:3], (1, 2, 0))
    else:
        rgb_disp = np.transpose(np.stack([rgb_image[0]] * 3), (1, 2, 0))
    rgb_disp = (rgb_disp * 255).clip(0, 255).astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(rgb_disp);    ax1.set_title("Input RGB");   ax1.axis("off")
    ax2.imshow(color_mask);  ax2.set_title(title);         ax2.axis("off")
    legend = [
        Patch(facecolor=np.array(CLASS_COLORS[i][::-1]) / 255, label=CLASS_LABELS[i])
        for i in range(NUM_CLASSES)
    ]
    ax2.legend(handles=legend, loc="lower right", fontsize=9)
    plt.tight_layout()
    viz_path = output_path.with_suffix(".viz.png")
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved: {viz_path}", flush=True)


# ── combined predict ───────────────────────────────────────────────────────
def predict_combined(
    input_tif: str,
    output_tif: str,
    visualize: bool = False,
    use_tta: bool   = True,
):
    input_path  = Path(input_tif)
    output_path = Path(output_tif)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gen_ckpt  = Path(GENERALIST_CKPT_DIR)  / "best_model.pt"
    spec_ckpt = Path(SPECIALIST_CKPT_DIR)  / "best_model.pt"

    if not gen_ckpt.exists():
        raise FileNotFoundError(f"Generalist checkpoint not found: {gen_ckpt}")
    if not spec_ckpt.exists():
        raise FileNotFoundError(
            f"Specialist checkpoint not found: {spec_ckpt}\n"
            f"Run: python specialist/03_train_specialist.py"
        )

    print(f"\n{'='*55}", flush=True)
    print(f"[Generalist]", flush=True)
    model_g, num_bands = load_model(gen_ckpt,  GENERALIST_MODEL)
    print(f"\n[Specialist]", flush=True)
    model_s, _         = load_model(spec_ckpt, SPECIALIST_MODEL)
    print(f"{'='*55}\n", flush=True)

    with rasterio.open(input_path) as tif:
        print(f"Input: {input_path.name}  {tif.width}×{tif.height}  "
              f"bands={tif.count}  CRS={tif.crs}", flush=True)
        band_idx = BAND_INDICES if BAND_INDICES else list(range(1, tif.count + 1))
        bands    = normalize_bands(tif.read(band_idx).astype(np.float32))
        profile  = tif.profile.copy()

    print(f"Sliding window  tile={TILE_SIZE}  overlap={INFERENCE_OVERLAP}  TTA={use_tta}", flush=True)

    # ── generalist pass ────────────────────────────────────────────────────
    print("\n[1/2] Generalist inference...", flush=True)
    gen_probs  = sliding_window_probs(model_g, bands, use_tta)  # [7, H, W]

    # ── specialist pass ────────────────────────────────────────────────────
    print("\n[2/2] Specialist inference...", flush=True)
    spec_probs = sliding_window_probs(model_s, bands, use_tta)  # [7, H, W]

    # ── merge ──────────────────────────────────────────────────────────────
    final_mask = gen_probs.argmax(axis=0).astype(np.uint8)   # start from generalist

    n_overrides = 0
    for cls in MINOR_CLASSES:                                  # [4, 5, 6]
        threshold   = CLASS_THRESHOLDS.get(cls, 0.45)
        override    = spec_probs[cls] > threshold
        n_overrides += int(override.sum())
        final_mask[override] = cls

    H, W    = final_mask.shape
    total_px = H * W
    print(f"\nOverlay stats:", flush=True)
    print(f"  Specialist overrode {n_overrides:,} px  "
          f"({n_overrides / total_px * 100:.2f}% of image)", flush=True)
    for cls in MINOR_CLASSES:
        n = int((final_mask == cls).sum())
        print(f"  {CLASS_LABELS[cls]:12s}  {n:>8,} px  ({n/total_px*100:.2f}%)", flush=True)

    # ── class distribution ─────────────────────────────────────────────────
    print("\nFinal class distribution:", flush=True)
    unique, counts = np.unique(final_mask, return_counts=True)
    for u, c in zip(unique, counts):
        lbl = CLASS_LABELS[int(u)] if int(u) < len(CLASS_LABELS) else "unknown"
        print(f"  {lbl:12s}  {c/total_px*100:.1f}%", flush=True)

    # ── save GeoTIFF ───────────────────────────────────────────────────────
    profile.update(count=1, dtype=rasterio.uint8, nodata=255, compress="lzw")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(final_mask[np.newaxis], 1)
        dst.update_tags(1, **{f"class{i}": l for i, l in enumerate(CLASS_LABELS)})
    print(f"\n[OK] Mask saved → {output_path}", flush=True)

    if visualize:
        save_visualization(final_mask, bands, output_path)

    return final_mask


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined generalist + specialist inference")
    parser.add_argument("--input",      required=True,  help="Input GeoTIFF")
    parser.add_argument("--output",     required=True,  help="Output mask GeoTIFF")
    parser.add_argument("--visualize",  action="store_true")
    parser.add_argument("--no-tta",     action="store_true", help="Disable TTA for speed")
    args = parser.parse_args()

    predict_combined(
        input_tif=args.input,
        output_tif=args.output,
        visualize=args.visualize,
        use_tta=not args.no_tta,
    )