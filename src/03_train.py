"""
03_train.py
───────────
Fine-tunes SegFormer on your geospatial tiles.

Key design choices:
- SegFormer-B2 backbone (good accuracy/speed tradeoff)
- Combined Dice + CrossEntropy loss (handles class imbalance)
- AdamW + cosine LR schedule
- Early stopping on val mIoU
- Saves best checkpoint + training curves
- Handles arbitrary number of input bands by patching the patch embedding
"""

import sys
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerConfig

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODEL_NAME, NUM_CLASSES, CLASS_LABELS, CLASS_WEIGHTS,
    NUM_EPOCHS, LR, WEIGHT_DECAY, PATIENCE,
    CHECKPOINT_DIR, DATA_PROCESSED_DIR, RANDOM_SEED
)
from dataset import build_dataloaders

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Model Builder ────────────────────────────────────────────────────────────

def build_model(num_input_bands: int) -> nn.Module:
    """
    Loads SegFormer and patches its patch embedding layer to accept
    `num_input_bands` channels instead of the default 3.
    """
    print(f"Loading {MODEL_NAME} with {num_input_bands} input band(s)...")

    # Load config and modify for our task
    cfg = SegformerConfig.from_pretrained(MODEL_NAME)
    cfg.num_labels    = NUM_CLASSES
    cfg.id2label      = {i: l for i, l in enumerate(CLASS_LABELS)}
    cfg.label2id      = {l: i for i, l in enumerate(CLASS_LABELS)}
    cfg.num_channels  = num_input_bands   # <-- key: tells SegFormer our band count

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        config=cfg,
        ignore_mismatched_sizes=True,   # allows head replacement
    )

    # If num_input_bands != 3, the pretrained patch embedding won't match.
    # We re-initialize it but copy weights for the RGB channels if possible.
    if num_input_bands != 3:
        old_embed = model.segformer.encoder.patch_embeddings[0].proj
        new_embed = nn.Conv2d(
            num_input_bands,
            old_embed.out_channels,
            kernel_size=old_embed.kernel_size,
            stride=old_embed.stride,
            padding=old_embed.padding,
            bias=old_embed.bias is not None,
        )
        # Initialize new embedding
        nn.init.kaiming_normal_(new_embed.weight)
        # Copy RGB weights for first 3 channels if available
        with torch.no_grad():
            channels_to_copy = min(3, num_input_bands)
            new_embed.weight[:, :channels_to_copy] = old_embed.weight[:, :channels_to_copy]
        model.segformer.encoder.patch_embeddings[0].proj = new_embed
        print(f"  Patched input embedding: 3 → {num_input_bands} channels (RGB weights preserved)")

    return model.to(DEVICE)


# ── Loss ─────────────────────────────────────────────────────────────────────

class DiceCELoss(nn.Module):
    """Combined Dice + CrossEntropy loss with class weights."""

    def __init__(self, class_weights=None, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=255)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W)  targets: (B, H, W)
        ce_loss = self.ce(logits, targets)

        # Dice loss
        probs   = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets.clamp(0, NUM_CLASSES-1), NUM_CLASSES).permute(0,3,1,2).float()
        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        union        = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice         = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        dice_loss    = dice.mean()

        return (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Mean Intersection over Union across all classes."""
    ious = []
    preds   = preds.view(-1)
    targets = targets.view(-1)
    for cls in range(num_classes):
        pred_c   = preds   == cls
        target_c = targets == cls
        intersection = (pred_c & target_c).sum().float()
        union        = (pred_c | target_c).sum().float()
        if union == 0:
            continue   # class not present — skip
        ious.append((intersection / union).item())
    return np.mean(ious) if ious else 0.0


# ── Train / Val Loop ─────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None, phase="train"):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss, total_miou, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(is_train):
        for images, masks in tqdm(loader, desc=f"  {phase}", leave=False):
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            outputs = model(pixel_values=images)
            logits  = outputs.logits   # (B, C, H/4, W/4) — SegFormer outputs at 1/4 scale

            # Upsample logits to full mask size
            logits_up = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss = criterion(logits_up, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            preds = logits_up.argmax(dim=1)
            miou  = compute_miou(preds.cpu(), masks.cpu(), NUM_CLASSES)

            total_loss += loss.item()
            total_miou += miou
            n_batches  += 1

    return total_loss / n_batches, total_miou / n_batches


# ── Main ─────────────────────────────────────────────────────────────────────

def train():
    ckpt_dir = Path(CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, num_bands = build_dataloaders()

    # Model
    model     = build_model(num_bands)
    criterion = DiceCELoss(class_weights=CLASS_WEIGHTS)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Training state
    best_miou     = 0.0
    patience_cnt  = 0
    history       = {"train_loss": [], "val_loss": [], "train_miou": [], "val_miou": []}

    print(f"\nStarting training — {NUM_EPOCHS} epochs, device={DEVICE}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_miou = run_epoch(model, train_loader, criterion, optimizer, "train")
        val_loss,   val_miou   = run_epoch(model, val_loader,   criterion, None,      "val")
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_mIoU={train_miou:.4f}  "
              f"val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  [{elapsed:.0f}s]")

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_miou"].append(train_miou)
        history["val_miou"].append(val_miou)

        # Save best checkpoint
        if val_miou > best_miou:
            best_miou    = val_miou
            patience_cnt = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_miou":   val_miou,
                "num_bands":  num_bands,
            }, ckpt_dir / "best_model.pt")
            print(f"  ✓ Saved best checkpoint (val_mIoU={best_miou:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # Save training curves
    _plot_history(history, ckpt_dir / "training_curves.png")
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Training complete. Best val_mIoU: {best_miou:.4f}")
    print(f"  Checkpoint: {ckpt_dir / 'best_model.pt'}")


def _plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, history["train_miou"], label="Train")
    ax2.plot(epochs, history["val_miou"],   label="Val")
    ax2.set_title("mIoU")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Training curves saved to {save_path}")


if __name__ == "__main__":
    train()
