"""
03_train.py — Fine-tunes SegFormer on geospatial tiles.
Supports incremental training: pass --resume to continue from last checkpoint.
When resuming, loads weights from best_model.pt and trains with lower LR.
"""

import sys
import json
import time
import argparse
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
import importlib
dataset = importlib.import_module("02_dataset")
build_dataloaders = dataset.build_dataloaders

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_input_bands, checkpoint_path=None):
    print(f"Loading {MODEL_NAME} with {num_input_bands} band(s)...", flush=True)

    cfg = SegformerConfig.from_pretrained(MODEL_NAME)
    cfg.num_labels   = NUM_CLASSES
    cfg.id2label     = {i: l for i, l in enumerate(CLASS_LABELS)}
    cfg.label2id     = {l: i for i, l in enumerate(CLASS_LABELS)}
    cfg.num_channels = num_input_bands

    if checkpoint_path and checkpoint_path.exists():
        # Incremental training — load our saved weights, skip HuggingFace download
        print(f"  Resuming from checkpoint: {checkpoint_path}", flush=True)
        model = SegformerForSemanticSegmentation(cfg)
        if num_input_bands != 3:
            _patch_embedding(model, num_input_bands)
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded checkpoint (epoch={ckpt['epoch']}, val_mIoU={ckpt['val_miou']:.4f})", flush=True)
    else:
        # Fresh training — download pretrained weights
        model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME, config=cfg, ignore_mismatched_sizes=True
        )
        if num_input_bands != 3:
            _patch_embedding(model, num_input_bands)

    return model.to(DEVICE)


def _patch_embedding(model, num_input_bands):
    old = model.segformer.encoder.patch_embeddings[0].proj
    new = nn.Conv2d(num_input_bands, old.out_channels,
                    kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=old.bias is not None)
    nn.init.kaiming_normal_(new.weight)
    with torch.no_grad():
        n = min(3, num_input_bands)
        new.weight[:, :n] = old.weight[:, :n]
    model.segformer.encoder.patch_embeddings[0].proj = new
    print(f"  Patched input embedding → {num_input_bands} channels", flush=True)


# ── Loss ──────────────────────────────────────────────────────────────────────

class DiceCELoss(nn.Module):
    def __init__(self, class_weights=None, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        w = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=w, ignore_index=255)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        probs   = F.softmax(logits, dim=1)
        oh      = F.one_hot(targets.clamp(0, NUM_CLASSES-1), NUM_CLASSES).permute(0,3,1,2).float()
        inter   = (probs * oh).sum(dim=(2,3))
        union   = probs.sum(dim=(2,3)) + oh.sum(dim=(2,3))
        dice    = (1 - (2*inter+1e-6)/(union+1e-6)).mean()
        return (1-self.dice_weight)*ce_loss + self.dice_weight*dice


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_miou(preds, targets, num_classes):
    ious = []
    p, t = preds.view(-1), targets.view(-1)
    for cls in range(num_classes):
        inter = ((p==cls) & (t==cls)).sum().float()
        union = ((p==cls) | (t==cls)).sum().float()
        if union > 0:
            ious.append((inter/union).item())
    return np.mean(ious) if ious else 0.0


# ── Epoch ─────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None, phase="train"):
    is_train = phase == "train"
    model.train() if is_train else model.eval()
    total_loss = total_miou = n = 0

    with torch.set_grad_enabled(is_train):
        for images, masks in tqdm(loader, desc=f"  {phase}", leave=False):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits_up = F.interpolate(
                model(pixel_values=images).logits,
                size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = criterion(logits_up, masks)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            total_miou += compute_miou(logits_up.argmax(1).cpu(), masks.cpu(), NUM_CLASSES)
            n += 1

    return total_loss/n, total_miou/n


# ── Train ─────────────────────────────────────────────────────────────────────

def train(resume=False):
    ckpt_dir  = Path(CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"

    train_loader, val_loader, num_bands = build_dataloaders()

    # Use lower LR when resuming (fine-tuning on new data)
    lr = LR * 0.3 if resume and ckpt_path.exists() else LR

    model     = build_model(num_bands, ckpt_path if resume else None)
    criterion = DiceCELoss(class_weights=CLASS_WEIGHTS)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_miou    = 0.0
    patience_cnt = 0
    history      = {"train_loss": [], "val_loss": [], "train_miou": [], "val_miou": []}

    mode = "RESUMING (incremental)" if resume and ckpt_path.exists() else "FRESH"
    print(f"\n{'='*50}", flush=True)
    print(f"Training mode: {mode} | lr={lr:.2e} | epochs={NUM_EPOCHS} | device={DEVICE}", flush=True)
    print(f"{'='*50}\n", flush=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_miou = run_epoch(model, train_loader, criterion, optimizer, "train")
        val_loss,   val_miou   = run_epoch(model, val_loader,   criterion, None,      "val")
        scheduler.step()

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_mIoU={train_miou:.4f}  "
              f"val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  [{time.time()-t0:.0f}s]", flush=True)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_miou"].append(train_miou)
        history["val_miou"].append(val_miou)

        if val_miou > best_miou:
            best_miou    = val_miou
            patience_cnt = 0
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_miou":        val_miou,
                "num_bands":       num_bands,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_mIoU={best_miou:.4f})", flush=True)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}", flush=True)
                break

    # Save curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    e = range(1, len(history["train_loss"])+1)
    ax1.plot(e, history["train_loss"], label="Train"); ax1.plot(e, history["val_loss"], label="Val")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(e, history["train_miou"], label="Train"); ax2.plot(e, history["val_miou"], label="Val")
    ax2.set_title("mIoU"); ax2.legend()
    plt.tight_layout()
    plt.savefig(ckpt_dir / "training_curves.png", dpi=120)
    plt.close()

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Done. Best val_mIoU={best_miou:.4f} | Checkpoint: {ckpt_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint (incremental training)")
    args = parser.parse_args()
    train(resume=args.resume)