"""
Train the specialist model (Bridge / Railway / Utility).

Identical logic to src/03_train.py.  Only differences:
  - imports from config_specialist  (mit-b2, specialist CHECKPOINT_DIR, etc.)
  - loads data via 02_dataset_specialist  (tiles_meta_specialist.json)

Usage:
    python specialist/03_train_specialist.py
    python specialist/03_train_specialist.py --resume
"""

import sys
import json
import time
import signal
import copy
import threading
import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerConfig

# ── paths ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from config_specialist import (
    MODEL_NAME, NUM_CLASSES, CLASS_LABELS, CLASS_WEIGHTS,
    NUM_EPOCHS, LR, WEIGHT_DECAY, PATIENCE,
    CHECKPOINT_DIR, DATA_PROCESSED_DIR, RANDOM_SEED,
    USE_AMP, GRAD_ACCUM_STEPS, WARMUP_EPOCHS,
    USE_EMA, EMA_DECAY,
    FOCAL_GAMMA, FOCAL_WEIGHT, DICE_WEIGHT,
    NOTIFY_INTERVAL_HOURS, BAND_INDICES,
)

dataset           = importlib.import_module("02_dataset_specialist")
build_dataloaders = dataset.build_dataloaders

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)


# ── model ──────────────────────────────────────────────────────────────────
def build_model(num_input_bands, checkpoint_path=None, init_weights_path=None):
    print(f"Loading {MODEL_NAME} with {num_input_bands} bands...", flush=True)
    cfg             = SegformerConfig.from_pretrained(MODEL_NAME)
    cfg.num_labels  = NUM_CLASSES
    cfg.id2label    = {i: l for i, l in enumerate(CLASS_LABELS)}
    cfg.label2id    = {l: i for i, l in enumerate(CLASS_LABELS)}
    cfg.num_channels = num_input_bands

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Resuming from {checkpoint_path}", flush=True)
        model = SegformerForSemanticSegmentation(cfg)
        if num_input_bands != 3:
            _patch_embedding(model, num_input_bands)
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded epoch={ckpt['epoch']}  val_mIoU={ckpt['val_miou']:.4f}", flush=True)
    elif init_weights_path and Path(init_weights_path).exists():
        print(f"Init weights from {init_weights_path} — fresh training state", flush=True)
        model = SegformerForSemanticSegmentation(cfg)
        if num_input_bands != 3:
            _patch_embedding(model, num_input_bands)
        ckpt  = torch.load(init_weights_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {len(missing)}", flush=True)
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME, config=cfg, ignore_mismatched_sizes=True
        )
        if num_input_bands != 3:
            _patch_embedding(model, num_input_bands)

    return model.to(DEVICE)


def _patch_embedding(model, num_input_bands):
    old = model.segformer.encoder.patch_embeddings[0].proj
    new = nn.Conv2d(
        num_input_bands, old.out_channels,
        kernel_size=old.kernel_size, stride=old.stride,
        padding=old.padding, bias=old.bias is not None,
    )
    nn.init.kaiming_normal_(new.weight)
    with torch.no_grad():
        n = min(3, num_input_bands)
        new.weight[:, :n] = old.weight[:, :n]
    model.segformer.encoder.patch_embeddings[0].proj = new
    print(f"Patched input embedding → {num_input_bands} channels", flush=True)


# ── EMA ────────────────────────────────────────────────────────────────────
class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.ema   = copy.deepcopy(model).eval()
        self.decay = decay
        self.step  = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for ema_p, mp in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(mp.data, alpha=1.0 - d)

    def state_dict(self):
        return self.ema.state_dict()


# ── loss ───────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None, label_smoothing=0.1):
        super().__init__()
        self.gamma          = gamma
        self.label_smoothing = label_smoothing
        self.w = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE) \
                 if class_weights else None

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, weight=self.w,
            reduction="none", ignore_index=255,
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class DiceLoss(nn.Module):
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        oh    = F.one_hot(targets.clamp(0, NUM_CLASSES - 1), NUM_CLASSES) \
                  .permute(0, 3, 1, 2).float()
        inter = (probs * oh).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
        dice  = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
        return dice[:, 1:].mean()   # skip background


class FocalDiceLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.focal = FocalLoss(FOCAL_GAMMA, class_weights)
        self.dice  = DiceLoss()

    def forward(self, logits, targets):
        return FOCAL_WEIGHT * self.focal(logits, targets) \
             + DICE_WEIGHT  * self.dice(logits, targets)


# ── scheduler ──────────────────────────────────────────────────────────────
def build_scheduler(optimizer, num_epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return max(1e-2, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ── metrics ────────────────────────────────────────────────────────────────
def compute_miou(preds, targets, num_classes):
    p, t  = preds.view(-1).cpu(), targets.view(-1).cpu()
    ious  = {}
    for cls in range(num_classes):
        inter = ((p == cls) & (t == cls)).sum().float()
        union = ((p == cls) | (t == cls)).sum().float()
        if union > 0:
            ious[cls] = (inter / union).item()
    mean_iou = float(np.mean(list(ious.values()))) if ious else 0.0
    return mean_iou, ious


# ── shared state for notification thread ───────────────────────────────────
class TrainingState:
    def __init__(self):
        self.lock        = threading.Lock()
        self.epoch       = 0
        self.train_loss  = 0.0
        self.val_loss    = 0.0
        self.train_miou  = 0.0
        self.val_miou    = 0.0
        self.best_miou   = 0.0
        self.folder_name = "specialist"
        self.folder_step = 1
        self.folder_total = 1
        self.done        = False

    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self):
        with self.lock:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("lock")}


state = TrainingState()

_crash_fired = False


def crash_handler(signum, frame):
    global _crash_fired
    if _crash_fired:
        sys.exit(1)
    _crash_fired = True
    signal.signal(signal.SIGINT,  signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    state.update(done=True)
    snap = state.snapshot()
    print(
        f"\n[SPECIALIST] Signal {signum} at epoch {snap['epoch']}. "
        f"Best val_mIoU={snap['best_miou']:.4f}  "
        f"Checkpoint safe at {CHECKPOINT_DIR}/best_model.pt",
        flush=True,
    )
    sys.exit(0 if signum == signal.SIGINT else 1)


# ── epoch ──────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, scaler=None,
              phase="train", grad_accum=1):
    is_train = phase == "train"
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    n = 0
    if is_train and optimizer:
        optimizer.zero_grad()

    with torch.set_grad_enabled(is_train):
        for step, (images, masks) in enumerate(tqdm(loader, desc=f"  {phase}", leave=False)):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            with torch.autocast(
                device_type=DEVICE.type, dtype=torch.float16,
                enabled=USE_AMP and DEVICE.type == "cuda"
            ):
                logits_up = F.interpolate(
                    model(pixel_values=images).logits,
                    size=masks.shape[-2:], mode="bilinear", align_corners=False,
                )
                loss = criterion(logits_up, masks)

            if is_train:
                loss = loss / grad_accum
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * (grad_accum if is_train else 1)
            all_preds.append(logits_up.argmax(1).detach().cpu())
            all_targets.append(masks.detach().cpu())
            n += 1

    preds_cat   = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    mean_miou, per_class = compute_miou(preds_cat, targets_cat, NUM_CLASSES)
    return total_loss / n, mean_miou, per_class


# ── train ──────────────────────────────────────────────────────────────────
def train(resume=False, init_weights=None):
    ckpt_dir = Path(CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"

    signal.signal(signal.SIGTERM, crash_handler)
    signal.signal(signal.SIGINT,  crash_handler)

    train_loader, val_loader, num_bands = build_dataloaders()

    lr    = LR * 0.3 if (resume and ckpt_path.exists()) else LR
    model = build_model(
        num_bands,
        checkpoint_path=ckpt_path   if resume else None,
        init_weights_path=init_weights if (init_weights and not resume) else None,
    )
    ema       = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None
    criterion = FocalDiceLoss(class_weights=CLASS_WEIGHTS)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, NUM_EPOCHS, WARMUP_EPOCHS)
    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP and DEVICE.type == "cuda")

    best_miou   = 0.0
    patience_cnt = 0
    history     = {"train_loss": [], "val_loss": [], "train_miou": [], "val_miou": []}

    mode = "RESUME" if resume else ("INIT-WEIGHTS" if init_weights else "FRESH")
    print("=" * 55, flush=True)
    print(f"[SPECIALIST] Mode={mode}  lr={lr:.2e}  epochs={NUM_EPOCHS}  device={DEVICE}", flush=True)
    print(f"  AMP={USE_AMP}  GradAccum={GRAD_ACCUM_STEPS}  EMA={USE_EMA}", flush=True)
    print(f"  Warmup={WARMUP_EPOCHS} epochs  Focal+Dice loss", flush=True)
    print("=" * 55, flush=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_miou, _ = run_epoch(
            model, train_loader, criterion, optimizer,
            scaler=scaler, phase="train", grad_accum=GRAD_ACCUM_STEPS,
        )
        if ema:
            ema.update(model)

        eval_model = ema.ema if ema else model
        val_loss, val_miou, per_class_iou = run_epoch(
            eval_model, val_loader, criterion, phase="val", grad_accum=1,
        )
        scheduler.step()

        pc_str = "  ".join(
            f"{CLASS_LABELS[c][:4]}={per_class_iou.get(c, 0):.3f}"
            for c in range(NUM_CLASSES)
        )
        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS}  "
            f"tr_loss={train_loss:.4f}  tr_mIoU={train_miou:.4f}  "
            f"val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  [{time.time()-t0:.0f}s]",
            flush=True,
        )
        print(f"  per-class IoU: {pc_str}", flush=True)

        for k, v in zip(
            ["train_loss", "val_loss", "train_miou", "val_miou"],
            [train_loss, val_loss, train_miou, val_miou],
        ):
            history[k].append(v)

        state.update(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            train_miou=train_miou, val_miou=val_miou,
        )

        if val_miou > best_miou:
            best_miou    = val_miou
            patience_cnt = 0
            state.update(best_miou=best_miou)
            torch.save(
                {
                    "epoch":         epoch,
                    "model_state":   ema.state_dict() if ema else model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_miou":      val_miou,
                    "num_bands":     num_bands,
                    "per_class_iou": per_class_iou,
                    "model_name":    MODEL_NAME,     # ← store backbone name
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best checkpoint (val_mIoU={best_miou:.4f})", flush=True)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"[SPECIALIST] Early stopping at epoch {epoch}", flush=True)
                break

    # ── save curves ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    e = range(1, len(history["train_loss"]) + 1)
    ax1.plot(e, history["train_loss"], label="Train")
    ax1.plot(e, history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(e, history["train_miou"], label="Train")
    ax2.plot(e, history["val_miou"],   label="Val")
    ax2.set_title("mIoU"); ax2.legend()
    plt.tight_layout()
    plt.savefig(ckpt_dir / "training_curves.png", dpi=120)
    plt.close()

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    state.update(done=True)
    print(f"[SPECIALIST] Done. Best val_mIoU={best_miou:.4f}  →  {ckpt_path}", flush=True)
    return best_miou


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train specialist model")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from specialist/checkpoints/best_model.pt")
    parser.add_argument("--init-weights", type=str, default=None,
                        help="Load weights (e.g. from generalist) but start training fresh")
    args = parser.parse_args()
    train(resume=args.resume, init_weights=args.init_weights)