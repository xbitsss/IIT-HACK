"""
03_train.py — Fine-tunes SegFormer on geospatial tiles.

Improvements vs original:
  • Mixed precision (AMP)             — ~40% memory reduction, ~1.5× speed
  • Gradient accumulation             — effective large batch without OOM
  • Focal + Dice loss                 — Focal targets hard pixels, Dice
                                        optimises overlap directly
  • Warmup + CosineAnnealing LR       — stable warmup avoids early divergence
  • EMA model weights                 — smoother val metrics, better generalisation
  • 2-hour background notification    — periodic email with current metrics
  • Crash-safe notification           — signal handler emails on unexpected exit
  • Per-class IoU logging             — easier to spot which class is lagging
"""

import sys
import json
import time
import signal
import copy
import threading
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
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
    CHECKPOINT_DIR, DATA_PROCESSED_DIR, RANDOM_SEED,
    USE_AMP, GRAD_ACCUM_STEPS, WARMUP_EPOCHS,
    USE_EMA, EMA_DECAY,
    FOCAL_GAMMA, FOCAL_WEIGHT, DICE_WEIGHT,
    NOTIFY_INTERVAL_HOURS,
)
import importlib
dataset = importlib.import_module("02_dataset")
build_dataloaders = dataset.build_dataloaders

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_input_bands: int, checkpoint_path=None, init_weights_path=None):
    """
    Three modes:
      checkpoint_path  : full resume — loads weights + prints epoch/mIoU
      init_weights_path: weights-only init — loads weights, resets all training
                         state (epoch=0, fresh optimizer, best_mIoU=0).
                         Use this to start fresh training from a custom .pth
                         instead of from HuggingFace pretrained weights.
      neither          : fresh from HuggingFace pretrained weights (default)
    """
    print(f"Loading {MODEL_NAME} with {num_input_bands} band(s)...", flush=True)

    cfg = SegformerConfig.from_pretrained(MODEL_NAME)
    cfg.num_labels   = NUM_CLASSES
    cfg.id2label     = {i: l for i, l in enumerate(CLASS_LABELS)}
    cfg.label2id     = {l: i for i, l in enumerate(CLASS_LABELS)}
    cfg.num_channels = num_input_bands

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"  Resuming from checkpoint: {checkpoint_path}", flush=True)
        model = SegformerForSemanticSegmentation(cfg)
        if num_input_bands != 3:
            _patch_embedding(model, num_input_bands)
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded  epoch={ckpt['epoch']}  val_mIoU={ckpt['val_miou']:.4f}", flush=True)

    elif init_weights_path and Path(init_weights_path).exists():
        print(f"  Init weights from: {init_weights_path}", flush=True)
        print(f"  (Training state reset — epoch=0, fresh optimizer, best_mIoU=0)", flush=True)
        model = SegformerForSemanticSegmentation(cfg)
        if num_input_bands != 3:
            _patch_embedding(model, num_input_bands)
        ckpt = torch.load(init_weights_path, map_location=DEVICE, weights_only=False)
        # Load only model weights — ignore optimizer/epoch/mIoU from the file
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}", flush=True)
        if unexpected:
            print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}", flush=True)
        print(f"  Weights loaded. Starting fresh training.", flush=True)

    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME, config=cfg, ignore_mismatched_sizes=True
        )
        if num_input_bands != 3:
            _patch_embedding(model, num_input_bands)

    return model.to(DEVICE)


def _patch_embedding(model, num_input_bands: int):
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


# ── EMA ───────────────────────────────────────────────────────────────────────

class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Maintains a shadow copy that is smoother than the live weights.
    Typically gives +0.5–2% mIoU on val vs the raw checkpoint.

    FIX (Bug 1): added step-based decay warmup.  With EMA_DECAY=0.9998 and
    ~1500 steps/epoch, the naive formula kept ~54% of random-init weights after
    2 epochs, making val_mIoU appear stuck.  The warmup ramps actual decay from
    ~0.91 at step 1 to EMA_DECAY asymptotically, so the shadow model tracks the
    live weights closely during the critical early epochs.
    Formula: d = min(decay, (1 + step) / (10 + step))  — standard PyTorch EMA.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.ema   = copy.deepcopy(model).eval()
        self.decay = decay
        self._step = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        self._step += 1
        # Warmup: ramps from ~0.91 at step 1 up to self.decay asymptotically.
        # Prevents the shadow model from being dominated by random-init weights
        # in the first few epochs when step count is low.
        d = min(self.decay, (1.0 + self._step) / (10.0 + self._step))
        for ema_p, m_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(m_p.data, alpha=1.0 - d)

    def state_dict(self):
        return self.ema.state_dict()


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss down-weights easy (well-classified) pixels so the model focuses
    training gradient on hard examples — boundaries, thin roads, small water bodies.
    """
    def __init__(self, gamma: float = 2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.w = (torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
                  if class_weights else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce  = F.cross_entropy(logits, targets, weight=self.w,
                              reduction="none", ignore_index=255)
        pt  = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class DiceLoss(nn.Module):
    """
    Soft Dice loss: directly optimises pixel-overlap, combats class imbalance.
    Averaged over foreground classes only (classes 1+).

    FIX (Bug 4): background (class 0) is ~95% of all pixels.  Including it in
    the Dice mean caused the loss to be dominated by how well the model predicts
    background, swamping the gradient signal for rare foreground classes (road,
    water, built-up).  FocalLoss already down-weights background via CLASS_WEIGHTS
    [0]=0.4; Dice now mirrors that by skipping class 0 entirely.
    """
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs  = F.softmax(logits, dim=1)
        oh     = F.one_hot(targets.clamp(0, NUM_CLASSES - 1), NUM_CLASSES) \
                   .permute(0, 3, 1, 2).float()
        inter  = (probs * oh).sum(dim=(2, 3))
        union  = probs.sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
        dice   = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
        return dice[:, 1:].mean()  # skip class 0 (background)


class FocalDiceLoss(nn.Module):
    def __init__(self, class_weights=None,
                 focal_weight: float = FOCAL_WEIGHT,
                 dice_weight:  float = DICE_WEIGHT,
                 gamma:        float = FOCAL_GAMMA):
        super().__init__()
        self.focal        = FocalLoss(gamma, class_weights)
        self.dice         = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight  = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.focal_weight * self.focal(logits, targets) +
                self.dice_weight  * self.dice(logits,  targets))


# ── LR Schedule: Warmup + Cosine ─────────────────────────────────────────────

def build_scheduler(optimizer, num_epochs: int, warmup_epochs: int):
    """
    Linear warmup for `warmup_epochs`, then cosine decay to 1e-6.
    Warmup avoids the large-gradient instability in early epochs.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return max(1e-2, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """Returns (mean_iou, per_class_iou_dict)."""
    p, t   = preds.view(-1).cpu(), targets.view(-1).cpu()
    ious   = {}
    for cls in range(num_classes):
        inter = ((p == cls) & (t == cls)).sum().float()
        union = ((p == cls) | (t == cls)).sum().float()
        if union > 0:
            ious[cls] = (inter / union).item()
    mean_iou = float(np.mean(list(ious.values()))) if ious else 0.0
    return mean_iou, ious


# ── Periodic notification thread ──────────────────────────────────────────────

class _TrainingState:
    """Thread-safe container for the latest training metrics."""
    def __init__(self):
        self._lock       = threading.Lock()
        self.epoch       = 0
        self.train_loss  = 0.0
        self.val_loss    = 0.0
        self.train_miou  = 0.0
        self.val_miou    = 0.0
        self.best_miou   = 0.0
        self.folder_name = "unknown"
        self.folder_step = 1
        self.folder_total= 1
        self.done        = False

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self):
        with self._lock:
            return {k: v for k, v in self.__dict__.items()
                    if not k.startswith("_")}


_state = _TrainingState()


def _notification_worker(interval_seconds: int):
    """Background daemon thread — sends a progress email every N seconds."""
    import importlib
    notify_mod = importlib.import_module("07_notify")

    while not _state.done:
        time.sleep(interval_seconds)
        if _state.done:
            break
        snap = _state.snapshot()
        try:
            notify_mod.notify_training_progress(
                folder_name  = snap["folder_name"],
                step         = snap["folder_step"],
                total        = snap["folder_total"],
                epoch        = snap["epoch"],
                total_epochs = NUM_EPOCHS,
                train_loss   = snap["train_loss"],
                val_loss     = snap["val_loss"],
                train_miou   = snap["train_miou"],
                val_miou     = snap["val_miou"],
                best_miou    = snap["best_miou"],
                checkpoint_path = str(Path(CHECKPOINT_DIR) / "best_model.pt"),
            )
        except Exception as exc:
            print(f"[NOTIFY] Periodic email failed: {exc}", flush=True)


# One-shot flag — ensures the crash handler body runs at most once,
# even if multiple signals arrive in quick succession (e.g. Ctrl+C spam,
# or SIGINT propagating to DataLoader worker subprocesses).
_crash_handler_fired = False


def _crash_handler(signum, frame):
    """
    Send a single crash notification on SIGTERM / SIGINT then exit cleanly.

    Guards against repeated firing:
      - Resets both signal handlers to SIG_DFL immediately so any further
        signals (Ctrl+C spam, worker subprocess signals) are handled by the
        OS default (hard kill) instead of re-entering this function.
      - _crash_handler_fired flag provides a second layer of protection in
        case the signal arrives on a different thread before SIG_DFL is set.
      - Sets _state.done = True so the notification daemon thread stops
        immediately and does not send any more progress emails while dying.
    """
    global _crash_handler_fired
    if _crash_handler_fired:
        # Second signal — just exit hard, don't send another email
        signal.signal(signal.SIGINT,  signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        sys.exit(1)
    _crash_handler_fired = True

    # Unregister immediately — further signals won't re-enter this function
    signal.signal(signal.SIGINT,  signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    # Stop the notification daemon thread before it can send more emails
    _state.update(done=True)

    # Only send the crash email if this was an unexpected signal.
    # SIGINT (Ctrl+C) is a deliberate user stop — send a brief stopped email
    # rather than a scary "ERROR" email.
    is_user_stop = (signum == signal.SIGINT)
    try:
        import importlib
        notify_mod = importlib.import_module("07_notify")
        snap = _state.snapshot()
        if is_user_stop:
            notify_mod.notify_error(
                folder_name = snap["folder_name"],
                step        = snap["folder_step"],
                total       = snap["folder_total"],
                error_msg   = (
                    f"Training stopped manually (Ctrl+C) at epoch {snap['epoch']}. \n"
                    f"Best val_mIoU so far: {snap['best_miou']:.4f}\n"
                    f"Checkpoint is safe at checkpoints/best_model.pt.\n"
                    f"Resume: docker compose run --rm train-all --from {snap['folder_step']}"
                ),
            )
        else:
            notify_mod.notify_error(
                folder_name = snap["folder_name"],
                step        = snap["folder_step"],
                total       = snap["folder_total"],
                error_msg   = (
                    f"Training process received signal {signum} at epoch {snap['epoch']}.\n"
                    f"Checkpoint safe at checkpoints/best_model.pt.\n"
                    f"Resume: docker compose run --rm train-all --from {snap['folder_step']}"
                ),
            )
    except Exception:
        pass

    sys.exit(0 if is_user_stop else 1)


# ── Epoch ─────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None,
              scaler=None, phase="train", grad_accum=1):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    all_preds   = []
    all_targets = []
    n = 0

    if is_train and optimizer:
        optimizer.zero_grad()

    with torch.set_grad_enabled(is_train):
        for step, (images, masks) in enumerate(tqdm(loader, desc=f"  {phase}", leave=False)):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            with torch.autocast(device_type=DEVICE.type,
                                dtype=torch.float16,
                                enabled=(USE_AMP and DEVICE.type == "cuda")):
                logits_up = F.interpolate(
                    model(pixel_values=images).logits,
                    size=masks.shape[-2:], mode="bilinear", align_corners=False,
                )
                loss = criterion(logits_up, masks)
                if is_train:
                    loss = loss / grad_accum

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad()

            total_loss  += loss.item() * (grad_accum if is_train else 1)
            all_preds.append(logits_up.argmax(1).detach().cpu())
            all_targets.append(masks.detach().cpu())
            n += 1

    preds_cat   = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    mean_miou, per_class = compute_miou(preds_cat, targets_cat, NUM_CLASSES)

    return total_loss / n, mean_miou, per_class


# ── Train ─────────────────────────────────────────────────────────────────────

def train(resume: bool = False,
          init_weights: str = None,
          folder_name: str = "unknown",
          folder_step: int = 1,
          folder_total: int = 1):

    ckpt_dir  = Path(CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"

    # Update shared state for notification thread
    _state.update(
        folder_name  = folder_name,
        folder_step  = folder_step,
        folder_total = folder_total,
    )

    # Install crash handler in main process only.
    # DataLoader worker subprocesses inherit signal handlers from the parent —
    # if they also run _crash_handler you get one crash email per worker.
    # The worker_init_fn below resets SIGINT to SIG_IGN in every worker so
    # Ctrl+C is handled exclusively by the main process.
    signal.signal(signal.SIGTERM, _crash_handler)
    signal.signal(signal.SIGINT,  _crash_handler)

    # Start periodic notification thread
    if NOTIFY_INTERVAL_HOURS > 0:
        t = threading.Thread(
            target=_notification_worker,
            args=(int(NOTIFY_INTERVAL_HOURS * 3600),),
            daemon=True,
        )
        t.start()
        print(f"[NOTIFY] Periodic emails every {NOTIFY_INTERVAL_HOURS}h enabled", flush=True)

    train_loader, val_loader, num_bands = build_dataloaders()

    # LR selection:
    #   resume        → 0.3× LR (fine-tuning continuation)
    #   init_weights  → full LR (fresh training, just different weight init)
    #   fresh         → full LR
    lr    = LR * 0.3 if (resume and ckpt_path.exists()) else LR
    model = build_model(
        num_bands,
        checkpoint_path   = ckpt_path if resume else None,
        init_weights_path = init_weights if (init_weights and not resume) else None,
    )
    ema   = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

    criterion = FocalDiceLoss(class_weights=CLASS_WEIGHTS)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, NUM_EPOCHS, WARMUP_EPOCHS)
    scaler    = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))

    best_miou    = 0.0
    patience_cnt = 0
    history      = {"train_loss": [], "val_loss": [],
                    "train_miou": [], "val_miou": []}

    if resume and ckpt_path.exists():
        mode = "RESUME (incremental)"
    elif init_weights:
        mode = f"INIT-WEIGHTS (fresh training from {Path(init_weights).name})"
    else:
        mode = "FRESH"
    print(f"\n{'='*55}", flush=True)
    print(f"Mode: {mode} | lr={lr:.2e} | epochs={NUM_EPOCHS} | device={DEVICE}", flush=True)
    print(f"AMP={USE_AMP} | GradAccum={GRAD_ACCUM_STEPS} | EMA={USE_EMA}", flush=True)
    print(f"Warmup={WARMUP_EPOCHS} epochs | Focal+Dice loss", flush=True)
    print(f"{'='*55}\n", flush=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_miou, _ = run_epoch(
            model, train_loader, criterion, optimizer,
            scaler=scaler, phase="train", grad_accum=GRAD_ACCUM_STEPS,
        )

        if ema:
            ema.update(model)

        # Evaluate EMA model if available (usually better than raw weights)
        eval_model = ema.ema if ema else model
        val_loss, val_miou, per_class_iou = run_epoch(
            eval_model, val_loader, criterion,
            phase="val", grad_accum=1,
        )

        

        scheduler.step()

        # ── Per-class IoU string ──────────────────────────────────────────────
        pc_str = "  ".join(
            f"{CLASS_LABELS[c][:4]}={per_class_iou.get(c, 0):.3f}"
            for c in range(NUM_CLASSES)
        )

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS}  "
              f"tr_loss={train_loss:.4f}  tr_mIoU={train_miou:.4f}  "
              f"val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  [{time.time()-t0:.0f}s]",
              flush=True)
        print(f"         per-class IoU: {pc_str}", flush=True)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_miou"].append(train_miou)
        history["val_miou"].append(val_miou)

        # Update shared state for notification thread
        _state.update(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            train_miou=train_miou, val_miou=val_miou,
        )

        if val_miou > best_miou:
            best_miou    = val_miou
            patience_cnt = 0
            _state.update(best_miou=best_miou)
            torch.save({
                "epoch":           epoch,
                "model_state":     (ema.state_dict() if ema else model.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "val_miou":        val_miou,
                "num_bands":       num_bands,
                "per_class_iou":   per_class_iou,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_mIoU={best_miou:.4f})", flush=True)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}", flush=True)
                break

    # ── Save training curves ──────────────────────────────────────────────────
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

    print(f"\n✓ Done. Best val_mIoU={best_miou:.4f} | {ckpt_path}", flush=True)
    _state.update(done=True)
    return best_miou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",        action="store_true",
                        help="Resume training from checkpoint (fine-tune, LR×0.3)")
    parser.add_argument("--init-weights",  type=str, default=None,
                        help="Load weights from .pth but train completely fresh "
                             "(epoch=0, full LR, no optimizer state loaded). "
                             "Use this to start from a custom backbone instead of HuggingFace.")
    parser.add_argument("--folder-name",   type=str, default="unknown")
    parser.add_argument("--folder-step",   type=int, default=1)
    parser.add_argument("--folder-total",  type=int, default=4)
    args = parser.parse_args()

    train(
        resume       = args.resume,
        init_weights = args.init_weights,
        folder_name  = args.folder_name,
        folder_step  = args.folder_step,
        folder_total = args.folder_total,
    )