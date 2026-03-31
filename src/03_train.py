"""
03_train.py — Fine-tunes SegFormer on geospatial tiles.

Features
────────
  • Mixed precision (AMP)              — ~40% memory reduction, ~1.5× speed
  • Gradient accumulation              — effective large batch without OOM
  • Focal + Dice loss                  — Focal targets hard pixels, Dice optimises overlap
  • Warmup + CosineAnnealing LR        — stable warmup avoids early divergence
  • EMA model weights                  — smoother val metrics, better generalisation
  • Run versioning                     — auto-increments, stored in run_registry.json
  • Versioned checkpoint copies        — generalist_vN_best.pt kept permanently
  • Run messages                       — describe what this run tests
  • 2-hour background notification     — periodic email with current metrics
  • Milestone emails at 25/50/75/100%  — progress tied to % of total epochs
  • Crash-safe notification            — signal handler emails on unexpected exit
  • Per-class IoU logging              — see which class is lagging
  • Explicit GPU/RAM cleanup on exit   — prevents OOM on next phase

Usage:
    python src/03_train.py
    python src/03_train.py --resume
    python src/03_train.py --run-message "Testing NIR band with mit-b3"
    python src/03_train.py --init-weights /path/to/weights.pth
"""

import sys
import json
import time
import signal
import copy
import shutil
import gc
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
dataset           = importlib.import_module("02_dataset")
build_dataloaders = dataset.build_dataloaders

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ── Milestone percentages ─────────────────────────────────────────────────────
_MILESTONE_PCTS = [25, 50, 75]


def _milestone_epoch_for_pct(pct: int, total: int) -> int:
    return max(1, round(total * pct / 100))


def _is_milestone(epoch: int, total_epochs: int):
    for pct in _MILESTONE_PCTS:
        if epoch == _milestone_epoch_for_pct(pct, total_epochs):
            return pct
    return None


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_input_bands: int, checkpoint_path=None, init_weights_path=None):
    """
    Three modes:
      checkpoint_path  : full resume — loads weights + optimizer state
      init_weights_path: weights-only init — resets epoch/optimizer/mIoU
      neither          : fresh from HuggingFace pretrained weights
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
        print(f"  (epoch=0, fresh optimizer, best_mIoU=0)", flush=True)
        model = SegformerForSemanticSegmentation(cfg)
        if num_input_bands != 3:
            _patch_embedding(model, num_input_bands)
        ckpt  = torch.load(init_weights_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}", flush=True)
        if unexpected:
            print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}", flush=True)

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
    Exponential Moving Average with step-based decay warmup.
    Warmup ramps from ~0.91 at step 1 to EMA_DECAY asymptotically so the
    shadow model tracks live weights closely in early epochs.
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
        d = min(self.decay, (1.0 + self._step) / (10.0 + self._step))
        for ema_p, m_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(m_p.data, alpha=1.0 - d)

    def state_dict(self):
        return self.ema.state_dict()


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
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
    """Soft Dice averaged over foreground classes only — background excluded."""
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        oh    = F.one_hot(targets.clamp(0, NUM_CLASSES - 1), NUM_CLASSES) \
                  .permute(0, 3, 1, 2).float()
        inter = (probs * oh).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
        dice  = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
        return dice[:, 1:].mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.focal = FocalLoss(FOCAL_GAMMA, class_weights)
        self.dice  = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return FOCAL_WEIGHT * self.focal(logits, targets) \
             + DICE_WEIGHT  * self.dice(logits, targets)


# ── LR Scheduler ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, num_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return max(1e-2, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    p, t  = preds.view(-1).cpu(), targets.view(-1).cpu()
    ious  = {}
    for cls in range(num_classes):
        inter = ((p == cls) & (t == cls)).sum().float()
        union = ((p == cls) | (t == cls)).sum().float()
        if union > 0:
            ious[cls] = (inter / union).item()
    mean_iou = float(np.mean(list(ious.values()))) if ious else 0.0
    return mean_iou, ious


# ── Shared training state ─────────────────────────────────────────────────────

class _TrainingState:
    def __init__(self):
        self._lock        = threading.Lock()
        self.epoch        = 0
        self.train_loss   = 0.0
        self.val_loss     = 0.0
        self.train_miou   = 0.0
        self.val_miou     = 0.0
        self.best_miou    = 0.0
        self.folder_name  = "unknown"
        self.folder_step  = 1
        self.folder_total = 1
        self.done         = False
        self.run_version  = None
        self.run_message  = ""

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return {k: v for k, v in self.__dict__.items()
                    if not k.startswith("_")}


_state               = _TrainingState()
_crash_handler_fired = False


def _crash_handler(signum, frame):
    global _crash_handler_fired
    if _crash_handler_fired:
        signal.signal(signal.SIGINT,  signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        sys.exit(1)
    _crash_handler_fired = True
    signal.signal(signal.SIGINT,  signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    _state.update(done=True)

    snap         = _state.snapshot()
    is_user_stop = (signum == signal.SIGINT)
    print(
        f"\n[TRAIN] Signal {signum} at epoch {snap['epoch']}. "
        f"Best val_mIoU={snap['best_miou']:.4f}  "
        f"Checkpoint safe at {CHECKPOINT_DIR}/best_model.pt",
        flush=True,
    )
    try:
        notify = importlib.import_module("07_notify")
        resume = "docker compose run --rm train-all --from " + str(snap["folder_step"])
        notify.notify_error(
            snap["folder_name"], snap["folder_step"], snap["folder_total"],
            ("Training stopped manually (Ctrl+C)" if is_user_stop
             else f"Training interrupted (signal {signum})") +
            f" at epoch {snap['epoch']}. Best mIoU={snap['best_miou']:.4f}. Resume: {resume}",
            version=snap["run_version"], run_message=snap["run_message"],
        )
    except Exception:
        pass
    sys.exit(0 if is_user_stop else 1)


def _notification_worker(interval_seconds: int):
    notify_mod = importlib.import_module("07_notify")
    while not _state.done:
        time.sleep(interval_seconds)
        if _state.done:
            break
        snap = _state.snapshot()
        if snap["epoch"] == 0:
            continue
        try:
            notify_mod.notify_training_progress(
                folder_name     = snap["folder_name"],
                step            = snap["folder_step"],
                total           = snap["folder_total"],
                epoch           = snap["epoch"],
                total_epochs    = NUM_EPOCHS,
                train_loss      = snap["train_loss"],
                val_loss        = snap["val_loss"],
                train_miou      = snap["train_miou"],
                val_miou        = snap["val_miou"],
                best_miou       = snap["best_miou"],
                checkpoint_path = str(Path(CHECKPOINT_DIR) / "best_model.pt"),
                version         = snap["run_version"],
                run_message     = snap["run_message"],
            )
        except Exception as exc:
            print(f"[NOTIFY] Periodic email failed: {exc}", flush=True)


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

            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
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


def _save_training_curves(history, ckpt_dir):
    try:
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
    except Exception:
        pass


# ── Train ─────────────────────────────────────────────────────────────────────

def train(resume: bool = False,
          init_weights: str = None,
          folder_name: str = "unknown",
          folder_step: int = 1,
          folder_total: int = 1,
          run_version: int = None,
          run_message: str = ""):

    ckpt_dir  = Path(CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"

    # ── Versioning ────────────────────────────────────────────────────────────
    if run_version is None:
        try:
            from run_version import bump_version
            run_version = bump_version("generalist")
            print(f"[VERSION] Auto-bumped to generalist v{run_version}", flush=True)
        except Exception as e:
            print(f"[VERSION] Could not auto-bump: {e}", flush=True)
            run_version = 0

    _state.update(
        folder_name=folder_name, folder_step=folder_step,
        folder_total=folder_total, run_version=run_version, run_message=run_message,
    )

    signal.signal(signal.SIGTERM, _crash_handler)
    signal.signal(signal.SIGINT,  _crash_handler)

    if NOTIFY_INTERVAL_HOURS > 0:
        threading.Thread(
            target=_notification_worker,
            args=(int(NOTIFY_INTERVAL_HOURS * 3600),),
            daemon=True,
        ).start()
        print(f"[NOTIFY] 2-hour progress emails enabled", flush=True)

    train_loader, val_loader, num_bands = build_dataloaders()

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
    history      = {"train_loss": [], "val_loss": [], "train_miou": [], "val_miou": []}
    per_class_iou = {}

    mode = ("RESUME" if (resume and ckpt_path.exists()) else
            "INIT-WEIGHTS" if init_weights else "FRESH")

    config_summary = (
        f"model={MODEL_NAME}  bands={num_bands}  epochs={NUM_EPOCHS}  "
        f"lr={lr:.2e}  AMP={USE_AMP}  EMA={USE_EMA}  device={DEVICE}"
    )

    milestone_epochs = {pct: _milestone_epoch_for_pct(pct, NUM_EPOCHS)
                        for pct in _MILESTONE_PCTS}

    print(f"\n{'='*60}", flush=True)
    print(f"  Generalist v{run_version}  |  Mode: {mode}", flush=True)
    if run_message:
        print(f"  Message: {run_message}", flush=True)
    print(f"  {config_summary}", flush=True)
    print(f"  Milestones: " +
          "  ".join(f"{pct}%→ep{ep}" for pct, ep in milestone_epochs.items()),
          flush=True)
    print(f"{'='*60}\n", flush=True)

    # Notify start
    try:
        notify = importlib.import_module("07_notify")
        notify.notify_run_start(
            model_type="generalist", version=run_version,
            run_message=run_message, mode=mode,
            folder_name=folder_name, step=folder_step, total=folder_total,
            config_summary=config_summary,
        )
    except Exception as e:
        print(f"[NOTIFY] Run-start email failed: {e}", flush=True)

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

        _state.update(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            train_miou=train_miou, val_miou=val_miou,
        )

        # ── Checkpoint on improvement ─────────────────────────────────────────
        if val_miou > best_miou:
            best_miou    = val_miou
            patience_cnt = 0
            _state.update(best_miou=best_miou)
            ckpt_data = {
                "epoch":           epoch,
                "model_state":     (ema.state_dict() if ema else model.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "val_miou":        val_miou,
                "num_bands":       num_bands,
                "per_class_iou":   per_class_iou,
                "model_name":      MODEL_NAME,
                "run_version":     run_version,
                "run_message":     run_message,
            }
            torch.save(ckpt_data, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_mIoU={best_miou:.4f})", flush=True)
            # Permanent versioned copy
            if run_version:
                try:
                    from run_version import versioned_ckpt_name
                    shutil.copy2(ckpt_path, ckpt_dir / versioned_ckpt_name("generalist", run_version))
                except Exception:
                    pass
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n[TRAIN] Early stopping at epoch {epoch}", flush=True)
                _save_training_curves(history, ckpt_dir)
                try:
                    notify = importlib.import_module("07_notify")
                    notify.notify_milestone(
                        stage="generalist", milestone_name="early_stopping",
                        version=run_version, run_message=run_message,
                        val_miou=val_miou, best_miou=best_miou,
                        epoch=epoch, total_epochs=NUM_EPOCHS,
                        details=f"No improvement for {PATIENCE} epochs. Best={best_miou:.4f}",
                        checkpoint_path=str(ckpt_path),
                        attachment_path=str(ckpt_dir / "training_curves.png"),
                    )
                except Exception:
                    pass
                break

        # ── Milestone email at 25% / 50% / 75% ───────────────────────────────
        matched_pct = _is_milestone(epoch, NUM_EPOCHS)
        if matched_pct is not None:
            _save_training_curves(history, ckpt_dir)
            try:
                notify = importlib.import_module("07_notify")
                notify.notify_milestone(
                    stage="generalist",
                    milestone_name=str(matched_pct) + "pct_done",
                    version=run_version, run_message=run_message,
                    val_miou=val_miou, best_miou=best_miou,
                    epoch=epoch, total_epochs=NUM_EPOCHS,
                    details=str(matched_pct) + "% of training complete. Best so far: " + f"{best_miou:.4f}",
                    checkpoint_path=str(ckpt_path),
                    attachment_path=str(ckpt_dir / "training_curves.png"),
                )
            except Exception as e:
                print(f"[NOTIFY] Milestone email failed: {e}", flush=True)

    # ── Final: 100% done ──────────────────────────────────────────────────────
    _save_training_curves(history, ckpt_dir)
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    try:
        notify = importlib.import_module("07_notify")
        notify.notify_milestone(
            stage="generalist", milestone_name="training_complete",
            version=run_version, run_message=run_message,
            val_miou=val_miou, best_miou=best_miou,
            epoch=len(history["train_loss"]), total_epochs=NUM_EPOCHS,
            details="Training finished. Best val_mIoU=" + f"{best_miou:.4f}",
            checkpoint_path=str(ckpt_path),
            attachment_path=str(ckpt_dir / "training_curves.png"),
        )
    except Exception:
        pass

    # ── Register run ──────────────────────────────────────────────────────────
    try:
        from run_version import register_run
        register_run(
            model_type="generalist", version=run_version,
            message=run_message, best_val_miou=best_miou,
            checkpoint=str(ckpt_path),
            extra={"folder": folder_name, "epochs_trained": len(history["train_loss"])},
        )
    except Exception as e:
        print(f"[VERSION] Could not register run: {e}", flush=True)

    print(f"\n✓ Done. Best val_mIoU={best_miou:.4f} | {ckpt_path}", flush=True)
    if run_version:
        print(f"  Versioned copy: {ckpt_dir}/generalist_v{run_version}_best.pt", flush=True)

    # ── Explicit memory cleanup (prevents OOM on next pipeline phase) ─────────
    _state.update(done=True)
    print("[TRAIN] Releasing GPU and CPU memory...", flush=True)
    del train_loader, val_loader
    del model, optimizer, scheduler, criterion, scaler
    if ema:
        del ema
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        print(f"  GPU after cleanup: allocated={allocated:.2f} GB  reserved={reserved:.2f} GB",
              flush=True)

    return best_miou


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GeoSeg generalist model")
    parser.add_argument("--resume",        action="store_true",
                        help="Resume from checkpoint (fine-tune, LR×0.3)")
    parser.add_argument("--init-weights",  type=str, default=None,
                        help="Load weights but start training fresh (epoch=0, full LR)")
    parser.add_argument("--folder-name",   type=str, default="unknown")
    parser.add_argument("--folder-step",   type=int, default=1)
    parser.add_argument("--folder-total",  type=int, default=2)
    parser.add_argument("--run-version",   type=int, default=None,
                        help="Version number (auto-incremented if omitted)")
    parser.add_argument("--run-message",   type=str, default="",
                        help="Human-readable description of this run")
    args = parser.parse_args()

    train(
        resume       = args.resume,
        init_weights = args.init_weights,
        folder_name  = args.folder_name,
        folder_step  = args.folder_step,
        folder_total = args.folder_total,
        run_version  = args.run_version,
        run_message  = args.run_message,
    )