#!/bin/bash
# 06_train_incremental.sh — Incremental training pipeline
#
# Approach B: all TIFFs live in one folder; they are shuffled and split
# 50/50 into shard 1 and shard 2 randomly each run.
#
# Data layout:
#   /raw_data/ALL/
#       image_001.tif
#       image_002.tif
#       ...
#       SHP1/  *.shp      ← shapefile group 1
#       SHP2/  *.shp      ← shapefile group 2  (or just one SHP dir)
#
# ─────────────────────────────────────────────────────────────────────────
# THREE ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────
#
#  1) Full run (shard 1 + shard 2 + specialist):
#       ./06_train_incremental.sh --data-dir /raw_data/ALL
#
#  2) Resume from shard 2 (skips shard 1):
#       ./06_train_incremental.sh --data-dir /raw_data/ALL --from 2
#
#  3) Specialist only (generalist checkpoint + replay must exist):
#       ./06_train_incremental.sh --specialist-only
#
# ─────────────────────────────────────────────────────────────────────────
# FLAGS
# ─────────────────────────────────────────────────────────────────────────
#   --data-dir /path           Root folder containing TIFFs
#                              (default: /raw_data/ALL)
#   --shp-dirs dir1:dir2       Colon-separated SHP dirs
#                              (default: auto-detect subdirs in --data-dir)
#   --from N                   Resume from shard N  (1 or 2)
#   --pretrained /path.pth     Fine-tune from existing checkpoint
#   --init-weights /path.pth   Load weights only — epoch 0, full LR
#   --specialist-only          Skip generalist shards entirely.
#                              Requires: checkpoints/best_model.pt
#                                        data/replay/ (from previous run)
#   --skip-specialist          Run generalist shards only; skip specialist
#
# ─────────────────────────────────────────────────────────────────────────
# ENVIRONMENT OVERRIDES
# ─────────────────────────────────────────────────────────────────────────
#   MAX_DISK_GB                Hard disk ceiling in GB      (default: 50)
#   REPLAY_TILES_PER_SHARD     Tiles kept per shard         (default: 300)

set -euo pipefail

export MAX_DISK_GB="${MAX_DISK_GB:-50}"
export REPLAY_TILES_PER_SHARD="${REPLAY_TILES_PER_SHARD:-300}"

DATA_DIR_B="/raw_data/ALL"
SHP_DIRS_B=""
PRETRAINED_CKPT=""
INIT_WEIGHTS=""
START_FROM=1
SKIP_SPECIALIST=0
SPECIALIST_ONLY=0

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)            START_FROM="$2";      shift 2 ;;
        --pretrained)      PRETRAINED_CKPT="$2"; shift 2 ;;
        --init-weights)    INIT_WEIGHTS="$2";    shift 2 ;;
        --data-dir)        DATA_DIR_B="$2";      shift 2 ;;
        --shp-dirs)        SHP_DIRS_B="$2";      shift 2 ;;
        --skip-specialist) SKIP_SPECIALIST=1;    shift   ;;
        --specialist-only) SPECIALIST_ONLY=1;    shift   ;;
        --help|-h)
            head -55 "$0" | tail -50
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: $0 [--data-dir /path] [--from N] [--specialist-only]"
            echo "          [--pretrained /path] [--init-weights /path]"
            echo "          [--shp-dirs dir1:dir2] [--skip-specialist]"
            exit 1
            ;;
    esac
done

# ── Mutual exclusion ─────────────────────────────────────────────────────────
if [ "$SPECIALIST_ONLY" -eq 1 ] && [ "$SKIP_SPECIALIST" -eq 1 ]; then
    echo "[ERROR] --specialist-only and --skip-specialist are mutually exclusive."
    exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
dir_bytes() {
    local dir="$1"
    if [ -d "$dir" ]; then
        du -sb "$dir" 2>/dev/null | awk '{print $1}' || echo "0"
    else
        echo "0"
    fi
}

compute_processed_budget_gb() {
    local replay_bytes
    replay_bytes=$(dir_bytes "data/replay")
    local max_bytes
    max_bytes=$(python3 -c "print(int(${MAX_DISK_GB} * 1024**3))")
    local budget_bytes=$(( max_bytes - replay_bytes ))
    if [ "$budget_bytes" -lt $(( 1 * 1024 * 1024 * 1024 )) ]; then
        budget_bytes=$(( 1 * 1024 * 1024 * 1024 ))
        echo "[WARN] Replay buffer large — processed budget clamped to 1 GB" >&2
    fi
    python3 -c "print(f'{${budget_bytes} / 1024**3:.3f}')"
}

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT 3: Specialist only
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SPECIALIST_ONLY" -eq 1 ]; then
    echo ""
    echo "========================================"
    echo " Entry Point 3: Specialist-Only Run"
    echo " Skipping all generalist shard processing."
    echo "========================================"

    # Validate prerequisites
    if [ ! -f "checkpoints/best_model.pt" ]; then
        echo "[ERROR] checkpoints/best_model.pt not found."
        echo "        Run shards 1 and 2 first (or supply --pretrained)."
        exit 1
    fi

    if ! python3 - <<'PYEOF' 2>/dev/null
import sys; sys.path.insert(0, 'src')
from replay_buffer import replay_exists
sys.exit(0 if replay_exists() else 1)
PYEOF
    then
        echo "[ERROR] Replay buffer (data/replay/) not found or empty."
        echo "        Run shards 1 and 2 first so replay tiles are saved."
        exit 1
    fi

    echo "[OK] Prerequisites satisfied — generalist checkpoint and replay buffer found."
    TOTAL=2   # for notification step labels
    CURRENT_FOLDER="specialist"
    CURRENT_STEP="$TOTAL"

    # ── Run specialist pipeline ───────────────────────────────────────────
    echo ""
    echo "──────────────────────────────────────────"
    echo " [S1] Building specialist dataset metadata"
    echo "──────────────────────────────────────────"
    if ! python specialist/01_build_specialist_meta.py 2>&1 | tee /tmp/specialist_meta_log.txt; then
        echo "[ERROR] Specialist metadata build failed"
        python src/07_notify.py --error \
            --folder "specialist/01_build_specialist_meta" --step "$TOTAL" --total "$TOTAL" \
            --error-msg "$(tail -40 /tmp/specialist_meta_log.txt | head -c 2000)" || true
        exit 1
    fi

    echo ""
    echo "──────────────────────────────────────────"
    echo " [S2] Training specialist model"
    echo "──────────────────────────────────────────"
    if ! python specialist/03_train_specialist.py 2>&1 | tee /tmp/specialist_train_log.txt; then
        echo "[ERROR] Specialist training failed"
        python src/07_notify.py --error \
            --folder "specialist" --step "$TOTAL" --total "$TOTAL" \
            --error-msg "$(tail -40 /tmp/specialist_train_log.txt | head -c 2000)" || true
        exit 1
    fi

    SPEC_TRAIN_LOSS=$(grep "tr_loss="  /tmp/specialist_train_log.txt | tail -1 | grep -oP "tr_loss=\K[0-9.]+"  || echo "0")
    SPEC_VAL_LOSS=$(  grep "val_loss=" /tmp/specialist_train_log.txt | tail -1 | grep -oP "val_loss=\K[0-9.]+" || echo "0")
    SPEC_TRAIN_MIOU=$(grep "tr_mIoU="  /tmp/specialist_train_log.txt | tail -1 | grep -oP "tr_mIoU=\K[0-9.]+"  || echo "0")
    SPEC_VAL_MIOU=$(  grep "val_mIoU=" /tmp/specialist_train_log.txt | tail -1 | grep -oP "val_mIoU=\K[0-9.]+" || echo "0")
    SPEC_EPOCHS=$(grep -c "^Epoch " /tmp/specialist_train_log.txt 2>/dev/null | tr -d '[:space:]' || echo "0")

    SPEC_PER_CLASS=$(python3 -c "
import torch, sys
try:
    ckpt = torch.load('specialist/checkpoints/best_model.pt', map_location='cpu', weights_only=False)
    pc   = ckpt.get('per_class_iou', {})
    def g(k): return float(pc.get(k, pc.get(str(k), 0.0)))
    print(f'Bridge={g(4):.4f}  Railway={g(5):.4f}  Utility={g(6):.4f}')
except Exception as e:
    print(f'(per-class read failed: {e})')
" 2>/dev/null || echo "(unavailable)")

    echo "[$(date +%H:%M:%S)] Specialist done — val_mIoU=${SPEC_VAL_MIOU}  ${SPEC_PER_CLASS}"

    python src/07_notify.py \
        --folder "Specialist (specialist-only) [${SPEC_PER_CLASS}]" \
        --step "$TOTAL" --total "$TOTAL" \
        --train-loss "${SPEC_TRAIN_LOSS:-0}" --val-loss "${SPEC_VAL_LOSS:-0}" \
        --train-miou "${SPEC_TRAIN_MIOU:-0}" --val-miou "${SPEC_VAL_MIOU:-0}" \
        --epochs "${SPEC_EPOCHS:-0}" --checkpoint "specialist/checkpoints/best_model.pt" || true

    echo ""
    echo "========================================"
    echo " Specialist-only run complete"
    echo " Specialist : specialist/checkpoints/best_model.pt"
    echo " Combined inference:"
    echo "   python specialist/04_inference_combined.py \\"
    echo "       --input /path/to/image.tif --output outputs/mask.tif"
    echo "========================================"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINTS 1 & 2: Generalist shards (+ optional specialist)
# ─────────────────────────────────────────────────────────────────────────────

# ── Discover TIFFs and build shard split ──────────────────────────────────────
if [ ! -d "$DATA_DIR_B" ]; then
    echo "[ERROR] --data-dir '$DATA_DIR_B' does not exist."
    exit 1
fi

mapfile -t ALL_TIFS < <(
    find "$DATA_DIR_B" -maxdepth 1 \( -iname "*.tif" -o -iname "*.tiff" \) | \
    python3 -c "import sys, random; lines=sys.stdin.read().splitlines(); random.shuffle(lines); print('\n'.join(lines))"
)
N_TIFS=${#ALL_TIFS[@]}
if [ "$N_TIFS" -eq 0 ]; then
    echo "[ERROR] No TIFF files found in $DATA_DIR_B"
    exit 1
fi

HALF=$(( N_TIFS / 2 ))
SHARD1_FILES=$(IFS=:; echo "${ALL_TIFS[*]:0:$HALF}")
SHARD2_FILES=$(IFS=:; echo "${ALL_TIFS[*]:$HALF}")

# ── Auto-detect SHP dirs ──────────────────────────────────────────────────────
if [ -z "$SHP_DIRS_B" ]; then
    for subdir in "$DATA_DIR_B"/*/; do
        [ -d "$subdir" ] || continue
        if find "$subdir" -maxdepth 1 -iname "*.shp" 2>/dev/null | grep -q .; then
            SHP_DIRS_B="${SHP_DIRS_B:+$SHP_DIRS_B:}${subdir%/}"
        fi
    done
fi

if [ -z "$SHP_DIRS_B" ]; then
    echo "[ERROR] No SHP subdirectories found in $DATA_DIR_B"
    echo "  Either add SHP folders or specify --shp-dirs dir1:dir2"
    exit 1
fi

FOLDERS=("$DATA_DIR_B" "$DATA_DIR_B")
TIFF_FILES_PER_SHARD=("$SHARD1_FILES" "$SHARD2_FILES")
TOTAL=2

# ── Validate --from ───────────────────────────────────────────────────────────
if [ "$START_FROM" -lt 1 ] || [ "$START_FROM" -gt "$TOTAL" ]; then
    echo "[ERROR] --from must be 1 or 2 (got $START_FROM)"
    exit 1
fi

echo "========================================"
echo " Incremental Training Pipeline"
echo " Data dir      : $DATA_DIR_B"
echo " Total TIFFs   : $N_TIFS  →  shard 1: $HALF  shard 2: $(( N_TIFS - HALF ))"
echo " SHP dirs      : $SHP_DIRS_B"
echo " Starting from : shard $START_FROM"
echo " Disk ceiling  : ${MAX_DISK_GB} GB"
echo " Replay/shard  : ${REPLAY_TILES_PER_SHARD} tiles"
echo " Skip specialis: $SKIP_SPECIALIST"
echo "========================================"

# ── Fresh start wipe ──────────────────────────────────────────────────────────
if [ "$START_FROM" -eq 1 ]; then
    echo ""
    echo "[FRESH START] Wiping stale data..."
    [ -d "data/replay" ]    && find data/replay -mindepth 1 -delete    && echo "  Cleared data/replay/"
    [ -d "data/processed" ] && find data/processed -mindepth 1 -delete && echo "  Cleared data/processed/"

    if [ -n "$INIT_WEIGHTS" ]; then
        if [ ! -f "$INIT_WEIGHTS" ]; then
            echo "[ERROR] --init-weights path does not exist: $INIT_WEIGHTS"
            exit 1
        fi
        rm -f checkpoints/best_model.pt checkpoints/training_curves.png checkpoints/history.json
        echo "  Init weights: $INIT_WEIGHTS (epoch=0, fresh optimizer)"
    elif [ -n "$PRETRAINED_CKPT" ]; then
        if [ ! -f "$PRETRAINED_CKPT" ]; then
            echo "[ERROR] --pretrained path does not exist: $PRETRAINED_CKPT"
            exit 1
        fi
        mkdir -p checkpoints
        cp "$PRETRAINED_CKPT" checkpoints/best_model.pt
        echo "  Installed pretrained checkpoint → checkpoints/best_model.pt"
    else
        rm -f checkpoints/best_model.pt checkpoints/training_curves.png checkpoints/history.json
    fi
    echo "[FRESH START] Done."
    echo ""
fi

# ── Crash handler ─────────────────────────────────────────────────────────────
CURRENT_FOLDER="unknown"
CURRENT_STEP=0
shell_crash_handler() {
    local EXIT_CODE=$?
    echo "[CRASH] Shell crashed at $CURRENT_FOLDER (exit $EXIT_CODE)"
    python src/07_notify.py --error \
        --folder "$CURRENT_FOLDER" --step "$CURRENT_STEP" --total "$TOTAL" \
        --error-msg "Shell crashed (exit $EXIT_CODE) at $CURRENT_FOLDER. Resume: --from $CURRENT_STEP" \
        || true
}
trap shell_crash_handler ERR

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP — two shards
# ─────────────────────────────────────────────────────────────────────────────
for i in "${!FOLDERS[@]}"; do
    STEP=$(( i + 1 ))

    if [ "$STEP" -lt "$START_FROM" ]; then
        echo "Skipping shard $STEP — (--from $START_FROM)"
        continue
    fi

    FOLDER="${FOLDERS[$i]}"
    FOLDER_NAME="shard_${STEP}"
    CURRENT_FOLDER="$FOLDER_NAME"
    CURRENT_STEP="$STEP"

    # BUG FIX (wipe-order): if this is the last shard AND we will run the
    # specialist, we must NOT wipe processed tiles at the end of this step.
    # Processed tiles (tiles_meta.json + images/) must survive long enough
    # for 01_build_specialist_meta.py to read them.  We wipe AFTER the
    # specialist pipeline finishes (see "Post-specialist wipe" section below).
    IS_LAST_FOR_SPECIALIST=0
    if [ "$STEP" -eq "$TOTAL" ] && [ "$SKIP_SPECIALIST" -eq 0 ]; then
        IS_LAST_FOR_SPECIALIST=1
    fi

    echo ""
    echo "========================================"
    echo " Shard $STEP/$TOTAL: $FOLDER_NAME"
    TIFF_COUNT=$(echo "${TIFF_FILES_PER_SHARD[$i]}" | tr ':' '\n' | grep -c . || echo "?")
    echo " TIFFs        : $TIFF_COUNT files (random subset)"
    echo " SHP dirs     : $SHP_DIRS_B"
    if [ "$IS_LAST_FOR_SPECIALIST" -eq 1 ]; then
        echo " [NOTE] Processed tiles kept alive for specialist pipeline"
    fi
    echo "========================================"

    # ── Disk accounting ───────────────────────────────────────────────────
    REPLAY_BYTES=$(dir_bytes "data/replay")
    REPLAY_GB=$(python3 -c "print(f'{${REPLAY_BYTES}/1024**3:.3f}')")
    PROCESSED_BUDGET_GB=$(compute_processed_budget_gb)
    echo "[$(date +%H:%M:%S)] Disk: replay=${REPLAY_GB} GB  budget=${PROCESSED_BUDGET_GB} GB  ceiling=${MAX_DISK_GB} GB"
    python3 -c "
replay_gb=${REPLAY_GB}; ceiling=${MAX_DISK_GB}; budget=${PROCESSED_BUDGET_GB}
assert replay_gb < ceiling, f'[ERROR] Replay exceeds ceiling! {replay_gb:.2f} >= {ceiling}'
assert budget >= 1.0,        f'[ERROR] Budget < 1 GB — increase MAX_DISK_GB'
"

    # ── Preprocess ────────────────────────────────────────────────────────
    echo "[$(date +%H:%M:%S)] Preprocessing $FOLDER_NAME..."
    PREPROCESS_ENV=(
        RAW_DATA_DIR="$FOLDER"
        SHP_DIR="$SHP_DIRS_B"
        SHP_DIRS_LIST="$SHP_DIRS_B"
        MAX_PROCESSED_GB="$PROCESSED_BUDGET_GB"
        TIFF_FILES="${TIFF_FILES_PER_SHARD[$i]}"
    )

    if ! env "${PREPROCESS_ENV[@]}" python src/01_preprocess.py 2>&1 | tee /tmp/preprocess_log.txt; then
        echo "[ERROR] Preprocessing failed for $FOLDER_NAME"
        python src/07_notify.py --error \
            --folder "$FOLDER_NAME" --step "$STEP" --total "$TOTAL" \
            --error-msg "$(tail -40 /tmp/preprocess_log.txt | head -c 2000)" || true
        echo "Skipping to next shard..."
        continue
    fi

    TILE_COUNT=$(python3 -c "
import json
try:
    d = json.load(open('data/processed/tiles_meta.json'))
    print(len(d['tiles']))
except: print(0)
" 2>/dev/null || echo "0")
    echo "[$(date +%H:%M:%S)] Preprocessed $TILE_COUNT tiles"

    PROC_BYTES=$(dir_bytes "data/processed")
    TOTAL_BYTES=$(( PROC_BYTES + REPLAY_BYTES ))
    python3 -c "
total_gb=${TOTAL_BYTES}/1024**3; ceiling=${MAX_DISK_GB}
proc_gb=${PROC_BYTES}/1024**3; replay_gb=${REPLAY_BYTES}/1024**3
print(f'  Disk after preprocess: processed={proc_gb:.2f} GB  replay={replay_gb:.2f} GB  total={total_gb:.2f} GB / {ceiling} GB')
if total_gb > ceiling*1.02: raise SystemExit(f'[ERROR] Disk ceiling exceeded: {total_gb:.2f} GB > {ceiling} GB')
"

    # ── Save replay ───────────────────────────────────────────────────────
    echo "[$(date +%H:%M:%S)] Saving replay slice for $FOLDER_NAME..."
    python3 -c "
import sys; sys.path.insert(0,'src')
from replay_buffer import save_replay_from_shard
save_replay_from_shard('$FOLDER_NAME')
" || echo "[WARN] Replay save failed — continuing without replay for this shard"

    # ── Train ─────────────────────────────────────────────────────────────
    TRAIN_FLAGS=()
    if [ -n "$INIT_WEIGHTS" ] && [ "$STEP" -eq 1 ]; then
        TRAIN_FLAGS+=("--init-weights" "$INIT_WEIGHTS")
        echo "  Mode: INIT-WEIGHTS (epoch=0, full LR, fresh optimizer)"
    elif [ "$STEP" -gt 1 ] || [ "$START_FROM" -gt 1 ] || [ -n "$PRETRAINED_CKPT" ]; then
        TRAIN_FLAGS+=("--resume")
        echo "  Mode: RESUME (fine-tuning from previous shard / pretrained)"
    else
        echo "  Mode: FRESH (first shard)"
    fi

    echo "[$(date +%H:%M:%S)] Training on $FOLDER_NAME..."
    if ! python src/03_train.py \
            "${TRAIN_FLAGS[@]}" \
            --folder-name  "$FOLDER_NAME" \
            --folder-step  "$STEP" \
            --folder-total "$TOTAL" \
            2>&1 | tee /tmp/train_log.txt; then
        TRAIN_EXIT=${PIPESTATUS[0]}
        echo "[ERROR] Training failed (exit $TRAIN_EXIT)"
        python src/07_notify.py --error \
            --folder "$FOLDER_NAME" --step "$STEP" --total "$TOTAL" \
            --error-msg "$(tail -40 /tmp/train_log.txt | head -c 2000)" || true
        exit 1
    fi

    TRAIN_LOSS=$(grep "tr_loss="  /tmp/train_log.txt | tail -1 | grep -oP "tr_loss=\K[0-9.]+"  || echo "0")
    VAL_LOSS=$(  grep "val_loss=" /tmp/train_log.txt | tail -1 | grep -oP "val_loss=\K[0-9.]+" || echo "0")
    TRAIN_MIOU=$(grep "tr_mIoU="  /tmp/train_log.txt | tail -1 | grep -oP "tr_mIoU=\K[0-9.]+"  || echo "0")
    VAL_MIOU=$(  grep "val_mIoU=" /tmp/train_log.txt | tail -1 | grep -oP "val_mIoU=\K[0-9.]+" || echo "0")
    EPOCHS=$(grep -c "^Epoch " /tmp/train_log.txt 2>/dev/null | tr -d '[:space:]' || echo "0")
    echo "[$(date +%H:%M:%S)] Done — val_mIoU=$VAL_MIOU  epochs=$EPOCHS"

    python src/07_notify.py \
        --folder "$FOLDER_NAME" --step "$STEP" --total "$TOTAL" \
        --train-loss "${TRAIN_LOSS:-0}" --val-loss "${VAL_LOSS:-0}" \
        --train-miou "${TRAIN_MIOU:-0}" --val-miou "${VAL_MIOU:-0}" \
        --epochs "${EPOCHS:-0}" --checkpoint "checkpoints/best_model.pt" || true

    # ── Wipe processed tiles ──────────────────────────────────────────────
    # BUG FIX: DO NOT wipe on the last shard if specialist will run next.
    # Specialist 01_build_specialist_meta.py needs tiles_meta.json +
    # images/masks to exist.  We wipe AFTER the specialist completes.
    if [ "$IS_LAST_FOR_SPECIALIST" -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] Clearing processed tiles (intermediate shard)..."
        rm -rf data/processed/images data/processed/masks data/processed/tiles_meta.json

        REPLAY_BYTES_AFTER=$(dir_bytes "data/replay")
        python3 -c "
replay_gb=${REPLAY_BYTES_AFTER}/1024**3; ceiling=${MAX_DISK_GB}
print(f'  Disk after wipe: replay={replay_gb:.2f} GB  (ceiling={ceiling} GB)')
assert replay_gb < ceiling, f'[ERROR] Replay alone exceeds ceiling!'
"
    else
        echo "[$(date +%H:%M:%S)] Keeping processed tiles for specialist pipeline..."
    fi

    echo "[$(date +%H:%M:%S)] ✓ Finished $FOLDER_NAME ($STEP/$TOTAL)"
done

echo ""
echo "========================================"
echo " All $TOTAL generalist shards complete!"
echo " Generalist checkpoint: checkpoints/best_model.pt"
echo "========================================"

# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIST PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_SPECIALIST" -eq 0 ]; then

    CURRENT_FOLDER="specialist"
    CURRENT_STEP="$TOTAL"

    echo ""
    echo "========================================"
    echo " Specialist Pipeline"
    echo " Training specialist for Bridge / Railway / Utility"
    echo " Disk ceiling: ${MAX_DISK_GB} GB"
    echo "========================================"

    # ── S1: Build specialist dataset metadata ─────────────────────────────
    echo "[$(date +%H:%M:%S)] Building specialist dataset metadata..."
    if ! python specialist/01_build_specialist_meta.py 2>&1 | tee /tmp/specialist_meta_log.txt; then
        echo "[ERROR] Specialist metadata build failed"
        python src/07_notify.py --error \
            --folder "specialist/01_build_specialist_meta" --step "$TOTAL" --total "$TOTAL" \
            --error-msg "$(tail -40 /tmp/specialist_meta_log.txt | head -c 2000)" || true
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] Specialist metadata built."

    # ── S2: Train specialist model ────────────────────────────────────────
    echo "[$(date +%H:%M:%S)] Training specialist model..."
    if ! env MAX_DISK_GB="$MAX_DISK_GB" \
            python specialist/03_train_specialist.py \
            2>&1 | tee /tmp/specialist_train_log.txt; then
        SPEC_TRAIN_EXIT=${PIPESTATUS[0]}
        echo "[ERROR] Specialist training failed (exit $SPEC_TRAIN_EXIT)"
        python src/07_notify.py --error \
            --folder "specialist" --step "$TOTAL" --total "$TOTAL" \
            --error-msg "$(tail -40 /tmp/specialist_train_log.txt | head -c 2000)" || true
        exit 1
    fi

    # ── S3: Parse metrics ─────────────────────────────────────────────────
    SPEC_TRAIN_LOSS=$(grep "tr_loss="  /tmp/specialist_train_log.txt | tail -1 | grep -oP "tr_loss=\K[0-9.]+"  || echo "0")
    SPEC_VAL_LOSS=$(  grep "val_loss=" /tmp/specialist_train_log.txt | tail -1 | grep -oP "val_loss=\K[0-9.]+" || echo "0")
    SPEC_TRAIN_MIOU=$(grep "tr_mIoU="  /tmp/specialist_train_log.txt | tail -1 | grep -oP "tr_mIoU=\K[0-9.]+"  || echo "0")
    SPEC_VAL_MIOU=$(  grep "val_mIoU=" /tmp/specialist_train_log.txt | tail -1 | grep -oP "val_mIoU=\K[0-9.]+" || echo "0")
    SPEC_EPOCHS=$(grep -c "^Epoch " /tmp/specialist_train_log.txt 2>/dev/null | tr -d '[:space:]' || echo "0")

    SPEC_PER_CLASS=$(python3 -c "
import torch, sys
try:
    ckpt = torch.load('specialist/checkpoints/best_model.pt', map_location='cpu', weights_only=False)
    pc   = ckpt.get('per_class_iou', {})
    def g(k): return float(pc.get(k, pc.get(str(k), 0.0)))
    print(f'Bridge={g(4):.4f}  Railway={g(5):.4f}  Utility={g(6):.4f}')
except Exception as e:
    print(f'(per-class read failed: {e})')
" 2>/dev/null || echo "(per-class unavailable)")

    echo "[$(date +%H:%M:%S)] Specialist done — val_mIoU=$SPEC_VAL_MIOU  $SPEC_PER_CLASS"

    # ── S4: Post-specialist wipe ──────────────────────────────────────────
    # BUG FIX: now that the specialist has finished, it is safe to wipe the
    # last shard's processed tiles that we kept alive above.
    if [ -d "data/processed/images" ] || [ -d "data/processed/masks" ] || \
       [ -f "data/processed/tiles_meta.json" ]; then
        echo "[$(date +%H:%M:%S)] Clearing last shard's processed tiles (post-specialist)..."
        rm -rf data/processed/images data/processed/masks data/processed/tiles_meta.json
        REPLAY_BYTES_FINAL=$(dir_bytes "data/replay")
        python3 -c "
replay_gb=${REPLAY_BYTES_FINAL}/1024**3; ceiling=${MAX_DISK_GB}
print(f'  Disk after final wipe: replay={replay_gb:.2f} GB  (ceiling={ceiling} GB)')
"
    fi

    # ── S5: Notify ────────────────────────────────────────────────────────
    python src/07_notify.py \
        --folder "Specialist [${SPEC_PER_CLASS}]" --step "$TOTAL" --total "$TOTAL" \
        --train-loss "${SPEC_TRAIN_LOSS:-0}" --val-loss "${SPEC_VAL_LOSS:-0}" \
        --train-miou "${SPEC_TRAIN_MIOU:-0}" --val-miou "${SPEC_VAL_MIOU:-0}" \
        --epochs "${SPEC_EPOCHS:-0}" --checkpoint "specialist/checkpoints/best_model.pt" || true

    echo ""
    echo "========================================"
    echo " Pipeline complete (generalist + specialist)"
    echo " Generalist : checkpoints/best_model.pt"
    echo " Specialist : specialist/checkpoints/best_model.pt"
    echo ""
    echo " Combined inference:"
    echo "   python specialist/04_inference_combined.py \\"
    echo "       --input /raw_data/image.tif --output outputs/combined_mask.tif"
    echo "========================================"

else
    echo ""
    echo "[SKIP] --skip-specialist set — specialist stage bypassed"
    echo ""
    echo "========================================"
    echo " Pipeline complete (generalist only)"
    echo " Generalist : checkpoints/best_model.pt"
    echo " Inference  : python src/04_inference.py"
    echo "========================================"
fi