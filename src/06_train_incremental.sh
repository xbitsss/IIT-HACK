#!/bin/bash
# 06_train_incremental.sh — Incremental training pipeline
#
# The dataset is sharded across 4 folders, each containing 2 TIFFs.
# All shapefiles are in CG_SHP (shared across all folders).
#
# Usage:
#   bash src/06_train_incremental.sh           # start fresh from CG_1
#   bash src/06_train_incremental.sh --from 2  # resume from CG_2 onward
#   bash src/06_train_incremental.sh --from 3  # resume from CG_3 onward
#
# After every folder the tile cache (data/processed/) is wiped to free disk.
# The model checkpoint (checkpoints/best_model.pt) is NEVER deleted.
# The next folder always uses --resume so it fine-tunes from the last best model.

set -euo pipefail

FOLDERS=(
    "/raw_data/CG_1"
    "/raw_data/CG_2"
    "/raw_data/CG_3"
    "/raw_data/CG_4"
)
SHP_DIR="/raw_data/CG_SHP"
TOTAL=${#FOLDERS[@]}
START_FROM=1

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)
            START_FROM="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--from N]"
            exit 1
            ;;
    esac
done

if [ "$START_FROM" -lt 1 ] || [ "$START_FROM" -gt "$TOTAL" ]; then
    echo "[ERROR] --from must be between 1 and $TOTAL"
    exit 1
fi

echo "========================================"
echo " Incremental Training Pipeline"
echo " Total folders : $TOTAL (each has 2 TIFFs)"
echo " Starting from : CG_$START_FROM"
echo " Shapefiles    : $SHP_DIR"
echo "========================================"

# ── Crash handler ─────────────────────────────────────────────────────────────
# This fires when the shell script itself crashes (e.g., disk full, OOM killer).
# The Python train process has its own signal handler for in-process crashes.
CURRENT_FOLDER="unknown"
CURRENT_STEP=0

shell_crash_handler() {
    EXIT_CODE=$?
    echo ""
    echo "[CRASH] Shell script crashed at folder: $CURRENT_FOLDER (exit $EXIT_CODE)"
    python src/07_notify.py --error \
        --folder   "$CURRENT_FOLDER" \
        --step     "$CURRENT_STEP" \
        --total    "$TOTAL" \
        --error-msg "Shell pipeline crashed (exit $EXIT_CODE) at $CURRENT_FOLDER. \
Checkpoint safe at checkpoints/best_model.pt. \
Resume: docker compose run --rm train-all --from $CURRENT_STEP" \
        || true
}
trap shell_crash_handler ERR

# ── Main loop ─────────────────────────────────────────────────────────────────
for i in "${!FOLDERS[@]}"; do
    STEP=$((i + 1))

    if [ "$STEP" -lt "$START_FROM" ]; then
        echo "Skipping CG_$STEP (--from $START_FROM)"
        continue
    fi

    FOLDER="${FOLDERS[$i]}"
    FOLDER_NAME=$(basename "$FOLDER")
    CURRENT_FOLDER="$FOLDER_NAME"
    CURRENT_STEP="$STEP"

    echo ""
    echo "========================================"
    echo " Step $STEP/$TOTAL: $FOLDER_NAME  (2 TIFFs)"
    echo "========================================"

    # ── Preprocess (this folder only) ────────────────────────────────────────
    echo "[$(date +%H:%M:%S)] Preprocessing $FOLDER_NAME..."
    if ! RAW_DATA_DIR="$FOLDER" SHP_DIR="$SHP_DIR" python src/01_preprocess.py; then
        echo "[ERROR] Preprocessing failed for $FOLDER_NAME"
        python src/07_notify.py --error \
            --folder "$FOLDER_NAME" --step $STEP --total $TOTAL \
            --error-msg "Preprocessing failed for $FOLDER_NAME. \
Check TIFFs at $FOLDER and shapefiles at $SHP_DIR." \
            || true
        echo "Skipping to next folder..."
        continue
    fi

    # Count tiles produced
    TILE_COUNT=$(python -c "
import json; d=json.load(open('data/processed/tiles_meta.json'))
print(len(d['tiles']))
" 2>/dev/null || echo "?")
    echo "[$(date +%H:%M:%S)] Preprocessing done — $TILE_COUNT tiles"

    # ── Save replay buffer from this shard ───────────────────────────────────
    # MUST happen BEFORE training (tiles are wiped afterwards).
    # The replay dir is persistent — tiles from all previous shards are retained.
    echo "[$(date +%H:%M:%S)] Saving replay buffer for $FOLDER_NAME..."
    python -c "
import sys; sys.path.insert(0, 'src')
from replay_buffer import save_replay_from_shard
save_replay_from_shard('$FOLDER_NAME')
    " || echo "[WARN] Replay save failed — continuing without replay for this shard"

    # ── Train ────────────────────────────────────────────────────────────────
    echo "[$(date +%H:%M:%S)] Training on $FOLDER_NAME..."

    # First folder from scratch if START_FROM==1; all others resume
    RESUME_FLAG=""
    if [ "$STEP" -gt 1 ] || [ "$START_FROM" -gt 1 ]; then
        RESUME_FLAG="--resume"
        echo "  Mode: RESUME (fine-tuning from checkpoint)"
    else
        echo "  Mode: FRESH (first folder)"
    fi

    # Pass folder context to train.py so its notification thread can include it
    if ! python src/03_train.py \
            $RESUME_FLAG \
            --folder-name  "$FOLDER_NAME" \
            --folder-step  "$STEP" \
            --folder-total "$TOTAL" \
            2>&1 | tee /tmp/train_log.txt; then
        TRAIN_EXIT=${PIPESTATUS[0]}
        echo "[ERROR] Training failed for $FOLDER_NAME (exit $TRAIN_EXIT)"
        python src/07_notify.py --error \
            --folder "$FOLDER_NAME" --step $STEP --total $TOTAL \
            --error-msg "$(tail -40 /tmp/train_log.txt)" \
            || true
        exit 1
    fi

    # ── Parse final stats ─────────────────────────────────────────────────────
    TRAIN_LOSS=$(grep "tr_loss="    /tmp/train_log.txt | tail -1 | grep -oP "tr_loss=\K[0-9.]+"    || echo "0")
    VAL_LOSS=$(grep   "val_loss="   /tmp/train_log.txt | tail -1 | grep -oP "val_loss=\K[0-9.]+"   || echo "0")
    TRAIN_MIOU=$(grep "tr_mIoU="   /tmp/train_log.txt | tail -1 | grep -oP "tr_mIoU=\K[0-9.]+"   || echo "0")
    VAL_MIOU=$(grep   "val_mIoU="  /tmp/train_log.txt | tail -1 | grep -oP "val_mIoU=\K[0-9.]+"  || echo "0")
    BEST_MIOU=$(grep  "Best val_mIoU=" /tmp/train_log.txt | tail -1 | grep -oP "Best val_mIoU=\K[0-9.]+" || echo "$VAL_MIOU")
    EPOCHS=$(grep -c "^Epoch " /tmp/train_log.txt 2>/dev/null | tr -d '[:space:]' || echo "0")

    echo "[$(date +%H:%M:%S)] Results — val_mIoU=$VAL_MIOU  best=$BEST_MIOU  epochs=$EPOCHS"

    # ── Notify folder complete ────────────────────────────────────────────────
    python src/07_notify.py \
        --folder        "$FOLDER_NAME" \
        --step          $STEP \
        --total         $TOTAL \
        --train-loss    "${TRAIN_LOSS:-0}" \
        --val-loss      "${VAL_LOSS:-0}" \
        --train-miou    "${TRAIN_MIOU:-0}" \
        --val-miou      "${VAL_MIOU:-0}" \
        --epochs        "${EPOCHS:-0}" \
        --checkpoint    "checkpoints/best_model.pt" \
        || true

    # ── Wipe tile cache to free disk space ───────────────────────────────────
    # NOTE: Only data/processed/ is wiped.  data/replay/ is NEVER deleted —
    # it contains the replay tiles needed to prevent catastrophic forgetting
    # on all future shards.
    echo "[$(date +%H:%M:%S)] Clearing processed tiles (freeing disk)..."
    rm -rf data/processed/images data/processed/masks data/processed/tiles_meta.json
    echo "  Tiles cleared. (Replay buffer at data/replay/ is preserved.)"

    echo "[$(date +%H:%M:%S)] ✓ Done $FOLDER_NAME ($STEP/$TOTAL)"
done

echo ""
echo "========================================"
echo " All $TOTAL folders complete!"
echo " Final model: checkpoints/best_model.pt"
echo "========================================"