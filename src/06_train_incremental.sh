#!/bin/bash
# 06_train_incremental.sh — Incremental training pipeline
#
# Usage:
#   bash src/06_train_incremental.sh           # start from CG_1 (fresh)
#   bash src/06_train_incremental.sh --from 2  # start from CG_2 (resume)
#   bash src/06_train_incremental.sh --from 3  # start from CG_3 (resume)

set -euo pipefail  # exit on any error, catch pipeline failures

FOLDERS=(
    "/raw_data/CG_1"
    "/raw_data/CG_2"
    "/raw_data/CG_3"
    "/raw_data/CG_4"
)
SHP_DIR="/raw_data/CG_SHP"
TOTAL=${#FOLDERS[@]}
START_FROM=1  # default: start from folder 1

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)
            START_FROM="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--from N]  (N = folder number to start from, e.g. 2 for CG_2)"
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
echo " Total folders: $TOTAL"
echo " Starting from: CG_$START_FROM"
echo " Shapefiles: $SHP_DIR"
echo "========================================"

# ── Trap for unexpected crashes ───────────────────────────────────────────────
CURRENT_FOLDER="unknown"
CURRENT_STEP=0

crash_handler() {
    EXIT_CODE=$?
    echo ""
    echo "[CRASH] Pipeline crashed unexpectedly at folder: $CURRENT_FOLDER (exit code $EXIT_CODE)"
    python src/07_notify.py --error \
        --folder "$CURRENT_FOLDER" \
        --step "$CURRENT_STEP" \
        --total "$TOTAL" \
        --error-msg "Pipeline crashed unexpectedly (exit code $EXIT_CODE). Check logs. Checkpoint is safe at checkpoints/best_model.pt. Resume with: docker compose run --rm train --resume" \
        || true  # don't fail if notify fails
}
trap crash_handler ERR

# ── Main loop ─────────────────────────────────────────────────────────────────
for i in "${!FOLDERS[@]}"; do
    STEP=$((i + 1))

    # Skip folders before START_FROM
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
    echo " Step $STEP/$TOTAL: $FOLDER_NAME"
    echo "========================================"

    # ── Preprocess ──────────────────────────────────────────────────────────
    echo "[$(date +%H:%M:%S)] Preprocessing $FOLDER_NAME..."
    if ! RAW_DATA_DIR="$FOLDER" SHP_DIR="$SHP_DIR" python src/01_preprocess.py; then
        echo "[ERROR] Preprocessing failed for $FOLDER_NAME"
        python src/07_notify.py --error \
            --folder "$FOLDER_NAME" --step $STEP --total $TOTAL \
            --error-msg "Preprocessing failed for $FOLDER_NAME. Check that TIFFs exist at $FOLDER" \
            || true
        echo "Skipping to next folder..."
        continue
    fi

    # ── Train ────────────────────────────────────────────────────────────────
    echo "[$(date +%H:%M:%S)] Training on $FOLDER_NAME..."
    if [ "$STEP" -eq 1 ] && [ "$START_FROM" -eq 1 ]; then
        echo "  Mode: FRESH (first folder)"
        python src/03_train.py 2>&1 | tee /tmp/train_log.txt
    else
        echo "  Mode: RESUME (continuing from checkpoint)"
        python src/03_train.py --resume 2>&1 | tee /tmp/train_log.txt
    fi
    TRAIN_EXIT=${PIPESTATUS[0]}

    if [ "$TRAIN_EXIT" -ne 0 ]; then
        echo "[ERROR] Training failed for $FOLDER_NAME (exit code $TRAIN_EXIT)"
        python src/07_notify.py --error \
            --folder "$FOLDER_NAME" --step $STEP --total $TOTAL \
            --error-msg "$(tail -30 /tmp/train_log.txt)" \
            || true
        exit 1
    fi

    # ── Parse training stats from log ────────────────────────────────────────
    TRAIN_LOSS=$(grep "train_loss=" /tmp/train_log.txt | tail -1 | grep -oP "train_loss=\K[0-9.]+" || echo "0")
    VAL_LOSS=$(grep "val_loss=" /tmp/train_log.txt | tail -1 | grep -oP "val_loss=\K[0-9.]+" || echo "0")
    TRAIN_MIOU=$(grep "train_mIoU=" /tmp/train_log.txt | tail -1 | grep -oP "train_mIoU=\K[0-9.]+" || echo "0")
    VAL_MIOU=$(grep "val_mIoU=" /tmp/train_log.txt | tail -1 | grep -oP "val_mIoU=\K[0-9.]+" || echo "0")
    EPOCHS=$(grep -c "^Epoch " /tmp/train_log.txt 2>/dev/null || echo "0")
    EPOCHS=$(echo "$EPOCHS" | tr -d '[:space:]')

    echo "[$(date +%H:%M:%S)] Results — val_loss=$VAL_LOSS  val_mIoU=$VAL_MIOU  epochs=$EPOCHS"

    # ── Notify success ───────────────────────────────────────────────────────
    python src/07_notify.py \
        --folder "$FOLDER_NAME" \
        --step $STEP --total $TOTAL \
        --train-loss "${TRAIN_LOSS:-0}" \
        --val-loss "${VAL_LOSS:-0}" \
        --train-miou "${TRAIN_MIOU:-0}" \
        --val-miou "${VAL_MIOU:-0}" \
        --epochs "${EPOCHS:-0}" \
        --checkpoint "checkpoints/best_model.pt" \
        || true

    # ── Clear tiles to free disk space ───────────────────────────────────────
    echo "[$(date +%H:%M:%S)] Clearing processed tiles..."
    rm -rf data/processed/images data/processed/masks data/processed/tiles_meta.json
    echo "  Tiles cleared — disk freed"

    echo "[$(date +%H:%M:%S)] ✓ Finished $FOLDER_NAME ($STEP/$TOTAL)"
done

echo ""
echo "========================================"
echo " All folders complete!"
echo " Final model: checkpoints/best_model.pt"
echo "========================================"