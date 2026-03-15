#!/bin/bash
FOLDERS=(
    "/raw_data/CG_1"
    "/raw_data/CG_2"
    "/raw_data/CG_3"
    "/raw_data/CG_4"
)
SHP_DIR="/raw_data/CG_SHP"
TOTAL=${#FOLDERS[@]}

echo "========================================"
echo " Incremental Training Pipeline"
echo " Folders: $TOTAL"
echo " Shapefiles: $SHP_DIR"
echo "========================================"

for i in "${!FOLDERS[@]}"; do
    FOLDER="${FOLDERS[$i]}"
    FOLDER_NAME=$(basename "$FOLDER")
    STEP=$((i + 1))

    echo ""
    echo "========================================"
    echo " Step $STEP/$TOTAL: $FOLDER_NAME"
    echo "========================================"

    # ── Preprocess ──
    echo "[$(date +%H:%M:%S)] Preprocessing $FOLDER_NAME..."
    RAW_DATA_DIR="$FOLDER" SHP_DIR="$SHP_DIR" python src/01_preprocess.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] Preprocessing failed for $FOLDER_NAME"
        python src/07_notify.py --error \
            --folder "$FOLDER_NAME" --step $STEP --total $TOTAL \
            --error-msg "Preprocessing failed for $FOLDER_NAME"
        continue
    fi

    # ── Train ──
    echo "[$(date +%H:%M:%S)] Training on $FOLDER_NAME..."
    if [ $STEP -eq 1 ]; then
        python src/03_train.py 2>&1 | tee /tmp/train_log.txt
    else
        python src/03_train.py --resume 2>&1 | tee /tmp/train_log.txt
    fi
    TRAIN_EXIT=$?

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "[ERROR] Training failed for $FOLDER_NAME"
        python src/07_notify.py --error \
            --folder "$FOLDER_NAME" --step $STEP --total $TOTAL \
            --error-msg "$(tail -20 /tmp/train_log.txt)"
        exit 1
    fi

    # ── Parse stats from log ──
    TRAIN_LOSS=$(grep "train_loss=" /tmp/train_log.txt | tail -1 | grep -oP "train_loss=\K[0-9.]+")
    VAL_LOSS=$(grep "val_loss=" /tmp/train_log.txt | tail -1 | grep -oP "val_loss=\K[0-9.]+")
    TRAIN_MIOU=$(grep "train_mIoU=" /tmp/train_log.txt | tail -1 | grep -oP "train_mIoU=\K[0-9.]+")
    VAL_MIOU=$(grep "val_mIoU=" /tmp/train_log.txt | tail -1 | grep -oP "val_mIoU=\K[0-9.]+")
    EPOCHS=$(grep -c "Epoch " /tmp/train_log.txt || echo "0")

    # ── Notify ──
    python src/07_notify.py \
        --folder "$FOLDER_NAME" \
        --step $STEP --total $TOTAL \
        --train-loss "${TRAIN_LOSS:-0}" \
        --val-loss "${VAL_LOSS:-0}" \
        --train-miou "${TRAIN_MIOU:-0}" \
        --val-miou "${VAL_MIOU:-0}" \
        --epochs "${EPOCHS:-0}" \
        --checkpoint "checkpoints/best_model.pt"

    # ── Clear tiles ──
    echo "[$(date +%H:%M:%S)] Clearing processed tiles..."
    rm -rf data/processed/images data/processed/masks data/processed/tiles_meta.json
    echo "  Tiles cleared"
done

echo ""
echo "========================================"
echo " All folders complete!"
echo " Final model: checkpoints/best_model.pt"
echo "========================================"