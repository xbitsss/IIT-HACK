#!/bin/bash
# 06_train_incremental.sh
# Trains the model folder by folder, resuming from the previous checkpoint.
# Each folder: preprocess → train (resume) → clear tiles → next folder
#
# Usage:
#   bash src/06_train_incremental.sh
#
# Set these to your actual folder paths:
FOLDERS=(
    "/raw_data/CG_1"
    "/raw_data/CG_2"
    "/raw_data/CG_3"
    "/raw_data/CG_4"
)
SHP_DIR="/raw_data/CG_SHP"

echo "========================================"
echo " Incremental Training Pipeline"
echo " Folders: ${#FOLDERS[@]}"
echo " Shapefiles: $SHP_DIR"
echo "========================================"

for i in "${!FOLDERS[@]}"; do
    FOLDER="${FOLDERS[$i]}"
    FOLDER_NAME=$(basename "$FOLDER")
    STEP=$((i + 1))
    TOTAL=${#FOLDERS[@]}

    echo ""
    echo "========================================"
    echo " Step $STEP/$TOTAL: $FOLDER_NAME"
    echo "========================================"

    # ── Preprocess ──
    echo "[$(date +%H:%M:%S)] Preprocessing $FOLDER_NAME..."
    RAW_DATA_DIR="$FOLDER" SHP_DIR="$SHP_DIR" python src/01_preprocess.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] Preprocessing failed for $FOLDER_NAME — skipping"
        continue
    fi

    # ── Train ──
    if [ $STEP -eq 1 ]; then
        echo "[$(date +%H:%M:%S)] Training from scratch on $FOLDER_NAME..."
        python src/03_train.py
    else
        echo "[$(date +%H:%M:%S)] Resuming training on $FOLDER_NAME..."
        python src/03_train.py --resume
    fi

    if [ $? -ne 0 ]; then
        echo "[ERROR] Training failed for $FOLDER_NAME"
        echo "Checkpoint preserved. Fix the issue and re-run with --resume."
        exit 1
    fi

    echo "[$(date +%H:%M:%S)] ✓ Finished $FOLDER_NAME"

    # ── Clear processed tiles to free space ──
    echo "[$(date +%H:%M:%S)] Clearing processed tiles..."
    rm -rf data/processed/images data/processed/masks data/processed/tiles_meta.json
    echo "  ✓ Tiles cleared"
done

echo ""
echo "========================================"
echo " All folders complete!"
echo " Final model: checkpoints/best_model.pt"
echo "========================================"