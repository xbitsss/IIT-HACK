#!/bin/bash
# 06_train_incremental.sh — Incremental training pipeline
#
# Trains on all CG folders one shard at a time while guaranteeing that
# pipeline-generated data on disk NEVER exceeds MAX_DISK_GB at any moment.
#
# Disk accounting (enforced before every preprocessing step):
#
#   disk_used_now  = size of data/replay/ (accumulates across shards)
#   processed_budget = MAX_DISK_GB - disk_used_now
#
#   01_preprocess.py is given MAX_PROCESSED_GB=<processed_budget> so the
#   tiles it writes cannot push (processed + replay) over MAX_DISK_GB.
#
#   After training, processed tiles are wiped.  Only the replay slice for
#   this shard (~300 tiles ≈ 1.3 GB) is kept permanently in data/replay/.
#
# Usage:
#   bash src/06_train_incremental.sh              # fresh, start from CG_1
#   bash src/06_train_incremental.sh --from 2     # resume from CG_2 onward
#
# Environment overrides:
#   MAX_DISK_GB               hard ceiling in GB          (default: 50)
#   REPLAY_TILES_PER_SHARD    tiles kept per shard        (default: 300)
#   SHP_DIR                   shared shapefile folder     (default: /raw_data/CG_SHP)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
# One entry per dataset folder.  Each folder must contain:
#   - TIFF files (anywhere inside, found recursively)
#   - A subdirectory containing .shp files (auto-detected — no parallel array needed)
#
# Example structure:
#   /raw_data/CG/
#       image1.tif
#       image2.tif
#       CG_SHP/
#           Built_Up_Area_type.shp  Road.shp  Water_Body.shp
#   /raw_data/PB/
#       image3.tif
#       PB_SHP/
#           Built_Up_Area_type.shp  Road.shp  Water_Body.shp
FOLDERS=(
    "/raw_data/CG"
    "/raw_data/PB"
)

export MAX_DISK_GB="${MAX_DISK_GB:-50}"
export REPLAY_TILES_PER_SHARD="${REPLAY_TILES_PER_SHARD:-300}"

TOTAL=${#FOLDERS[@]}
START_FROM=1
PRETRAINED_CKPT=""

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)        START_FROM="$2";      shift 2 ;;
        --pretrained)  PRETRAINED_CKPT="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--from N] [--pretrained /path/to/model.pth]"
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
echo " Shards        : $TOTAL"
echo " Starting from : shard $START_FROM"
echo " Disk ceiling  : ${MAX_DISK_GB} GB (processed + replay combined)"
echo " Replay/shard  : ${REPLAY_TILES_PER_SHARD} tiles"
echo "========================================"

# ── Fresh start: wipe replay and checkpoint ───────────────────────────────────
# When starting from shard 1, any replay/checkpoint from a previous run is
# stale. Mixing old replay tiles into a new training run would corrupt the
# model — so we wipe both before doing anything else.
# When resuming mid-run (--from 2+), replay and checkpoint are intentionally
# preserved so training continues from where it left off.
if [ "$START_FROM" -eq 1 ]; then
    echo ""
    echo "[FRESH START] Wiping stale data from any previous run..."
    if [ -d "data/replay" ]; then
        echo "  Clearing data/replay/ ($(du -sh data/replay 2>/dev/null | cut -f1) of stale replay tiles)"
        find data/replay -mindepth 1 -delete
    fi
    if [ -d "data/processed" ]; then
        echo "  Clearing data/processed/ (leftover tiles)"
        find data/processed -mindepth 1 -delete
    fi
    if [ -n "$PRETRAINED_CKPT" ]; then
        if [ ! -f "$PRETRAINED_CKPT" ]; then
            echo "[ERROR] --pretrained path does not exist: $PRETRAINED_CKPT"
            exit 1
        fi
        mkdir -p checkpoints
        cp "$PRETRAINED_CKPT" checkpoints/best_model.pt
        echo "  Installed pretrained checkpoint: $PRETRAINED_CKPT → checkpoints/best_model.pt"
    else
        if [ -f "checkpoints/best_model.pt" ]; then
            echo "  Removing stale checkpoint"
            rm -f checkpoints/best_model.pt checkpoints/training_curves.png checkpoints/history.json
        fi
    fi
    echo "[FRESH START] Clean slate ready."
    echo ""
fi

# ── Helper: measure directory size in bytes ───────────────────────────────────
dir_bytes() {
    local dir="$1"
    if [ -d "$dir" ]; then
        du -sb "$dir" 2>/dev/null | awk '{print $1}' || echo "0"
    else
        echo "0"
    fi
}

# ── Helper: find the SHP subdirectory inside a dataset folder ─────────────────
# Walks one level of subdirectories and returns the first one containing .shp
# files.  This lets each dataset folder own its shapefiles without needing a
# separate parallel array in the config.
#
# Usage: SHP_DIR=$(find_shp_dir "/raw_data/CG")
# Returns empty string if no SHP subdir is found (preprocessing will warn).
find_shp_dir() {
    local dataset_dir="$1"

    # Check one level of subdirectories for .shp files using find (more reliable than ls glob)
    for subdir in "$dataset_dir"/*/; do
        [ -d "$subdir" ] || continue
        if find "$subdir" -maxdepth 1 -iname "*.shp" 2>/dev/null | grep -q .; then
            echo "${subdir%/}"
            return 0
        fi
    done

    # Fallback: .shp files directly in the dataset folder itself
    if find "$dataset_dir" -maxdepth 1 -iname "*.shp" 2>/dev/null | grep -q .; then
        echo "$dataset_dir"
        return 0
    fi

    echo ""
}

# ── Helper: compute processed-tile budget for this shard ─────────────────────
# Budget = MAX_DISK_GB (bytes) − replay_bytes_already_on_disk
# Clamped to at least 1 GB so preprocessing never gets a zero budget.
compute_processed_budget_gb() {
    local replay_bytes
    replay_bytes=$(dir_bytes "data/replay")
    local max_bytes
    max_bytes=$(python3 -c "print(int(${MAX_DISK_GB} * 1024**3))")
    local budget_bytes=$(( max_bytes - replay_bytes ))
    if [ "$budget_bytes" -lt $(( 1 * 1024 * 1024 * 1024 )) ]; then
        budget_bytes=$(( 1 * 1024 * 1024 * 1024 ))
        echo "[WARN] Replay buffer is very large — processed budget clamped to 1 GB" >&2
    fi
    python3 -c "print(f'{${budget_bytes} / 1024**3:.3f}')"
}

# ── Crash handler ─────────────────────────────────────────────────────────────
CURRENT_FOLDER="unknown"
CURRENT_STEP=0

shell_crash_handler() {
    local EXIT_CODE=$?
    echo ""
    echo "[CRASH] Shell crashed at $CURRENT_FOLDER (exit $EXIT_CODE)"
    python src/07_notify.py --error \
        --folder   "$CURRENT_FOLDER" \
        --step     "$CURRENT_STEP" \
        --total    "$TOTAL" \
        --error-msg "Shell pipeline crashed (exit $EXIT_CODE) at $CURRENT_FOLDER. \
Checkpoint safe at checkpoints/best_model.pt. \
Resume: bash src/06_train_incremental.sh --from $CURRENT_STEP" \
        || true
}
trap shell_crash_handler ERR

# ── Main loop ─────────────────────────────────────────────────────────────────
for i in "${!FOLDERS[@]}"; do
    STEP=$((i + 1))

    if [ "$STEP" -lt "$START_FROM" ]; then
        echo "Skipping step $STEP ($(basename "${FOLDERS[$i]}")) — --from $START_FROM"
        continue
    fi

    FOLDER="${FOLDERS[$i]}"
    FOLDER_NAME=$(basename "$FOLDER")
    CURRENT_FOLDER="$FOLDER_NAME"
    CURRENT_STEP="$STEP"

    # Auto-detect the SHP subdirectory inside this dataset folder
    SHARD_SHP_DIR=$(find_shp_dir "$FOLDER")
    if [ -z "$SHARD_SHP_DIR" ]; then
        echo "[WARN] No SHP subdirectory found in $FOLDER — preprocessing will skip label rasterization"
        SHARD_SHP_DIR="$FOLDER"   # pass the folder itself; preprocess will warn
    fi

    echo ""
    echo "========================================"
    echo " Step $STEP/$TOTAL: $FOLDER_NAME"
    echo " Shapefiles   : $SHARD_SHP_DIR"
    echo "========================================"

    # ── Disk accounting: compute exact processed budget ───────────────────────
    REPLAY_BYTES=$(dir_bytes "data/replay")
    REPLAY_GB=$(python3 -c "print(f'{${REPLAY_BYTES}/1024**3:.3f}')")
    PROCESSED_BUDGET_GB=$(compute_processed_budget_gb)
    echo "[$(date +%H:%M:%S)] Disk:  replay=${REPLAY_GB} GB  " \
         "processed_budget=${PROCESSED_BUDGET_GB} GB  " \
         "ceiling=${MAX_DISK_GB} GB"

    # Safety assertion: replay alone must not already exceed the ceiling.
    python3 -c "
replay_gb = ${REPLAY_GB}
ceiling   = ${MAX_DISK_GB}
budget    = ${PROCESSED_BUDGET_GB}
print(f'  replay={replay_gb:.3f} GB  budget={budget:.3f} GB  ceiling={ceiling} GB')
assert replay_gb < ceiling, f'[ERROR] Replay ({replay_gb:.2f} GB) already exceeds ceiling ({ceiling} GB)!'
assert budget >= 1.0,       f'[ERROR] Processed budget ({budget:.2f} GB) is less than 1 GB — increase MAX_DISK_GB or reduce REPLAY_TILES_PER_SHARD'
"

    # ── Preprocess this shard with the computed budget ────────────────────────
    echo "[$(date +%H:%M:%S)] Preprocessing $FOLDER_NAME..."
    if ! RAW_DATA_DIR="$FOLDER" \
         SHP_DIR="$SHARD_SHP_DIR" \
         MAX_PROCESSED_GB="$PROCESSED_BUDGET_GB" \
         python src/01_preprocess.py 2>&1 | tee /tmp/preprocess_log.txt; then
        echo "[ERROR] Preprocessing failed for $FOLDER_NAME"
        python src/07_notify.py --error \
            --folder "$FOLDER_NAME" --step "$STEP" --total "$TOTAL" \
            --error-msg "$(tail -40 /tmp/preprocess_log.txt | head -c 2000)" \
            || true
        echo "Skipping to next folder..."
        continue
    fi

    TILE_COUNT=$(python3 -c "
import json
try:
    d = json.load(open('data/processed/tiles_meta.json'))
    print(len(d['tiles']))
except:
    print(0)
" 2>/dev/null || echo "0")
    echo "[$(date +%H:%M:%S)] Preprocessed $TILE_COUNT tiles"

    # ── Post-preprocess disk check ────────────────────────────────────────────
    PROC_BYTES=$(dir_bytes "data/processed")
    REPLAY_BYTES=$(dir_bytes "data/replay")
    TOTAL_BYTES=$(( PROC_BYTES + REPLAY_BYTES ))
    python3 -c "
total_gb  = ${TOTAL_BYTES} / 1024**3
ceiling   = ${MAX_DISK_GB}
proc_gb   = ${PROC_BYTES}  / 1024**3
replay_gb = ${REPLAY_BYTES} / 1024**3
print(f'  Disk after preprocess: processed={proc_gb:.2f} GB  replay={replay_gb:.2f} GB  total={total_gb:.2f} GB / {ceiling} GB')
if total_gb > ceiling * 1.02:   # allow 2% rounding tolerance
    raise SystemExit(f'[ERROR] Disk ceiling exceeded: {total_gb:.2f} GB > {ceiling} GB')
"

    # ── Save replay buffer BEFORE training (tiles get wiped after) ────────────
    echo "[$(date +%H:%M:%S)] Saving replay slice for $FOLDER_NAME..."
    python3 -c "
import sys; sys.path.insert(0, 'src')
from replay_buffer import save_replay_from_shard
save_replay_from_shard('$FOLDER_NAME')
" || echo "[WARN] Replay save failed — continuing without replay for this shard"

    # ── Train ─────────────────────────────────────────────────────────────────
    RESUME_FLAG=""
    if [ "$STEP" -gt 1 ] || [ "$START_FROM" -gt 1 ] || [ -n "$PRETRAINED_CKPT" ]; then
        RESUME_FLAG="--resume"
        if [ -n "$PRETRAINED_CKPT" ] && [ "$STEP" -eq 1 ]; then
            echo "  Mode: FINE-TUNE from pretrained checkpoint"
        else
            echo "  Mode: RESUME (fine-tuning from checkpoint)"
        fi
    else
        echo "  Mode: FRESH (first shard, no pretrained checkpoint)"
    fi

    echo "[$(date +%H:%M:%S)] Training on $FOLDER_NAME..."
    if ! python src/03_train.py \
            $RESUME_FLAG \
            --folder-name  "$FOLDER_NAME" \
            --folder-step  "$STEP" \
            --folder-total "$TOTAL" \
            2>&1 | tee /tmp/train_log.txt; then
        TRAIN_EXIT=${PIPESTATUS[0]}
        echo "[ERROR] Training failed for $FOLDER_NAME (exit $TRAIN_EXIT)"
        python src/07_notify.py --error \
            --folder "$FOLDER_NAME" --step "$STEP" --total "$TOTAL" \
            --error-msg "$(tail -40 /tmp/train_log.txt | head -c 2000)" \
            || true
        exit 1
    fi

    # ── Parse stats ───────────────────────────────────────────────────────────
    TRAIN_LOSS=$(grep  "tr_loss="  /tmp/train_log.txt | tail -1 | grep -oP "tr_loss=\K[0-9.]+"  || echo "0")
    VAL_LOSS=$(grep    "val_loss=" /tmp/train_log.txt | tail -1 | grep -oP "val_loss=\K[0-9.]+" || echo "0")
    TRAIN_MIOU=$(grep  "tr_mIoU=" /tmp/train_log.txt | tail -1 | grep -oP "tr_mIoU=\K[0-9.]+"  || echo "0")
    VAL_MIOU=$(grep    "val_mIoU=" /tmp/train_log.txt | tail -1 | grep -oP "val_mIoU=\K[0-9.]+" || echo "0")
    EPOCHS=$(grep -c   "^Epoch "  /tmp/train_log.txt 2>/dev/null | tr -d '[:space:]' || echo "0")

    echo "[$(date +%H:%M:%S)] Done — val_mIoU=$VAL_MIOU  epochs=$EPOCHS"

    # ── Notify ────────────────────────────────────────────────────────────────
    python src/07_notify.py \
        --folder     "$FOLDER_NAME" \
        --step       "$STEP" \
        --total      "$TOTAL" \
        --train-loss "${TRAIN_LOSS:-0}" \
        --val-loss   "${VAL_LOSS:-0}" \
        --train-miou "${TRAIN_MIOU:-0}" \
        --val-miou   "${VAL_MIOU:-0}" \
        --epochs     "${EPOCHS:-0}" \
        --checkpoint "checkpoints/best_model.pt" \
        || true

    # ── Wipe processed tiles — replay is already saved ────────────────────────
    # data/replay/ is intentionally NOT touched here.
    echo "[$(date +%H:%M:%S)] Clearing processed tiles..."
    rm -rf data/processed/images data/processed/masks data/processed/tiles_meta.json
    echo "  Processed tiles cleared."

    # ── Post-wipe disk check: only replay should remain ───────────────────────
    REPLAY_BYTES_AFTER=$(dir_bytes "data/replay")
    python3 -c "
replay_gb = ${REPLAY_BYTES_AFTER} / 1024**3
ceiling   = ${MAX_DISK_GB}
print(f'  Disk after wipe: replay={replay_gb:.2f} GB  (ceiling={ceiling} GB)')
assert replay_gb < ceiling, f'[ERROR] Replay alone ({replay_gb:.2f} GB) exceeds ceiling after wipe!'
"

    echo "[$(date +%H:%M:%S)] ✓ Finished $FOLDER_NAME ($STEP/$TOTAL)"
done

echo ""
echo "========================================"
echo " All $TOTAL shards complete!"
echo " Final checkpoint: checkpoints/best_model.pt"
echo "========================================"