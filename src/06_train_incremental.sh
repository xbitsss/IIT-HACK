#!/bin/bash
# 06_train_incremental.sh — Incremental training pipeline
#
# TWO APPROACHES:
#
# Approach A (default) — subfolder per shard, each with its own SHP/:
#   /raw_data/
#       CG/   *.tif   SHP/*.shp
#       PB/   *.tif   SHP/*.shp
#   Auto-discovered — just drop folders in.
#   Usage: docker compose run --rm train-all
#
# Approach B — all TIFFs in one folder, 2 SHP folders, random 50/50 split:
#   /raw_data/ALL/
#       *.tif  (all images)
#       SHP1/  *.shp
#       SHP2/  *.shp
#   Files reshuffled randomly every run — shard 1 & 2 see different files each time.
#   Usage: docker compose run --rm train-all --approach b --data-dir /raw_data/ALL
#          docker compose run --rm train-all --approach b --data-dir /raw_data/ALL \
#              --shp-dirs /raw_data/ALL/SHP1:/raw_data/ALL/SHP2
#
# FLAGS:
#   --from N                   Resume from shard N (skips shards 1..N-1)
#   --pretrained /path.pth     Fine-tune from checkpoint (LR×0.3, continues epoch count)
#   --init-weights /path.pth   Load weights only — train from epoch 0, full LR,
#                              fresh optimizer. Use to start from custom backbone.
#   --approach a|b             Select approach (default: a)
#   --data-dir /path           Root folder for approach B (default: /raw_data/ALL)
#   --shp-dirs dir1:dir2       Colon-separated SHP dirs for approach B
#                              (default: auto-detect SHP/ subdirs inside --data-dir)
#
# Environment overrides:
#   RAW_DATA_ROOT              Root folder scanned for approach A  (default: /raw_data)
#   MAX_DISK_GB                Hard disk ceiling in GB             (default: 50)
#   REPLAY_TILES_PER_SHARD     Tiles kept per shard in replay      (default: 300)

set -euo pipefail

RAW_DATA_ROOT="${RAW_DATA_ROOT:-/raw_data}"
export MAX_DISK_GB="${MAX_DISK_GB:-50}"
export REPLAY_TILES_PER_SHARD="${REPLAY_TILES_PER_SHARD:-300}"

APPROACH="a"
DATA_DIR_B="/raw_data/ALL"
SHP_DIRS_B=""
PRETRAINED_CKPT=""
INIT_WEIGHTS=""
START_FROM=1
SKIP_SPECIALIST=0

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)            START_FROM="$2";     shift 2 ;;
        --pretrained)      PRETRAINED_CKPT="$2"; shift 2 ;;
        --init-weights)    INIT_WEIGHTS="$2";   shift 2 ;;
        --approach)        APPROACH="$2";       shift 2 ;;
        --data-dir)        DATA_DIR_B="$2";     shift 2 ;;
        --shp-dirs)        SHP_DIRS_B="$2";     shift 2 ;;
        --skip-specialist) SKIP_SPECIALIST=1;   shift   ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--from N] [--pretrained /path.pth] [--init-weights /path.pth]"
            echo "          [--approach a|b] [--data-dir /path] [--shp-dirs dir1:dir2]"
            echo "          [--skip-specialist]"
            exit 1
            ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
dir_bytes() {
    local dir="$1"
    if [ -d "$dir" ]; then
        du -sb "$dir" 2>/dev/null | awk '{print $1}' || echo "0"
    else
        echo "0"
    fi
}

# Find SHP subdir: looks for a folder named SHP (case-insensitive) first,
# then any subdir containing .shp files.
find_shp_dir() {
    local dataset_dir="$1"
    for subdir in "$dataset_dir"/*/; do
        [ -d "$subdir" ] || continue
        local bname
        bname=$(basename "$subdir")
        if [ "${bname^^}" = "SHP" ]; then
            echo "${subdir%/}"; return 0
        fi
    done
    for subdir in "$dataset_dir"/*/; do
        [ -d "$subdir" ] || continue
        if find "$subdir" -maxdepth 1 -iname "*.shp" 2>/dev/null | grep -q .; then
            echo "${subdir%/}"; return 0
        fi
    done
    if find "$dataset_dir" -maxdepth 1 -iname "*.shp" 2>/dev/null | grep -q .; then
        echo "$dataset_dir"; return 0
    fi
    echo ""
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

# ── Build FOLDERS array ───────────────────────────────────────────────────────
if [ "$APPROACH" = "b" ]; then
    # Approach B: random 50/50 split of all TIFFs in DATA_DIR_B
    if [ ! -d "$DATA_DIR_B" ]; then
        echo "[ERROR] --data-dir '$DATA_DIR_B' does not exist."
        exit 1
    fi

    # Collect all TIFFs, shuffle with a fresh random seed every run
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

    # Auto-detect SHP dirs if not specified
    if [ -z "$SHP_DIRS_B" ]; then
        SHP_DIRS_B=""
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

    FOLDERS=("$DATA_DIR_B" "$DATA_DIR_B")   # same parent dir, different TIFF subsets
    TIFF_FILES_PER_SHARD=("$SHARD1_FILES" "$SHARD2_FILES")
    TOTAL=2

    echo "========================================"
    echo " Incremental Training Pipeline — Approach B"
    echo " Data dir      : $DATA_DIR_B"
    echo " Total TIFFs   : $N_TIFS  →  shard 1: $HALF  shard 2: $(( N_TIFS - HALF ))"
    echo " SHP dirs      : $SHP_DIRS_B"
    echo " Starting from : shard $START_FROM"
    echo " Disk ceiling  : ${MAX_DISK_GB} GB"
    echo " Replay/shard  : ${REPLAY_TILES_PER_SHARD} tiles"
    echo " Note          : TIFF split is random — different files every run"
    echo "========================================"

else
    # Approach A: each subfolder of RAW_DATA_ROOT = one shard
    mapfile -t FOLDERS < <(
        find "$RAW_DATA_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | while read -r dir; do
            if find "$dir" -maxdepth 3 \( -iname "*.tif" -o -iname "*.tiff" \) 2>/dev/null | grep -q .; then
                echo "$dir"
            fi
        done
    )
    TIFF_FILES_PER_SHARD=()
    for _ in "${FOLDERS[@]:-}"; do TIFF_FILES_PER_SHARD+=(""); done

    if [ ${#FOLDERS[@]} -eq 0 ]; then
        echo "[ERROR] No shard folders found in $RAW_DATA_ROOT"
        echo "  Each subfolder must contain at least one .tif/.tiff file."
        exit 1
    fi
    TOTAL=${#FOLDERS[@]}

    echo "========================================"
    echo " Incremental Training Pipeline — Approach A"
    echo " Root          : $RAW_DATA_ROOT"
    echo " Shards found  : $TOTAL"
    for i in "${!FOLDERS[@]}"; do
        echo "   $((i+1)). $(basename "${FOLDERS[$i]}")"
    done
    echo " Starting from : shard $START_FROM"
    echo " Disk ceiling  : ${MAX_DISK_GB} GB"
    echo " Replay/shard  : ${REPLAY_TILES_PER_SHARD} tiles"
    echo "========================================"
fi

if [ "$START_FROM" -lt 1 ] || [ "$START_FROM" -gt "$TOTAL" ]; then
    echo "[ERROR] --from must be between 1 and $TOTAL"
    exit 1
fi

# ── Fresh start ───────────────────────────────────────────────────────────────
if [ "$START_FROM" -eq 1 ]; then
    echo ""
    echo "[FRESH START] Wiping stale data..."
    [ -d "data/replay" ]    && find data/replay -mindepth 1 -delete    && echo "  Cleared data/replay/"
    [ -d "data/processed" ] && find data/processed -mindepth 1 -delete && echo "  Cleared data/processed/"

    if [ -n "$INIT_WEIGHTS" ]; then
        # init-weights: validate path but DO NOT copy to checkpoint.
        # 03_train.py will load weights-only and start fresh training.
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

# ── Main loop ─────────────────────────────────────────────────────────────────
for i in "${!FOLDERS[@]}"; do
    STEP=$((i + 1))

    if [ "$STEP" -lt "$START_FROM" ]; then
        echo "Skipping step $STEP ($(basename "${FOLDERS[$i]}")) — --from $START_FROM"
        continue
    fi

    FOLDER="${FOLDERS[$i]}"
    FOLDER_NAME=$(basename "$FOLDER")
    if [ "$APPROACH" = "b" ]; then
        FOLDER_NAME="shard_${STEP}"
    fi
    CURRENT_FOLDER="$FOLDER_NAME"
    CURRENT_STEP="$STEP"

    # ── SHP dir(s) for this shard ─────────────────────────────────────────────
    if [ "$APPROACH" = "b" ]; then
        SHARD_SHP_DIRS="$SHP_DIRS_B"
    else
        SHARD_SHP_DIR=$(find_shp_dir "$FOLDER")
        if [ -z "$SHARD_SHP_DIR" ]; then
            echo "[WARN] No SHP dir found in $FOLDER"
            SHARD_SHP_DIR="$FOLDER"
        fi
        SHARD_SHP_DIRS="$SHARD_SHP_DIR"
    fi

    echo ""
    echo "========================================"
    echo " Step $STEP/$TOTAL: $FOLDER_NAME"
    echo " Shapefiles   : $SHARD_SHP_DIRS"
    if [ "$APPROACH" = "b" ]; then
        TIFF_COUNT=$(echo "${TIFF_FILES_PER_SHARD[$i]}" | tr ':' '\n' | grep -c .)
        echo " TIFFs        : $TIFF_COUNT (random subset)"
    fi
    echo "========================================"

    # ── Disk accounting ───────────────────────────────────────────────────────
    REPLAY_BYTES=$(dir_bytes "data/replay")
    REPLAY_GB=$(python3 -c "print(f'{${REPLAY_BYTES}/1024**3:.3f}')")
    PROCESSED_BUDGET_GB=$(compute_processed_budget_gb)
    echo "[$(date +%H:%M:%S)] Disk: replay=${REPLAY_GB} GB  budget=${PROCESSED_BUDGET_GB} GB  ceiling=${MAX_DISK_GB} GB"
    python3 -c "
replay_gb=${REPLAY_GB}; ceiling=${MAX_DISK_GB}; budget=${PROCESSED_BUDGET_GB}
print(f'  replay={replay_gb:.3f} GB  budget={budget:.3f} GB  ceiling={ceiling} GB')
assert replay_gb < ceiling, f'[ERROR] Replay exceeds ceiling!'
assert budget >= 1.0, f'[ERROR] Budget < 1 GB — increase MAX_DISK_GB'
"

    # ── Preprocess ────────────────────────────────────────────────────────────
    echo "[$(date +%H:%M:%S)] Preprocessing $FOLDER_NAME..."
    PREPROCESS_ENV=(
        RAW_DATA_DIR="$FOLDER"
        SHP_DIR="$SHARD_SHP_DIRS"
        SHP_DIRS_LIST="$SHARD_SHP_DIRS"
        MAX_PROCESSED_GB="$PROCESSED_BUDGET_GB"
    )
    if [ "$APPROACH" = "b" ] && [ -n "${TIFF_FILES_PER_SHARD[$i]}" ]; then
        PREPROCESS_ENV+=(TIFF_FILES="${TIFF_FILES_PER_SHARD[$i]}")
    fi

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
    d=json.load(open('data/processed/tiles_meta.json'))
    print(len(d['tiles']))
except: print(0)
" 2>/dev/null || echo "0")
    echo "[$(date +%H:%M:%S)] Preprocessed $TILE_COUNT tiles"

    PROC_BYTES=$(dir_bytes "data/processed")
    REPLAY_BYTES=$(dir_bytes "data/replay")
    TOTAL_BYTES=$(( PROC_BYTES + REPLAY_BYTES ))
    python3 -c "
total_gb=${TOTAL_BYTES}/1024**3; ceiling=${MAX_DISK_GB}
proc_gb=${PROC_BYTES}/1024**3; replay_gb=${REPLAY_BYTES}/1024**3
print(f'  Disk after preprocess: processed={proc_gb:.2f} GB  replay={replay_gb:.2f} GB  total={total_gb:.2f} GB / {ceiling} GB')
if total_gb > ceiling*1.02: raise SystemExit(f'[ERROR] Disk ceiling exceeded: {total_gb:.2f} GB > {ceiling} GB')
"

    # ── Save replay ───────────────────────────────────────────────────────────
    echo "[$(date +%H:%M:%S)] Saving replay slice for $FOLDER_NAME..."
    python3 -c "
import sys; sys.path.insert(0,'src')
from replay_buffer import save_replay_from_shard
save_replay_from_shard('$FOLDER_NAME')
" || echo "[WARN] Replay save failed"

    # ── Train ─────────────────────────────────────────────────────────────────
    TRAIN_FLAGS=()

    if [ -n "$INIT_WEIGHTS" ] && [ "$STEP" -eq 1 ]; then
        # First shard with init-weights: load weights, train fresh
        TRAIN_FLAGS+=("--init-weights" "$INIT_WEIGHTS")
        echo "  Mode: INIT-WEIGHTS (epoch=0, full LR, fresh optimizer)"
    elif [ "$STEP" -gt 1 ] || [ "$START_FROM" -gt 1 ] || [ -n "$PRETRAINED_CKPT" ]; then
        TRAIN_FLAGS+=("--resume")
        if [ -n "$PRETRAINED_CKPT" ] && [ "$STEP" -eq 1 ]; then
            echo "  Mode: FINE-TUNE from pretrained checkpoint"
        else
            echo "  Mode: RESUME (fine-tuning from previous shard)"
        fi
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

    # ── Wipe processed tiles ──────────────────────────────────────────────────
    echo "[$(date +%H:%M:%S)] Clearing processed tiles..."
    rm -rf data/processed/images data/processed/masks data/processed/tiles_meta.json

    REPLAY_BYTES_AFTER=$(dir_bytes "data/replay")
    python3 -c "
replay_gb=${REPLAY_BYTES_AFTER}/1024**3; ceiling=${MAX_DISK_GB}
print(f'  Disk after wipe: replay={replay_gb:.2f} GB  (ceiling={ceiling} GB)')
assert replay_gb < ceiling, f'[ERROR] Replay alone exceeds ceiling!'
"
    echo "[$(date +%H:%M:%S)] ✓ Finished $FOLDER_NAME ($STEP/$TOTAL)"
done

echo ""
echo "========================================"
echo " All $TOTAL generalist shards complete!"
echo " Generalist checkpoint: checkpoints/best_model.pt"
echo "========================================"

# ── Specialist pipeline ───────────────────────────────────────────────────────
if [ "$SKIP_SPECIALIST" -eq 0 ]; then

    CURRENT_FOLDER="specialist"
    CURRENT_STEP="$TOTAL"

    echo ""
    echo "========================================"
    echo " Specialist Pipeline"
    echo " Training specialist for Bridge / Railway / Utility"
    echo " Disk ceiling: ${MAX_DISK_GB} GB"
    echo "========================================"

    # ── S1: Build specialist dataset metadata ─────────────────────────────────
    echo "[$(date +%H:%M:%S)] Building specialist dataset metadata..."
    if ! python specialist/01_build_specialist_meta.py 2>&1 | tee /tmp/specialist_meta_log.txt; then
        echo "[ERROR] Specialist metadata build failed"
        python src/07_notify.py --error \
            --folder "specialist/01_build_specialist_meta" --step "$TOTAL" --total "$TOTAL" \
            --error-msg "$(tail -40 /tmp/specialist_meta_log.txt | head -c 2000)" || true
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] Specialist metadata built."

    # ── S2: Train specialist model ────────────────────────────────────────────
    # MAX_DISK_GB is passed explicitly so the specialist respects the same
    # disk budget as the generalist shards (already exported but made explicit
    # here to mirror the PREPROCESS_ENV pattern used above).
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

    # ── Parse metrics (same grep pattern as generalist loop) ─────────────────
    SPEC_TRAIN_LOSS=$(grep "tr_loss="  /tmp/specialist_train_log.txt | tail -1 | grep -oP "tr_loss=\K[0-9.]+"  || echo "0")
    SPEC_VAL_LOSS=$(  grep "val_loss=" /tmp/specialist_train_log.txt | tail -1 | grep -oP "val_loss=\K[0-9.]+" || echo "0")
    SPEC_TRAIN_MIOU=$(grep "tr_mIoU="  /tmp/specialist_train_log.txt | tail -1 | grep -oP "tr_mIoU=\K[0-9.]+"  || echo "0")
    SPEC_VAL_MIOU=$(  grep "val_mIoU=" /tmp/specialist_train_log.txt | tail -1 | grep -oP "val_mIoU=\K[0-9.]+" || echo "0")
    SPEC_EPOCHS=$(grep -c "^Epoch " /tmp/specialist_train_log.txt 2>/dev/null | tr -d '[:space:]' || echo "0")

    # Read per-class IoU for Bridge (4), Railway (5), Utility (6) from checkpoint.
    # These keys may not exist if the specialist was built against a 4-class config;
    # the python snippet handles that gracefully.
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

    # ── S3: Notify specialist completion ──────────────────────────────────────
    # Minor-class per-class IoU is embedded in --folder so it appears in the
    # email subject and body without requiring changes to 07_notify.py.
    SPEC_FOLDER_LABEL="Specialist [${SPEC_PER_CLASS}]"
    python src/07_notify.py \
        --folder "$SPEC_FOLDER_LABEL" --step "$TOTAL" --total "$TOTAL" \
        --train-loss "${SPEC_TRAIN_LOSS:-0}" --val-loss "${SPEC_VAL_LOSS:-0}" \
        --train-miou "${SPEC_TRAIN_MIOU:-0}" --val-miou "${SPEC_VAL_MIOU:-0}" \
        --epochs "${SPEC_EPOCHS:-0}" --checkpoint "specialist/checkpoints/best_model.pt" || true

    echo "[$(date +%H:%M:%S)] ✓ Specialist pipeline complete"
    echo ""
    echo "========================================"
    echo " Pipeline complete (generalist + specialist)"
    echo " Generalist : checkpoints/best_model.pt"
    echo " Specialist : specialist/checkpoints/best_model.pt"
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