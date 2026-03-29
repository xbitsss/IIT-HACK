#!/bin/bash
# 06_train_incremental.sh — Incremental GeoSeg training pipeline
#
# Process B (only approach): TIFFs and SHP dirs live together;
# each TIFF is spatially matched to its correct SHP directory automatically.
#
# Data layout:
#   /raw_data/ALL/
#       image_001.tif
#       image_002.tif
#       SHP1/  *.shp      ← shapefile group 1 (generalist + specialist classes)
#       SHP2/  *.shp      ← shapefile group 2 (or just one SHP dir)
#
# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINTS — start from wherever you need
# ─────────────────────────────────────────────────────────────────────────────
#
#  Full pipeline (shard 1 → shard 2 → specialist):
#    ./src/06_train_incremental.sh --data-dir /raw_data/ALL
#
#  Resume from shard 2 (shard 1 already done):
#    ./src/06_train_incremental.sh --data-dir /raw_data/ALL --from 2
#
#  Specialist only (tiles must exist in processed/ or replay/):
#    ./src/06_train_incremental.sh --specialist-only
#
#  Specialist only, NO tiles yet (bootstraps preprocessing from scratch):
#    ./src/06_train_incremental.sh --specialist-only --data-dir /raw_data/ALL
#
#  Generalist only, skip specialist:
#    ./src/06_train_incremental.sh --data-dir /raw_data/ALL --skip-specialist
#
# ─────────────────────────────────────────────────────────────────────────────
# FLAGS
# ─────────────────────────────────────────────────────────────────────────────
#   --data-dir /path      Root folder (TIFFs + SHP subdirs) (default: /raw_data/ALL)
#   --shp-dirs d1:d2      Colon-separated SHP dirs (default: auto-detect)
#   --from N              Resume from shard N (1 or 2)
#   --pretrained /f.pth   Fine-tune from existing checkpoint
#   --init-weights /f.pth Load weights only — epoch 0, full LR
#   --specialist-only     Skip all generalist shards
#   --skip-specialist     Run generalist shards only
#   --message "text"      Description of this run (stored in checkpoint + emails)
#
# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT OVERRIDES
# ─────────────────────────────────────────────────────────────────────────────
#   MAX_DISK_GB            Hard disk ceiling in GB       (default: 50)
#   REPLAY_TILES_PER_SHARD Tiles kept per shard          (default: 300)

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
RUN_MESSAGE=""

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
        --message|-m)      RUN_MESSAGE="$2";     shift 2 ;;
        --help|-h)
            head -65 "$0" | tail -60
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: $0 [--data-dir /path] [--from N] [--message \"desc\"]"
            echo "          [--specialist-only] [--skip-specialist]"
            echo "          [--pretrained /path] [--init-weights /path]"
            echo "          [--shp-dirs dir1:dir2]"
            echo ""
            echo "See PIPELINE_GUIDE.md for full documentation."
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
ts()  { date +"%H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

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

notify_shell_milestone() {
    # $1=stage  $2=milestone-name  $3=details
    python3 src/07_notify.py --milestone \
        --folder "$1" --milestone-name "$2" --details "$3" \
        --version "${GEN_VERSION:-0}" --run-message "$RUN_MESSAGE" || true
}

# ── Auto-detect SHP dirs from a data directory ────────────────────────────────
detect_shp_dirs() {
    local data_dir="$1"
    local found=""
    for subdir in "$data_dir"/*/; do
        [ -d "$subdir" ] || continue
        if find "$subdir" -maxdepth 1 -iname "*.shp" 2>/dev/null | grep -q .; then
            found="${found:+$found:}${subdir%/}"
        fi
    done
    echo "$found"
}

# ── Version management ────────────────────────────────────────────────────────
GEN_VERSION=0
SPEC_VERSION=0

# =============================================================================
# ENTRY POINT: Specialist only
# =============================================================================
if [ "$SPECIALIST_ONLY" -eq 1 ]; then
    echo ""
    echo "========================================"
    echo " Specialist-Only Run"
    echo " Message: ${RUN_MESSAGE:-(none)}"
    echo "========================================"

    SPEC_VERSION=$(python3 src/run_version.py --bump specialist \
        --message "$RUN_MESSAGE" 2>/dev/null || echo "0")
    log "Specialist version: v${SPEC_VERSION}"

    # ── Check for available tiles ─────────────────────────────────────────────
    HAS_PROCESSED=$(python3 -c "
import json; from pathlib import Path
p = Path('data/processed/tiles_meta.json')
print('1' if p.exists() and json.load(open(p)).get('tiles') else '0')
" 2>/dev/null || echo "0")

    HAS_REPLAY=$(python3 -c "
import sys; sys.path.insert(0,'src')
from replay_buffer import replay_exists
print('1' if replay_exists() else '0')
" 2>/dev/null || echo "0")

    # ── Bootstrap: preprocess if no tiles exist ───────────────────────────────
    if [ "$HAS_PROCESSED" = "0" ] && [ "$HAS_REPLAY" = "0" ]; then
        echo ""
        echo "──────────────────────────────────────────────────────────────"
        echo " [BOOTSTRAP] No tiles found — preprocessing from raw data"
        echo " This finds all minor-class tiles (Bridge/Railway/Utility)"
        echo " and creates the specialist tile set from scratch."
        echo "──────────────────────────────────────────────────────────────"

        if [ ! -d "$DATA_DIR_B" ]; then
            echo ""
            echo "[ERROR] No tiles found AND --data-dir '$DATA_DIR_B' does not exist."
            echo ""
            echo "  You need to provide the raw data folder so we can preprocess it."
            echo "  Options:"
            echo "    A) Run with --data-dir pointing to your TIFFs:"
            echo "         ./src/06_train_incremental.sh --specialist-only --data-dir /raw_data/ALL"
            echo ""
            echo "    B) Run generalist shards first (creates replay tiles as a side-effect):"
            echo "         ./src/06_train_incremental.sh --data-dir /raw_data/ALL"
            echo ""
            echo "    C) Copy an existing data/replay/ from another machine."
            echo ""
            python3 src/07_notify.py --error \
                --folder "specialist-bootstrap" --step 0 --total 2 \
                --error-msg "No tiles and no --data-dir provided for specialist bootstrap." \
                --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true
            exit 1
        fi

        # Auto-detect SHP dirs
        if [ -z "$SHP_DIRS_B" ]; then
            SHP_DIRS_B=$(detect_shp_dirs "$DATA_DIR_B")
        fi
        if [ -z "$SHP_DIRS_B" ]; then
            echo "[ERROR] No SHP subdirectories found in $DATA_DIR_B"
            echo "  Check that your SHP files are in subdirectories with .shp extension."
            echo "  Or specify: --shp-dirs /path/to/SHP1:/path/to/SHP2"
            exit 1
        fi

        log "Bootstrap preprocessing with specialist config (Bridge/Railway/Utility included)..."
        log "  Data dir : $DATA_DIR_B"
        log "  SHP dirs : $SHP_DIRS_B"

        PROCESSED_BUDGET_GB=$(compute_processed_budget_gb)

        # Run preprocessing with SPECIALIST_PREPROCESS=1 so config_specialist's
        # SHAPEFILE_MAP (which includes Bridge/Railway/Utility) is used
        if ! env \
            RAW_DATA_DIR="$DATA_DIR_B" \
            SHP_DIR="$(echo "$SHP_DIRS_B" | cut -d: -f1)" \
            SHP_DIRS_LIST="$SHP_DIRS_B" \
            MAX_PROCESSED_GB="$PROCESSED_BUDGET_GB" \
            SPECIALIST_PREPROCESS="1" \
            python3 src/01_preprocess.py 2>&1 | tee /tmp/specialist_bootstrap_preprocess_log.txt; then

            log "[ERROR] Bootstrap preprocessing failed"
            python3 src/07_notify.py --error \
                --folder "specialist-bootstrap" --step 0 --total 2 \
                --error-msg "$(tail -40 /tmp/specialist_bootstrap_preprocess_log.txt | head -c 2000)" \
                --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true
            echo ""
            echo "  To retry: ./src/06_train_incremental.sh --specialist-only --data-dir $DATA_DIR_B"
            echo "  Check: config_specialist.py → SHAPEFILE_MAP has Bridge/Railway/Utility entries"
            exit 1
        fi

        TILE_COUNT=$(python3 -c "
import json
try:
    d = json.load(open('data/processed/tiles_meta.json'))
    tiles = d['tiles']
    minor = sum(1 for t in tiles if {4,5,6} & set(t.get('class_ids',[])))
    print(f'{len(tiles)} total  {minor} minor-class')
except: print('0')
" 2>/dev/null || echo "0")
        log "Bootstrap preprocessing complete: ${TILE_COUNT} tiles"
        notify_shell_milestone "specialist-bootstrap" "preprocess_done" \
            "Bootstrap preprocessing complete: ${TILE_COUNT} tiles."

        HAS_PROCESSED="1"
    fi

    log "Tile source: processed=${HAS_PROCESSED} replay=${HAS_REPLAY}"

    # ── S1: Build specialist metadata (50% minor / 50% major split) ───────────
    log "Building specialist dataset metadata (50/50 minor/major split)..."
    if ! python3 specialist/01_build_specialist_meta.py \
            2>&1 | tee /tmp/specialist_meta_log.txt; then
        log "[ERROR] Specialist metadata build failed"
        python3 src/07_notify.py --error \
            --folder "specialist-meta" --step 0 --total 2 \
            --error-msg "$(tail -40 /tmp/specialist_meta_log.txt | head -c 2000)" \
            --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true
        echo ""
        echo "  Fix: Check that Bridge/Railway/Utility tiles exist in data/processed/ or data/replay/"
        echo "  Then retry: ./src/06_train_incremental.sh --specialist-only"
        exit 1
    fi

    META_SUMMARY=$(python3 -c "
import json
try:
    d = json.load(open('data/processed/tiles_meta_specialist.json'))
    print(f\"{d.get('total_tiles','?')} tiles  minor={d.get('n_minor_tiles','?')}  major={d.get('n_major_tiles','?')}\")
except: print('?')
" 2>/dev/null || echo "?")
    log "Specialist metadata built: ${META_SUMMARY}"
    notify_shell_milestone "specialist" "preprocess_done" \
        "Specialist meta built (${META_SUMMARY})."

    # ── S2: Train specialist ───────────────────────────────────────────────────
    log "Training specialist model (v${SPEC_VERSION})..."
    if ! python3 specialist/03_train_specialist.py \
            --run-version "$SPEC_VERSION" \
            --run-message "$RUN_MESSAGE" \
            2>&1 | tee /tmp/specialist_train_log.txt; then
        log "[ERROR] Specialist training failed"
        python3 src/07_notify.py --error \
            --folder "specialist" --step 2 --total 2 \
            --error-msg "$(tail -40 /tmp/specialist_train_log.txt | head -c 2000)" \
            --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true
        echo ""
        echo "  To resume specialist training:"
        echo "    ./src/06_train_incremental.sh --specialist-only --message 'resume'"
        echo "    python specialist/03_train_specialist.py --resume"
        exit 1
    fi

    SPEC_VAL_MIOU=$(  grep "val_mIoU=" /tmp/specialist_train_log.txt | tail -1 | grep -oP "val_mIoU=\K[0-9.]+"  || echo "0")
    SPEC_TRAIN_LOSS=$(grep "tr_loss="  /tmp/specialist_train_log.txt | tail -1 | grep -oP "tr_loss=\K[0-9.]+"   || echo "0")
    SPEC_VAL_LOSS=$(  grep "val_loss=" /tmp/specialist_train_log.txt | tail -1 | grep -oP "val_loss=\K[0-9.]+"  || echo "0")
    SPEC_TRAIN_MIOU=$(grep "tr_mIoU="  /tmp/specialist_train_log.txt | tail -1 | grep -oP "tr_mIoU=\K[0-9.]+"   || echo "0")
    SPEC_EPOCHS=$(grep -c "^Epoch " /tmp/specialist_train_log.txt 2>/dev/null | tr -d '[:space:]' || echo "0")

    SPEC_PER_CLASS=$(python3 -c "
import torch
try:
    ckpt = torch.load('specialist/checkpoints/best_model.pt', map_location='cpu', weights_only=False)
    pc   = ckpt.get('per_class_iou', {})
    def g(k): return float(pc.get(k, pc.get(str(k), 0.0)))
    print(f'Bridge={g(4):.4f}  Railway={g(5):.4f}  Utility={g(6):.4f}')
except Exception as e:
    print(f'(per-class: {e})')
" 2>/dev/null || echo "(unavailable)")

    log "Specialist done — val_mIoU=${SPEC_VAL_MIOU}  ${SPEC_PER_CLASS}"

    python3 src/07_notify.py \
        --folder "Specialist v${SPEC_VERSION} [${SPEC_PER_CLASS}]" \
        --step 2 --total 2 \
        --train-loss "${SPEC_TRAIN_LOSS:-0}" --val-loss "${SPEC_VAL_LOSS:-0}" \
        --train-miou "${SPEC_TRAIN_MIOU:-0}" --val-miou "${SPEC_VAL_MIOU:-0}" \
        --epochs "${SPEC_EPOCHS:-0}" \
        --checkpoint "specialist/checkpoints/best_model.pt" \
        --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true

    echo ""
    echo "========================================"
    echo " Specialist-only run complete"
    echo " Specialist v${SPEC_VERSION}: specialist/checkpoints/best_model.pt"
    echo " Versioned  : specialist/checkpoints/specialist_v${SPEC_VERSION}_best.pt"
    echo ""
    echo " Combined inference:"
    echo "   python specialist/04_inference_combined.py \\"
    echo "       --input /path/to/image.tif --output outputs/mask.tif"
    echo "========================================"
    exit 0
fi

# =============================================================================
# ENTRY POINTS 1 & 2: Generalist shards (+ optional specialist)
# =============================================================================

# ── Discover TIFFs ────────────────────────────────────────────────────────────
if [ ! -d "$DATA_DIR_B" ]; then
    echo "[ERROR] --data-dir '$DATA_DIR_B' does not exist."
    echo "  Set the correct path:  --data-dir /path/to/your/data"
    exit 1
fi

mapfile -t ALL_TIFS < <(
    find "$DATA_DIR_B" -maxdepth 1 \( -iname "*.tif" -o -iname "*.tiff" \) | \
    python3 -c "
import sys, random, os
lines = sys.stdin.read().splitlines()
random.seed(int(os.environ.get('RANDOM_SEED', '42')))
random.shuffle(lines)
print('\n'.join(lines))
"
)
N_TIFS=${#ALL_TIFS[@]}
if [ "$N_TIFS" -eq 0 ]; then
    echo "[ERROR] No TIFF files found in $DATA_DIR_B"
    echo "  Check: ls -la $DATA_DIR_B"
    exit 1
fi

HALF=$(( N_TIFS / 2 ))
SHARD1_FILES=$(IFS=:; echo "${ALL_TIFS[*]:0:$HALF}")
SHARD2_FILES=$(IFS=:; echo "${ALL_TIFS[*]:$HALF}")

# ── Auto-detect SHP dirs ──────────────────────────────────────────────────────
if [ -z "$SHP_DIRS_B" ]; then
    SHP_DIRS_B=$(detect_shp_dirs "$DATA_DIR_B")
fi

if [ -z "$SHP_DIRS_B" ]; then
    echo "[ERROR] No SHP subdirectories found in $DATA_DIR_B"
    echo "  Expected: $DATA_DIR_B/SHP1/*.shp"
    echo "  Or specify: --shp-dirs dir1:dir2"
    exit 1
fi

TOTAL=2

# ── Validate --from ───────────────────────────────────────────────────────────
if [ "$START_FROM" -lt 1 ] || [ "$START_FROM" -gt "$TOTAL" ]; then
    echo "[ERROR] --from must be 1 or 2 (got $START_FROM)"
    exit 1
fi

# ── Bump generalist version ────────────────────────────────────────────────────
if [ "$START_FROM" -eq 1 ]; then
    GEN_VERSION=$(python3 src/run_version.py --bump generalist \
        --message "$RUN_MESSAGE" 2>/dev/null || echo "0")
else
    GEN_VERSION=$(python3 src/run_version.py --current generalist 2>/dev/null || echo "0")
fi
log "Generalist version: v${GEN_VERSION}  message: '${RUN_MESSAGE}'"

echo ""
echo "========================================"
echo " Incremental Training Pipeline"
echo " Generalist v${GEN_VERSION}"
echo " Message      : ${RUN_MESSAGE:-(none)}"
echo " Data dir     : $DATA_DIR_B"
echo " Total TIFFs  : $N_TIFS  (shard1=${HALF}  shard2=$(( N_TIFS - HALF )))"
echo " SHP dirs     : $SHP_DIRS_B"
echo " Starting from: shard $START_FROM"
echo " Disk ceiling : ${MAX_DISK_GB} GB"
echo " Replay/shard : ${REPLAY_TILES_PER_SHARD} tiles"
echo " Specialist   : $([ "$SKIP_SPECIALIST" -eq 1 ] && echo "SKIP" || echo "YES")"
echo "========================================"

# ── Fresh start wipe ──────────────────────────────────────────────────────────
if [ "$START_FROM" -eq 1 ]; then
    log "Fresh start — wiping stale data..."
    [ -d "data/replay" ]    && find data/replay -mindepth 1 -delete && log "  Cleared data/replay/"
    [ -d "data/processed" ] && find data/processed -mindepth 1 -delete && log "  Cleared data/processed/"

    if [ -n "$INIT_WEIGHTS" ]; then
        [ ! -f "$INIT_WEIGHTS" ] && echo "[ERROR] --init-weights not found: $INIT_WEIGHTS" && exit 1
        rm -f checkpoints/best_model.pt checkpoints/training_curves.png checkpoints/history.json
        log "  Init weights: $INIT_WEIGHTS"
    elif [ -n "$PRETRAINED_CKPT" ]; then
        [ ! -f "$PRETRAINED_CKPT" ] && echo "[ERROR] --pretrained not found: $PRETRAINED_CKPT" && exit 1
        mkdir -p checkpoints
        cp "$PRETRAINED_CKPT" checkpoints/best_model.pt
        log "  Installed pretrained checkpoint → checkpoints/best_model.pt"
    else
        rm -f checkpoints/best_model.pt checkpoints/training_curves.png checkpoints/history.json
    fi

    python3 -c "
import json, datetime
json.dump({
    'run_version': ${GEN_VERSION}, 'message': '${RUN_MESSAGE}',
    'started': datetime.datetime.now().isoformat(),
    'start_from': ${START_FROM}, 'last_completed_shard': 0,
    'specialist_done': False,
}, open('pipeline_state.json', 'w'), indent=2)
" 2>/dev/null || true
fi

# ── Crash / ERR trap ──────────────────────────────────────────────────────────
CURRENT_FOLDER="unknown"
CURRENT_STEP=0

_shell_crash() {
    local EXIT_CODE=${1:-$?}
    log "[CRASH] Pipeline error at $CURRENT_FOLDER (exit $EXIT_CODE)"
    log "  Resume: ./src/06_train_incremental.sh --from $CURRENT_STEP --data-dir $DATA_DIR_B"
    log "  Or specialist only: ./src/06_train_incremental.sh --specialist-only"
    python3 src/07_notify.py --error \
        --folder "$CURRENT_FOLDER" --step "$CURRENT_STEP" --total "$TOTAL" \
        --error-msg "Pipeline crashed (exit $EXIT_CODE) at $CURRENT_FOLDER.
Resume: ./src/06_train_incremental.sh --from $CURRENT_STEP --data-dir $DATA_DIR_B
Or specialist only: ./src/06_train_incremental.sh --specialist-only" \
        --version "$GEN_VERSION" --run-message "$RUN_MESSAGE" || true
}
trap '_shell_crash $?' ERR

# =============================================================================
# MAIN LOOP — two generalist shards
# =============================================================================
TIFF_FILES_PER_SHARD=("$SHARD1_FILES" "$SHARD2_FILES")

for i in 0 1; do
    STEP=$(( i + 1 ))

    if [ "$STEP" -lt "$START_FROM" ]; then
        log "Skipping shard $STEP (--from $START_FROM)"
        continue
    fi

    FOLDER_NAME="shard_${STEP}"
    CURRENT_FOLDER="$FOLDER_NAME"
    CURRENT_STEP="$STEP"

    IS_LAST_FOR_SPECIALIST=0
    if [ "$STEP" -eq "$TOTAL" ] && [ "$SKIP_SPECIALIST" -eq 0 ]; then
        IS_LAST_FOR_SPECIALIST=1
    fi

    TIFF_COUNT=$(echo "${TIFF_FILES_PER_SHARD[$i]}" | tr ':' '\n' | grep -c . || echo "?")

    echo ""
    echo "========================================"
    echo " Shard $STEP/$TOTAL: $FOLDER_NAME  (${TIFF_COUNT} TIFFs)"
    echo " Generalist v${GEN_VERSION}"
    [ -n "$RUN_MESSAGE" ] && echo " Message: $RUN_MESSAGE"
    [ "$IS_LAST_FOR_SPECIALIST" -eq 1 ] && \
        echo " [NOTE] Tiles kept alive after this shard for specialist"
    echo "========================================"

    # ── Disk accounting ───────────────────────────────────────────────────────
    REPLAY_BYTES=$(dir_bytes "data/replay")
    REPLAY_GB=$(python3 -c "print(f'{${REPLAY_BYTES}/1024**3:.3f}')")
    PROCESSED_BUDGET_GB=$(compute_processed_budget_gb)
    log "Disk: replay=${REPLAY_GB} GB  budget=${PROCESSED_BUDGET_GB} GB  ceiling=${MAX_DISK_GB} GB"
    python3 -c "
replay_gb=${REPLAY_GB}; ceiling=${MAX_DISK_GB}; budget=${PROCESSED_BUDGET_GB}
assert replay_gb < ceiling, f'[ERROR] Replay exceeds ceiling! {replay_gb:.2f} >= {ceiling}'
assert budget >= 1.0,       f'[ERROR] Budget < 1 GB — increase MAX_DISK_GB'
"

    # ── Preprocess ────────────────────────────────────────────────────────────
    log "Preprocessing $FOLDER_NAME..."
    if ! env \
        RAW_DATA_DIR="$DATA_DIR_B" \
        SHP_DIR="$(echo "$SHP_DIRS_B" | cut -d: -f1)" \
        SHP_DIRS_LIST="$SHP_DIRS_B" \
        MAX_PROCESSED_GB="$PROCESSED_BUDGET_GB" \
        TIFF_FILES="${TIFF_FILES_PER_SHARD[$i]}" \
        python3 src/01_preprocess.py 2>&1 | tee /tmp/preprocess_log.txt; then

        log "[ERROR] Preprocessing failed for $FOLDER_NAME"
        python3 src/07_notify.py --error \
            --folder "$FOLDER_NAME" --step "$STEP" --total "$TOTAL" \
            --error-msg "$(tail -40 /tmp/preprocess_log.txt | head -c 2000)" \
            --version "$GEN_VERSION" --run-message "$RUN_MESSAGE" || true
        echo "  To retry this shard: ../src/06_train_incremental.sh --from $STEP --data-dir $DATA_DIR_B"
        exit 1
    fi

    TILE_COUNT=$(python3 -c "
import json
try:
    d = json.load(open('data/processed/tiles_meta.json'))
    print(len(d['tiles']))
except: print(0)
" 2>/dev/null || echo "0")
    log "Preprocessed ${TILE_COUNT} tiles"
    notify_shell_milestone "$FOLDER_NAME" "preprocess_done" \
        "Preprocessed ${TILE_COUNT} tiles. Budget: ${PROCESSED_BUDGET_GB} GB."

    PROC_BYTES=$(dir_bytes "data/processed")
    TOTAL_BYTES=$(( PROC_BYTES + REPLAY_BYTES ))
    python3 -c "
total_gb=${TOTAL_BYTES}/1024**3; ceiling=${MAX_DISK_GB}
proc_gb=${PROC_BYTES}/1024**3; replay_gb=${REPLAY_BYTES}/1024**3
print(f'  Disk: processed={proc_gb:.2f} GB  replay={replay_gb:.2f} GB  total={total_gb:.2f}/{ceiling} GB')
if total_gb > ceiling*1.02:
    raise SystemExit(f'[ERROR] Disk ceiling exceeded: {total_gb:.2f} GB > {ceiling} GB')
"

    # ── Save replay ───────────────────────────────────────────────────────────
    log "Saving replay slice for $FOLDER_NAME..."
    if python3 -c "
import sys; sys.path.insert(0,'src')
from replay_buffer import save_replay_from_shard
save_replay_from_shard('$FOLDER_NAME')
" 2>&1 | tee /tmp/replay_log.txt; then
        REPLAY_TILES=$(grep -oP "Saved \K[0-9]+" /tmp/replay_log.txt | tail -1 || echo "?")
        log "Replay saved: ~${REPLAY_TILES} tiles"
        notify_shell_milestone "$FOLDER_NAME" "replay_saved" \
            "Replay buffer updated: ~${REPLAY_TILES} tiles from ${FOLDER_NAME}."
    else
        log "[WARN] Replay save failed — continuing without replay for this shard"
    fi

    # ── Train ─────────────────────────────────────────────────────────────────
    TRAIN_FLAGS=()
    if [ -n "$INIT_WEIGHTS" ] && [ "$STEP" -eq 1 ]; then
        TRAIN_FLAGS+=("--init-weights" "$INIT_WEIGHTS")
        log "  Mode: INIT-WEIGHTS"
    elif [ "$STEP" -gt 1 ] || [ "$START_FROM" -gt 1 ] || [ -n "$PRETRAINED_CKPT" ]; then
        TRAIN_FLAGS+=("--resume")
        log "  Mode: RESUME"
    else
        log "  Mode: FRESH"
    fi

    log "Training $FOLDER_NAME (generalist v${GEN_VERSION})..."
    if ! python3 src/03_train.py \
            "${TRAIN_FLAGS[@]}" \
            --folder-name  "$FOLDER_NAME" \
            --folder-step  "$STEP" \
            --folder-total "$TOTAL" \
            --run-version  "$GEN_VERSION" \
            --run-message  "$RUN_MESSAGE" \
            2>&1 | tee /tmp/train_log.txt; then

        TRAIN_EXIT=${PIPESTATUS[0]}
        log "[ERROR] Training failed (exit $TRAIN_EXIT)"
        python3 src/07_notify.py --error \
            --folder "$FOLDER_NAME" --step "$STEP" --total "$TOTAL" \
            --error-msg "$(tail -40 /tmp/train_log.txt | head -c 2000)" \
            --version "$GEN_VERSION" --run-message "$RUN_MESSAGE" || true
        echo "  To resume: ./src/06_train_incremental.sh --from $STEP --data-dir $DATA_DIR_B"
        exit 1
    fi

    TRAIN_LOSS=$(grep "tr_loss="  /tmp/train_log.txt | tail -1 | grep -oP "tr_loss=\K[0-9.]+"  || echo "0")
    VAL_LOSS=$(  grep "val_loss=" /tmp/train_log.txt | tail -1 | grep -oP "val_loss=\K[0-9.]+" || echo "0")
    TRAIN_MIOU=$(grep "tr_mIoU="  /tmp/train_log.txt | tail -1 | grep -oP "tr_mIoU=\K[0-9.]+"  || echo "0")
    VAL_MIOU=$(  grep "val_mIoU=" /tmp/train_log.txt | tail -1 | grep -oP "val_mIoU=\K[0-9.]+" || echo "0")
    EPOCHS=$(grep -c "^Epoch " /tmp/train_log.txt 2>/dev/null | tr -d '[:space:]' || echo "0")
    log "Done — val_mIoU=${VAL_MIOU}  epochs=${EPOCHS}"

    python3 src/07_notify.py \
        --folder "Generalist v${GEN_VERSION} — ${FOLDER_NAME}" \
        --step "$STEP" --total "$TOTAL" \
        --train-loss "${TRAIN_LOSS:-0}" --val-loss "${VAL_LOSS:-0}" \
        --train-miou "${TRAIN_MIOU:-0}" --val-miou "${VAL_MIOU:-0}" \
        --epochs "${EPOCHS:-0}" \
        --checkpoint "checkpoints/best_model.pt" \
        --version "$GEN_VERSION" --run-message "$RUN_MESSAGE" || true

    python3 -c "
import json, datetime
try: state = json.load(open('pipeline_state.json'))
except: state = {}
state['last_completed_shard'] = ${STEP}
state['updated'] = datetime.datetime.now().isoformat()
json.dump(state, open('pipeline_state.json', 'w'), indent=2)
" 2>/dev/null || true

    # ── Wipe processed tiles (unless specialist needs them) ───────────────────
    if [ "$IS_LAST_FOR_SPECIALIST" -eq 0 ]; then
        log "Clearing processed tiles (intermediate shard)..."
        rm -rf data/processed/images data/processed/masks data/processed/tiles_meta.json
        REPLAY_BYTES_AFTER=$(dir_bytes "data/replay")
        python3 -c "
replay_gb=${REPLAY_BYTES_AFTER}/1024**3; ceiling=${MAX_DISK_GB}
print(f'  Disk after wipe: replay={replay_gb:.2f} GB  (ceiling={ceiling} GB)')
"
    else
        log "Keeping processed tiles alive for specialist pipeline..."
    fi

    log "✓ Finished $FOLDER_NAME ($STEP/$TOTAL)"
done

echo ""
echo "========================================"
echo " All $TOTAL generalist shards complete!"
echo " Generalist v${GEN_VERSION}: checkpoints/best_model.pt"
echo " Versioned  : checkpoints/generalist_v${GEN_VERSION}_best.pt"
echo "========================================"

# =============================================================================
# SPECIALIST PIPELINE
# =============================================================================
if [ "$SKIP_SPECIALIST" -eq 0 ]; then

    CURRENT_FOLDER="specialist"
    CURRENT_STEP="$TOTAL"

    SPEC_VERSION=$(python3 src/run_version.py --bump specialist \
        --message "$RUN_MESSAGE" 2>/dev/null || echo "0")
    log "Specialist version: v${SPEC_VERSION}"

    echo ""
    echo "========================================"
    echo " Specialist Pipeline (v${SPEC_VERSION})"
    echo " Training specialist for Bridge / Railway / Utility"
    echo "========================================"

    log "Building specialist dataset metadata (50/50 minor/major split)..."
    if ! python3 specialist/01_build_specialist_meta.py \
            2>&1 | tee /tmp/specialist_meta_log.txt; then
        log "[ERROR] Specialist metadata build failed"
        python3 src/07_notify.py --error \
            --folder "specialist-meta" --step "$TOTAL" --total "$TOTAL" \
            --error-msg "$(tail -40 /tmp/specialist_meta_log.txt | head -c 2000)" \
            --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true
        echo "  To retry specialist: ./src/06_train_incremental.sh --specialist-only"
        exit 1
    fi
    notify_shell_milestone "specialist" "preprocess_done" "Specialist meta built."

    log "Training specialist model (v${SPEC_VERSION})..."
    if ! env MAX_DISK_GB="$MAX_DISK_GB" \
            python3 specialist/03_train_specialist.py \
            --run-version "$SPEC_VERSION" \
            --run-message "$RUN_MESSAGE" \
            2>&1 | tee /tmp/specialist_train_log.txt; then
        SPEC_TRAIN_EXIT=${PIPESTATUS[0]}
        log "[ERROR] Specialist training failed (exit $SPEC_TRAIN_EXIT)"
        python3 src/07_notify.py --error \
            --folder "specialist" --step "$TOTAL" --total "$TOTAL" \
            --error-msg "$(tail -40 /tmp/specialist_train_log.txt | head -c 2000)" \
            --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true
        echo "  To retry: ./src/06_train_incremental.sh --specialist-only"
        exit 1
    fi

    SPEC_TRAIN_LOSS=$(grep "tr_loss="  /tmp/specialist_train_log.txt | tail -1 | grep -oP "tr_loss=\K[0-9.]+"  || echo "0")
    SPEC_VAL_LOSS=$(  grep "val_loss=" /tmp/specialist_train_log.txt | tail -1 | grep -oP "val_loss=\K[0-9.]+" || echo "0")
    SPEC_TRAIN_MIOU=$(grep "tr_mIoU="  /tmp/specialist_train_log.txt | tail -1 | grep -oP "tr_mIoU=\K[0-9.]+"  || echo "0")
    SPEC_VAL_MIOU=$(  grep "val_mIoU=" /tmp/specialist_train_log.txt | tail -1 | grep -oP "val_mIoU=\K[0-9.]+" || echo "0")
    SPEC_EPOCHS=$(grep -c "^Epoch " /tmp/specialist_train_log.txt 2>/dev/null | tr -d '[:space:]' || echo "0")

    SPEC_PER_CLASS=$(python3 -c "
import torch
try:
    ckpt = torch.load('specialist/checkpoints/best_model.pt', map_location='cpu', weights_only=False)
    pc   = ckpt.get('per_class_iou', {})
    def g(k): return float(pc.get(k, pc.get(str(k), 0.0)))
    print(f'Bridge={g(4):.4f}  Railway={g(5):.4f}  Utility={g(6):.4f}')
except Exception as e:
    print(f'(per-class: {e})')
" 2>/dev/null || echo "(unavailable)")

    log "Specialist done — val_mIoU=${SPEC_VAL_MIOU}  ${SPEC_PER_CLASS}"

    # Post-specialist wipe
    if [ -d "data/processed/images" ] || [ -d "data/processed/masks" ] || \
       [ -f "data/processed/tiles_meta.json" ]; then
        log "Clearing last shard's processed tiles (post-specialist)..."
        rm -rf data/processed/images data/processed/masks data/processed/tiles_meta.json
    fi

    python3 -c "
import json, datetime
try: state = json.load(open('pipeline_state.json'))
except: state = {}
state['specialist_done'] = True
state['specialist_version'] = ${SPEC_VERSION}
state['finished'] = datetime.datetime.now().isoformat()
json.dump(state, open('pipeline_state.json', 'w'), indent=2)
" 2>/dev/null || true

    python3 src/07_notify.py \
        --folder "Specialist v${SPEC_VERSION} [${SPEC_PER_CLASS}]" \
        --step "$TOTAL" --total "$TOTAL" \
        --train-loss "${SPEC_TRAIN_LOSS:-0}" --val-loss "${SPEC_VAL_LOSS:-0}" \
        --train-miou "${SPEC_TRAIN_MIOU:-0}" --val-miou "${SPEC_VAL_MIOU:-0}" \
        --epochs "${SPEC_EPOCHS:-0}" \
        --checkpoint "specialist/checkpoints/best_model.pt" \
        --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true

    echo ""
    echo "========================================"
    echo " Pipeline complete (generalist + specialist)"
    echo " Generalist v${GEN_VERSION} : checkpoints/best_model.pt"
    echo " Versioned              : checkpoints/generalist_v${GEN_VERSION}_best.pt"
    echo " Specialist v${SPEC_VERSION}  : specialist/checkpoints/best_model.pt"
    echo " Versioned              : specialist/checkpoints/specialist_v${SPEC_VERSION}_best.pt"
    echo ""
    echo " Combined inference:"
    echo "   python specialist/04_inference_combined.py \\"
    echo "       --input /raw_data/image.tif --output outputs/combined_mask.tif"
    echo ""
    echo " Version history:  python src/run_version.py --list"
    echo "========================================"

else
    echo ""
    echo "[SKIP] --skip-specialist — specialist stage bypassed"
    echo ""
    echo "========================================"
    echo " Pipeline complete (generalist only)"
    echo " Generalist v${GEN_VERSION}: checkpoints/best_model.pt"
    echo " Versioned         : checkpoints/generalist_v${GEN_VERSION}_best.pt"
    echo "========================================"
fi