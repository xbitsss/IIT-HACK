#!/bin/bash
# src/06_train_incremental.sh -- Incremental GeoSeg training pipeline
#
# Memory management
# -----------------
# Exit 137 = Linux OOM killer. Root cause: 16 preprocess workers x ~2 GB each
# ~ 32 GB, plus OS page cache from the previous shard, can exceed the container
# memory limit. This script guards against that with three mechanisms:
#
#   1. check_ram_before_phase() -- refuses to start a phase if < MIN_RAM_GB free.
#   2. Dynamic MAX_PREPROCESS_WORKERS -- workers = floor((available_gb - 8) / 2).
#      Passed as env var to 01_preprocess.py so workers scale with real RAM.
#   3. free_memory_between_phases() -- sync + drop_caches (if root) + gc.collect()
#      + explicit CUDA cache clear. Called after every major phase.
#
# This script lives in src/ and is always called from the project root (/app):
#   bash src/06_train_incremental.sh [flags]
#
# -----------------------------------------------------------------------------
# ENTRY POINTS
# -----------------------------------------------------------------------------
#
#  Full pipeline:
#    docker compose run --rm train-all --data-dir /raw_data/ALL
#
#  Resume from shard 2:
#    docker compose run --rm train-all --data-dir /raw_data/ALL --from 2
#
#  Specialist only (tiles in processed/ or replay/ must exist):
#    docker compose run --rm specialist
#
#  Specialist only, NO tiles yet (bootstrap from raw TIFFs):
#    docker compose run --rm specialist --data-dir /raw_data/ALL
#
#  Generalist only:
#    docker compose run --rm train-all --data-dir /raw_data/ALL --skip-specialist
#
# -----------------------------------------------------------------------------
# FLAGS
# -----------------------------------------------------------------------------
#   --data-dir /path      Root folder (TIFFs + SHP subdirs) [default: /raw_data/ALL]
#   --shp-dirs d1:d2      Colon-separated SHP dirs (default: auto-detect)
#   --from N              Resume from shard N (1 or 2)
#   --pretrained /f.pth   Fine-tune from existing checkpoint
#   --init-weights /f.pth Load weights only -- epoch 0, full LR
#   --specialist-only     Skip all generalist shards
#   --skip-specialist     Run generalist shards only
#   --message "text"      Description of this run
#
# -----------------------------------------------------------------------------
# ENVIRONMENT OVERRIDES
# -----------------------------------------------------------------------------
#   MAX_DISK_GB              Hard disk ceiling GB       (default: 50)
#   REPLAY_TILES_PER_SHARD   Tiles kept per shard       (default: 300)
#   MIN_RAM_GB               Minimum RAM before a phase (default: 12)

set -euo pipefail
SCRIPT_SELF="${BASH_SOURCE[0]}"

export MAX_DISK_GB="${MAX_DISK_GB:-50}"
export REPLAY_TILES_PER_SHARD="${REPLAY_TILES_PER_SHARD:-300}"
MIN_RAM_GB="${MIN_RAM_GB:-12}"

DATA_DIR_B="/raw_data/ALL"
SHP_DIRS_B=""
PRETRAINED_CKPT=""
INIT_WEIGHTS=""
START_FROM=1
SKIP_SPECIALIST=0
SPECIALIST_ONLY=0
RUN_MESSAGE=""

# -- Parse arguments -----------------------------------------------------------
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
            head -70 "$0" | tail -65
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: $SCRIPT_SELF [--data-dir /path] [--from N] [--message \"desc\"]"
            echo "  See PIPELINE_GUIDE.md for full documentation."
            exit 1
            ;;
    esac
done

if [ "$SPECIALIST_ONLY" -eq 1 ] && [ "$SKIP_SPECIALIST" -eq 1 ]; then
    echo "[ERROR] --specialist-only and --skip-specialist are mutually exclusive."
    exit 1
fi

# -- Helpers -------------------------------------------------------------------
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
        echo "[WARN] Replay buffer large -- processed budget clamped to 1 GB" >&2
    fi
    python3 -c "print(f'{${budget_bytes} / 1024**3:.3f}')"
}

# -- Memory management helpers -------------------------------------------------

# Returns available RAM in GB (integer-truncated for shell arithmetic)
available_ram_gb() {
    awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo "99"
}

# Computes safe worker count: floor((available_gb - 8) / 2), clamped to [2,16]
safe_preprocess_workers() {
    local avail_gb="$1"
    python3 -c "
avail = float('$avail_gb')
# Reserve 8 GB for OS + main process + safety buffer
usable = max(0.0, avail - 8.0)
# Each worker needs ~2 GB (TIFF chunk + shapefile GDFs + overhead)
workers = max(2, min(16, int(usable / 2.0)))
print(workers)
" 2>/dev/null || echo "4"
}

# Checks available RAM and exits with an actionable error if below threshold
check_ram_before_phase() {
    local phase_name="$1"
    local required_gb="${2:-$MIN_RAM_GB}"
    local avail
    avail=$(available_ram_gb)
    log "RAM check before ${phase_name}: ${avail} GB available (need ${required_gb} GB)"
    python3 -c "
avail = float('$avail')
needed = float('$required_gb')
if avail < needed:
    print('[ERROR] Not enough RAM to safely start $phase_name.')
    print(f'  Available: {avail:.1f} GB  Required: {needed:.1f} GB')
    print('  Options:')
    print('    - Reduce MAX_PREPROCESS_WORKERS (currently auto-calculated)')
    print('    - Increase container memory limit in docker-compose.yml (shm_size + mem_limit)')
    print('    - Reduce TILE_SIZE in config.py (default 512 -- try 384)')
    import sys; sys.exit(1)
else:
    print(f'  OK: {avail:.1f} GB >= {needed:.1f} GB')
"
}

# Frees OS page cache + GPU memory between pipeline phases
free_memory_between_phases() {
    log "Freeing memory between phases..."
    sync
    # Drop OS page cache, dentries, inodes (root only -- safe to skip if not root)
    if [ "$(id -u)" = "0" ]; then
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null && log "  OS page cache dropped" || true
    else
        log "  (Not root -- skipping drop_caches; sync done)"
    fi
    # Python gc.collect + CUDA cache clear
    python3 -c "
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    alloc  = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved()  / 1024**3
    print(f'  GPU: allocated={alloc:.2f} GB  reserved={reserv:.2f} GB')
else:
    print('  No CUDA device -- GPU cleanup skipped')
" 2>/dev/null || true
    local avail
    avail=$(available_ram_gb)
    log "  RAM after cleanup: ${avail} GB available"
}

notify_shell_milestone() {
    python3 src/07_notify.py --milestone \
        --folder "$1" \
        --milestone-name "$2" \
        --details "$3" \
        --version "${GEN_VERSION:-0}" \
        --run-message "$RUN_MESSAGE" || true
}

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

# -- Version tracking ----------------------------------------------------------
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

    # -- Helper: run the specialist bootstrap preprocessing ---------------------
    _run_specialist_bootstrap() {
        local reason="$1"
        echo ""
        echo "========================================"
        echo " [BOOTSTRAP] ${reason}"
        echo " Preprocessing with SPECIALIST_PREPROCESS=1"
        echo " (rasterises Bridge/Railway/Utility from shapefiles)"
        echo "========================================"

        if [ ! -d "$DATA_DIR_B" ]; then
            echo ""
            echo "[ERROR] Bootstrap needed but --data-dir '$DATA_DIR_B' does not exist."
            echo ""
            echo "  You must provide the raw data folder so Bridge/Railway/Utility"
            echo "  tiles can be rasterised from your shapefiles."
            echo ""
            echo "  Run with --data-dir pointing to your TIFFs:"
            echo "    docker compose run --rm specialist --data-dir /raw_data/ALL"
            echo ""
            echo "  Also check that config_specialist.py SHAPEFILE_MAP has entries for:"
            echo "    bridge, railway, utility"
            python3 src/07_notify.py --error \
                --folder "specialist-bootstrap" --step 0 --total 2 \
                --error-msg "Bootstrap needed (${reason}) but --data-dir not provided." \
                --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true
            exit 1
        fi

        if [ -z "$SHP_DIRS_B" ]; then
            SHP_DIRS_B=$(detect_shp_dirs "$DATA_DIR_B")
        fi
        if [ -z "$SHP_DIRS_B" ]; then
            echo "[ERROR] No SHP subdirectories found in $DATA_DIR_B"
            echo "  Expected: $DATA_DIR_B/SHP1/*.shp"
            echo "  Or: --shp-dirs /path/SHP1:/path/SHP2"
            exit 1
        fi

        check_ram_before_phase "specialist-bootstrap-preprocess" "16"
        RAM_AVAIL=$(available_ram_gb)
        BOOT_WORKERS=$(safe_preprocess_workers "$RAM_AVAIL")
        log "Bootstrap: ${BOOT_WORKERS} workers (${RAM_AVAIL} GB available)"
        log "  Data dir : $DATA_DIR_B"
        log "  SHP dirs : $SHP_DIRS_B"

        PROCESSED_BUDGET_GB=$(compute_processed_budget_gb)
        if ! env \
            RAW_DATA_DIR="$DATA_DIR_B" \
            SHP_DIR="$(echo "$SHP_DIRS_B" | cut -d: -f1)" \
            SHP_DIRS_LIST="$SHP_DIRS_B" \
            MAX_PROCESSED_GB="$PROCESSED_BUDGET_GB" \
            MAX_PREPROCESS_WORKERS="$BOOT_WORKERS" \
            SPECIALIST_PREPROCESS="1" \
            python3 src/01_preprocess.py 2>&1 | tee /tmp/specialist_bootstrap_log.txt; then

            log "[ERROR] Bootstrap preprocessing failed"
            python3 src/07_notify.py --error \
                --folder "specialist-bootstrap" --step 0 --total 2 \
                --error-msg "Bootstrap preprocessing failed. Check SHP dirs and config_specialist.py SHAPEFILE_MAP." \
                --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true
            echo ""
            echo "  Most likely cause: config_specialist.py SHAPEFILE_MAP is missing"
            echo "  entries for 'bridge', 'railway', and/or 'utility'."
            echo ""
            echo "  Add them and retry:"
            echo "    docker compose run --rm specialist --data-dir /raw_data/ALL"
            exit 1
        fi
        free_memory_between_phases

        TILE_COUNT=$(python3 -c "
import json
try:
    d = json.load(open('data/processed/tiles_meta.json'))
    tiles = d['tiles']
    minor = sum(1 for t in tiles if {4,5,6} & set(t.get('class_ids',[])))
    print(str(len(tiles)) + ' total  ' + str(minor) + ' minor-class')
except: print('0')
" 2>/dev/null || echo "0")
        log "Bootstrap complete: ${TILE_COUNT} tiles"
        notify_shell_milestone "specialist-bootstrap" "preprocess_done" \
            "Bootstrap preprocessing complete: ${TILE_COUNT} tiles."
        HAS_PROCESSED="1"
    }

    # -- Bootstrap case 1: no tiles at all -------------------------------------
    if [ "$HAS_PROCESSED" = "0" ] && [ "$HAS_REPLAY" = "0" ]; then
        _run_specialist_bootstrap "No tiles found -- preprocessing from raw data"
    fi

    # -- Bootstrap case 2: tiles exist but none have minor classes (4/5/6) -----
    # This is the common case: generalist was trained without Bridge/Railway/Utility
    # shapefiles, so all tiles only have class_ids 0-3.
    HAS_MINOR=$(python3 -c "
import json; from pathlib import Path; sys_exit = __import__('sys').exit
sources = []
p = Path('data/processed/tiles_meta.json')
if p.exists():
    sources += json.load(open(p)).get('tiles', [])
import sys; sys.path.insert(0,'src')
try:
    from replay_buffer import load_all_replay, replay_exists
    if replay_exists():
        tiles, _, _ = load_all_replay()
        sources += tiles
except: pass
minor = sum(1 for t in sources if {4,5,6} & set(t.get('class_ids',[])))
print('1' if minor > 0 else '0')
" 2>/dev/null || echo "0")

    if [ "$HAS_MINOR" = "0" ]; then
        log "No minor-class tiles (Bridge/Railway/Utility) found in processed/ or replay/"
        log "Triggering specialist bootstrap preprocessing..."
        _run_specialist_bootstrap "Tiles exist but none have class_ids 4/5/6 (Bridge/Railway/Utility)"
    else
        log "Minor-class tiles confirmed present -- skipping bootstrap"
    fi

    log "Tile source: processed=${HAS_PROCESSED}  replay=${HAS_REPLAY}"

    # -- Build specialist metadata (50/50 minor/major split) -------------------
    log "Building specialist metadata (50/50 minor/major split)..."
    python3 specialist/01_build_specialist_meta.py \
        2>&1 | tee /tmp/specialist_meta_log.txt
    META_EXIT=${PIPESTATUS[0]}

    if [ "$META_EXIT" -eq 2 ]; then
        # Exit code 2: tiles found but still no minor-class tiles after bootstrap.
        # This means the bootstrap ran but the shapefiles produced 0 minor tiles
        # (e.g. SHAPEFILE_MAP entries are missing or filenames are wrong).
        log "[ERROR] Specialist metadata build: zero minor-class tiles even after bootstrap"
        log "        Check that config_specialist.py SHAPEFILE_MAP has correct entries"
        log "        for bridge, railway, utility and that those .shp files exist."
        python3 src/07_notify.py --error \
            --folder "specialist-meta" --step 0 --total 2 \
            --error-msg "Zero minor-class tiles found even after bootstrap. Check config_specialist.py SHAPEFILE_MAP." \
            --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true
        exit 1
    elif [ "$META_EXIT" -ne 0 ]; then
        log "[ERROR] Specialist metadata build failed (exit ${META_EXIT})"
        python3 src/07_notify.py --error \
            --folder "specialist-meta" --step 0 --total 2 \
            --error-msg "Specialist metadata build failed (exit ${META_EXIT})." \
            --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true
        exit 1
    fi
    META_SUMMARY=$(python3 -c "
import json
try:
    d = json.load(open('data/processed/tiles_meta_specialist.json'))
    print(str(d.get('total_tiles','-')) + ' tiles  minor=' + str(d.get('n_minor_tiles','-')) + '  major=' + str(d.get('n_major_tiles','-')))
except: print('-')
" 2>/dev/null || echo "-")
    log "Specialist metadata: ${META_SUMMARY}"
    notify_shell_milestone "specialist" "preprocess_done" \
        "Specialist meta built: ${META_SUMMARY}."

    check_ram_before_phase "specialist-training" "$MIN_RAM_GB"
    free_memory_between_phases

    log "Training specialist (v${SPEC_VERSION})..."
    if ! python3 specialist/03_train_specialist.py \
            --run-version "$SPEC_VERSION" \
            --run-message "$RUN_MESSAGE" \
            2>&1 | tee /tmp/specialist_train_log.txt; then
        log "[ERROR] Specialist training failed"
        python3 src/07_notify.py --error \
            --folder "specialist" --step 2 --total 2 \
            --error-msg "Specialist training failed. See logs." \
            --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true
        echo "  Resume: $SCRIPT_SELF --specialist-only --message 'resume'"
        exit 1
    fi
    free_memory_between_phases

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
    print('Bridge=' + f'{g(4):.4f}' + '  Railway=' + f'{g(5):.4f}' + '  Utility=' + f'{g(6):.4f}')
except Exception as e:
    print('(per-class: ' + str(e) + ')')
" 2>/dev/null || echo "(unavailable)")

    log "Specialist done -- val_mIoU=${SPEC_VAL_MIOU}  ${SPEC_PER_CLASS}"
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
    echo " Minor-class: ${SPEC_PER_CLASS}"
    echo ""
    echo " Combined inference:"
    echo "   docker compose run --rm combined-infer \\"
    echo "       --input /raw_data/image.tif --output /app/outputs/mask.tif"
    echo "========================================"
    exit 0
fi

# =============================================================================
# ENTRY POINTS 1 & 2: Generalist shards (+ optional specialist)
# =============================================================================

if [ ! -d "$DATA_DIR_B" ]; then
    echo "[ERROR] --data-dir '$DATA_DIR_B' does not exist."
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
    exit 1
fi

HALF=$(( N_TIFS / 2 ))
SHARD1_FILES=$(IFS=:; echo "${ALL_TIFS[*]:0:$HALF}")
SHARD2_FILES=$(IFS=:; echo "${ALL_TIFS[*]:$HALF}")

if [ -z "$SHP_DIRS_B" ]; then
    SHP_DIRS_B=$(detect_shp_dirs "$DATA_DIR_B")
fi
if [ -z "$SHP_DIRS_B" ]; then
    echo "[ERROR] No SHP subdirectories found in $DATA_DIR_B"
    echo "  Expected: $DATA_DIR_B/SHP1/*.shp"
    echo "  Or: --shp-dirs /path/SHP1:/path/SHP2"
    exit 1
fi

TOTAL=2
if [ "$START_FROM" -lt 1 ] || [ "$START_FROM" -gt "$TOTAL" ]; then
    echo "[ERROR] --from must be 1 or 2 (got $START_FROM)"
    exit 1
fi

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

# -- Fresh start wipe ----------------------------------------------------------
if [ "$START_FROM" -eq 1 ]; then
    log "Fresh start -- wiping stale data..."
    [ -d "data/replay" ]    && find data/replay    -mindepth 1 -delete && log "  Cleared data/replay/"
    [ -d "data/processed" ] && find data/processed -mindepth 1 -delete && log "  Cleared data/processed/"

    if [ -n "$INIT_WEIGHTS" ]; then
        [ ! -f "$INIT_WEIGHTS" ] && echo "[ERROR] --init-weights not found: $INIT_WEIGHTS" && exit 1
        rm -f checkpoints/best_model.pt checkpoints/training_curves.png checkpoints/history.json
        log "  Init weights: $INIT_WEIGHTS"
    elif [ -n "$PRETRAINED_CKPT" ]; then
        [ ! -f "$PRETRAINED_CKPT" ] && echo "[ERROR] --pretrained not found: $PRETRAINED_CKPT" && exit 1
        mkdir -p checkpoints
        cp "$PRETRAINED_CKPT" checkpoints/best_model.pt
        log "  Installed pretrained -> checkpoints/best_model.pt"
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

    free_memory_between_phases
fi

# -- Crash / ERR trap ----------------------------------------------------------
CURRENT_FOLDER="unknown"
CURRENT_STEP=0

_shell_crash() {
    local EXIT_CODE=${1:-$-}
    log "[CRASH] Pipeline error at $CURRENT_FOLDER (exit $EXIT_CODE)"
    log "  Resume: $SCRIPT_SELF --from $CURRENT_STEP --data-dir $DATA_DIR_B"
    log "  Or specialist only: $SCRIPT_SELF --specialist-only"
    python3 src/07_notify.py --error \
        --folder "$CURRENT_FOLDER" \
        --step "$CURRENT_STEP" \
        --total "$TOTAL" \
        --error-msg "Pipeline crashed (exit $EXIT_CODE) at $CURRENT_FOLDER. Resume: $SCRIPT_SELF --from $CURRENT_STEP --data-dir $DATA_DIR_B" \
        --version "$GEN_VERSION" \
        --run-message "$RUN_MESSAGE" || true
}
trap '_shell_crash $-' ERR

# =============================================================================
# MAIN LOOP -- two generalist shards
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

    TIFF_COUNT=$(echo "${TIFF_FILES_PER_SHARD[$i]}" | tr ':' '\n' | grep -c . || echo "-")

    echo ""
    echo "========================================"
    echo " Shard $STEP/$TOTAL: $FOLDER_NAME  (${TIFF_COUNT} TIFFs)"
    echo " Generalist v${GEN_VERSION}"
    [ -n "$RUN_MESSAGE" ] && echo " Message: $RUN_MESSAGE"
    [ "$IS_LAST_FOR_SPECIALIST" -eq 1 ] && \
        echo " [NOTE] Tiles kept for specialist after this shard"
    echo "========================================"

    # -- Pre-phase RAM check + dynamic worker count ----------------------------
    check_ram_before_phase "preprocess-${FOLDER_NAME}" "16"
    RAM_AVAIL=$(available_ram_gb)
    MAX_WORKERS=$(safe_preprocess_workers "$RAM_AVAIL")
    log "RAM: ${RAM_AVAIL} GB available -> MAX_PREPROCESS_WORKERS=${MAX_WORKERS}"

    # -- Disk accounting -------------------------------------------------------
    REPLAY_BYTES=$(dir_bytes "data/replay")
    REPLAY_GB=$(python3 -c "print(f'{${REPLAY_BYTES}/1024**3:.3f}')")
    PROCESSED_BUDGET_GB=$(compute_processed_budget_gb)
    log "Disk: replay=${REPLAY_GB} GB  budget=${PROCESSED_BUDGET_GB} GB  ceiling=${MAX_DISK_GB} GB"
    python3 -c "
replay_gb=${REPLAY_GB}; ceiling=${MAX_DISK_GB}; budget=${PROCESSED_BUDGET_GB}
assert replay_gb < ceiling, f'[ERROR] Replay exceeds ceiling! {replay_gb:.2f} >= {ceiling}'
assert budget >= 1.0, f'[ERROR] Budget < 1 GB -- increase MAX_DISK_GB'
"

    # -- Preprocess ------------------------------------------------------------
    log "Preprocessing $FOLDER_NAME (workers=${MAX_WORKERS})..."
    if ! env \
        RAW_DATA_DIR="$DATA_DIR_B" \
        SHP_DIR="$(echo "$SHP_DIRS_B" | cut -d: -f1)" \
        SHP_DIRS_LIST="$SHP_DIRS_B" \
        MAX_PROCESSED_GB="$PROCESSED_BUDGET_GB" \
        MAX_PREPROCESS_WORKERS="$MAX_WORKERS" \
        TIFF_FILES="${TIFF_FILES_PER_SHARD[$i]}" \
        python3 src/01_preprocess.py 2>&1 | tee /tmp/preprocess_log.txt; then

        log "[ERROR] Preprocessing failed for $FOLDER_NAME"
        python3 src/07_notify.py --error \
            --folder "$FOLDER_NAME" --step "$STEP" --total "$TOTAL" \
            --error-msg "Preprocessing failed for $FOLDER_NAME. Check shapefile coverage." \
            --version "$GEN_VERSION" --run-message "$RUN_MESSAGE" || true
        echo "  Inspect: docker compose run --rm inspect"
        echo "  Retry: $SCRIPT_SELF --from $STEP --data-dir $DATA_DIR_B"
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
    raise SystemExit('[ERROR] Disk ceiling exceeded: ' + str(round(total_gb,2)) + ' GB > ' + str(ceiling) + ' GB')
"

    # -- Free memory after preprocessing before replay save --------------------
    free_memory_between_phases

    # -- Save replay -----------------------------------------------------------
    log "Saving replay slice for $FOLDER_NAME..."
    if python3 -c "
import sys; sys.path.insert(0,'src')
from replay_buffer import save_replay_from_shard
save_replay_from_shard('$FOLDER_NAME')
" 2>&1 | tee /tmp/replay_log.txt; then
        REPLAY_TILES=$(grep -oP "Saved \K[0-9]+" /tmp/replay_log.txt | tail -1 || echo "-")
        log "Replay saved: ~${REPLAY_TILES} tiles"
        notify_shell_milestone "$FOLDER_NAME" "replay_saved" \
            "Replay updated: ~${REPLAY_TILES} tiles from ${FOLDER_NAME}."
    else
        log "[WARN] Replay save failed -- continuing without replay for this shard"
    fi

    # -- Free memory before training -------------------------------------------
    check_ram_before_phase "training-${FOLDER_NAME}" "$MIN_RAM_GB"
    free_memory_between_phases

    # -- Train -----------------------------------------------------------------
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
            --error-msg "Training failed at $FOLDER_NAME (exit $TRAIN_EXIT). Checkpoint is safe." \
            --version "$GEN_VERSION" --run-message "$RUN_MESSAGE" || true
        echo "  Resume: $SCRIPT_SELF --from $STEP --data-dir $DATA_DIR_B"
        exit 1
    fi

    TRAIN_LOSS=$(grep "tr_loss="  /tmp/train_log.txt | tail -1 | grep -oP "tr_loss=\K[0-9.]+"  || echo "0")
    VAL_LOSS=$(  grep "val_loss=" /tmp/train_log.txt | tail -1 | grep -oP "val_loss=\K[0-9.]+" || echo "0")
    TRAIN_MIOU=$(grep "tr_mIoU="  /tmp/train_log.txt | tail -1 | grep -oP "tr_mIoU=\K[0-9.]+"  || echo "0")
    VAL_MIOU=$(  grep "val_mIoU=" /tmp/train_log.txt | tail -1 | grep -oP "val_mIoU=\K[0-9.]+" || echo "0")
    EPOCHS=$(grep -c "^Epoch " /tmp/train_log.txt 2>/dev/null | tr -d '[:space:]' || echo "0")
    log "Done -- val_mIoU=${VAL_MIOU}  epochs=${EPOCHS}"

    python3 src/07_notify.py \
        --folder "Generalist v${GEN_VERSION} -- ${FOLDER_NAME}" \
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

    # -- Free memory after training --------------------------------------------
    free_memory_between_phases

    # -- Wipe processed tiles --------------------------------------------------
    if [ "$IS_LAST_FOR_SPECIALIST" -eq 0 ]; then
        log "Clearing processed tiles (intermediate shard)..."
        rm -rf data/processed/images data/processed/masks data/processed/tiles_meta.json
        REPLAY_BYTES_AFTER=$(dir_bytes "data/replay")
        python3 -c "
replay_gb=${REPLAY_BYTES_AFTER}/1024**3; ceiling=${MAX_DISK_GB}
print(f'  Disk after wipe: replay={replay_gb:.2f} GB  (ceiling={ceiling} GB)')
"
        # Free page cache now that large .npy files are deleted
        free_memory_between_phases
    else
        log "Keeping processed tiles alive for specialist pipeline..."
    fi

    log "OK Finished $FOLDER_NAME ($STEP/$TOTAL)"
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
    echo "========================================"

    log "Building specialist metadata (50/50 minor/major split)..."
    python3 specialist/01_build_specialist_meta.py \
        2>&1 | tee /tmp/specialist_meta_log.txt
    META_EXIT=${PIPESTATUS[0]}

    if [ "$META_EXIT" -eq 2 ]; then
        # Exit code 2: generalist tiles exist but none have minor-class IDs 4/5/6.
        # This means the user's shapefiles don't include Bridge/Railway/Utility.
        # The specialist cannot be trained without those tiles.
        log "[ERROR] Zero minor-class tiles found. Specialist cannot train without Bridge/Railway/Utility tiles."
        log "  To fix: add bridge/railway/utility entries to config_specialist.py SHAPEFILE_MAP"
        log "  then re-run: $SCRIPT_SELF --specialist-only --data-dir $DATA_DIR_B"
        python3 src/07_notify.py --error \
            --folder "specialist-meta" --step "$TOTAL" --total "$TOTAL" \
            --error-msg "Zero minor-class tiles (Bridge/Railway/Utility). Add shapefiles to config_specialist.py SHAPEFILE_MAP and retry with --specialist-only." \
            --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true
        exit 1
    elif [ "$META_EXIT" -ne 0 ]; then
        log "[ERROR] Specialist metadata build failed (exit ${META_EXIT})"
        python3 src/07_notify.py --error \
            --folder "specialist-meta" --step "$TOTAL" --total "$TOTAL" \
            --error-msg "Specialist metadata build failed (exit ${META_EXIT})." \
            --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true
        echo "  Retry: $SCRIPT_SELF --specialist-only"
        exit 1
    fi
    notify_shell_milestone "specialist" "preprocess_done" "Specialist metadata built."

    check_ram_before_phase "specialist-training" "$MIN_RAM_GB"
    free_memory_between_phases

    log "Training specialist (v${SPEC_VERSION})..."
    if ! env MAX_DISK_GB="$MAX_DISK_GB" \
            python3 specialist/03_train_specialist.py \
            --run-version "$SPEC_VERSION" \
            --run-message "$RUN_MESSAGE" \
            2>&1 | tee /tmp/specialist_train_log.txt; then
        SPEC_TRAIN_EXIT=${PIPESTATUS[0]}
        log "[ERROR] Specialist training failed (exit $SPEC_TRAIN_EXIT)"
        python3 src/07_notify.py --error \
            --folder "specialist" --step "$TOTAL" --total "$TOTAL" \
            --error-msg "Specialist training failed (exit $SPEC_TRAIN_EXIT)." \
            --version "$SPEC_VERSION" --run-message "$RUN_MESSAGE" || true
        echo "  Retry: $SCRIPT_SELF --specialist-only"
        exit 1
    fi
    free_memory_between_phases

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
    print('Bridge=' + f'{g(4):.4f}' + '  Railway=' + f'{g(5):.4f}' + '  Utility=' + f'{g(6):.4f}')
except Exception as e:
    print('(per-class: ' + str(e) + ')')
" 2>/dev/null || echo "(unavailable)")

    log "Specialist done -- val_mIoU=${SPEC_VAL_MIOU}  ${SPEC_PER_CLASS}"

    # Post-specialist wipe
    if [ -d "data/processed/images" ] || [ -d "data/processed/masks" ] || \
       [ -f "data/processed/tiles_meta.json" ]; then
        log "Clearing last shard processed tiles (post-specialist)..."
        rm -rf data/processed/images data/processed/masks data/processed/tiles_meta.json
        free_memory_between_phases
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
    echo "   docker compose run --rm combined-infer \\"
    echo "       --input /raw_data/image.tif --output /app/outputs/combined_mask.tif"
    echo ""
    echo " Version history:  python src/run_version.py --list"
    echo "========================================"

else
    echo ""
    echo "[SKIP] --skip-specialist -- specialist stage bypassed"
    echo ""
    echo "========================================"
    echo " Pipeline complete (generalist only)"
    echo " Generalist v${GEN_VERSION}: checkpoints/best_model.pt"
    echo "========================================"
fi