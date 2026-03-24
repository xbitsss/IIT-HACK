"""
config.py — All pipeline settings in one place.

Disk budget guarantee
─────────────────────
At no point in time will the pipeline use more than MAX_DISK_GB of storage
for pipeline-generated data.  "Pipeline-generated data" means:

    data/processed/   ← current shard tiles (wiped after each shard)
    data/replay/      ← replay tiles kept from all previous shards (never wiped)

The shell script (06_train_incremental.sh) calculates exactly how many bytes
the replay buffer already occupies before starting each shard, then sets
MAX_PROCESSED_GB = MAX_DISK_GB − replay_bytes_already_on_disk
so that processed + replay never exceed MAX_DISK_GB together.

Tile size at TILE_SIZE=512, 4 bands float32 image + uint8 mask:
  image : 4 × 512 × 512 × 4 bytes = 4,194,304 bytes ≈ 4.00 MB
  mask  :     512 × 512 × 1 byte  =   262,144 bytes ≈ 0.25 MB
  total per tile                   ≈ 4.25 MB

With 2 dataset shards × 300 replay tiles = 600 replay tiles ≈ 2.5 GB total replay.
That leaves ≈ 47.5 GB for processed tiles on the last shard (~11,176 tiles).
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PROCESSED_DIR = os.environ.get("PROCESSED_DIR",  "data/processed")
CHECKPOINT_DIR     = os.environ.get("CHECKPOINT_DIR", "checkpoints")
OUTPUT_DIR         = os.environ.get("OUTPUT_DIR",     "outputs")

# ─── Current shard folder (set per-shard by the shell script) ─────────────────
# The shell script exports RAW_DATA_DIR=<folder> before calling 01_preprocess.py
# for each shard.  This is the only folder scanned during that preprocessing run.
DATA_RAW_DIR = os.environ.get("RAW_DATA_DIR", "/raw_data/CG")

# All shard folders — auto-discovered from RAW_DATA_ROOT at import time.
# Every immediate subdirectory of RAW_DATA_ROOT that contains TIFF files
# is treated as one shard.  No list to maintain — just drop a folder in.
_raw_dirs_env = os.environ.get("RAW_DATA_DIRS", "")
if _raw_dirs_env:
    # Manual override: colon-separated list of explicit shard paths
    ALL_RAW_DIRS = [p.strip() for p in _raw_dirs_env.split(":") if p.strip()]
else:
    # Auto-discover: scan RAW_DATA_ROOT for subdirs containing TIFFs
    _root = os.environ.get("RAW_DATA_ROOT", "/raw_data")
    try:
        from pathlib import Path as _Path
        _root_path = _Path(_root)
        if _root_path.exists():
            ALL_RAW_DIRS = sorted(
                str(d) for d in _root_path.iterdir()
                if d.is_dir() and (
                    list(d.rglob("*.tif")) or list(d.rglob("*.tiff"))
                )
            )
        else:
            ALL_RAW_DIRS = []
    except Exception:
        ALL_RAW_DIRS = []

    # Fallback if discovery finds nothing (e.g. during build time)
    if not ALL_RAW_DIRS:
        ALL_RAW_DIRS = ["/raw_data/CG", "/raw_data/PB"]

# ─── Shapefile directory ───────────────────────────────────────────────────────
# SHP_DIR is auto-detected per dataset by the shell script — it finds the
# subdirectory inside each dataset folder that contains .shp files.
# The default here is only used when running 01_preprocess.py standalone.
SHP_DIR = os.environ.get("SHP_DIR", "/raw_data/CG/CG_SHP")

# ─── Exact shapefile filenames → class name ───────────────────────────────────
# Each class maps to a LIST of filenames — all are merged before rasterization.
# Geometry types are handled automatically:
#   polygon → rasterized directly
#   line    → buffered by LINE_BUFFER_M metres then rasterized
#   point   → buffered by POINT_BUFFER_M metres then rasterized
SHAPEFILE_MAP = {
    "builtup":   ["Built_Up_Area_type.shp"],
    "road":      ["Road.shp", "Road_Centre_Line.shp"],
    "waterbody": ["Water_Body.shp", "Water_Body_Line.shp", "Waterbody_Point.shp"],
    "bridge":    ["Bridge.shp"],
    "railway":   ["Railway.shp"],
    "utility":   ["Utility_Poly.shp", "Utility_Poly_.shp", "Utility.shp"],
}

# Buffer sizes for non-polygon geometries (in metres, converted to CRS units at runtime)
POINT_BUFFER_M = 8.0   # point features → 8 m radius disc
LINE_BUFFER_M  = 3.0   # line features  → 3 m half-width strip

# ─── Class Definitions ────────────────────────────────────────────────────────
CLASSES = {
    "background": 0,
    "builtup":    1,
    "road":       2,
    "waterbody":  3,
    "bridge":     4,
    "railway":    5,
    "utility":    6,
}
NUM_CLASSES  = len(CLASSES)
CLASS_LABELS = ["Background", "Built-up", "Road", "Water Body", "Bridge", "Railway", "Utility"]

CLASS_COLORS = {
    0: (50,  50,  50),   # dark grey  — background
    1: (210, 140,  80),  # orange     — built-up
    2: (255, 220,  80),  # yellow     — road
    3: (80,  160, 230),  # blue       — water body
    4: (180,  80,  80),  # red        — bridge
    5: (140,  80, 200),  # purple     — railway
    6: (80,  200, 120),  # green      — utility
}

CLASS_PRIORITY = {
    # Higher priority = painted last (wins over lower priority)
    "bridge":    1,   # bridges sit on top of roads/water
    "railway":   2,
    "builtup":   3,
    "waterbody": 4,
    "utility":   5,
    "road":      6,   # road painted last → road lines visible over builtup
}

# ─── Band Configuration ───────────────────────────────────────────────────────
BAND_INDICES = [1, 2, 3, 4]

# ─── Tiling ───────────────────────────────────────────────────────────────────
TILE_SIZE          = 512
TILE_OVERLAP       = 64
MIN_VALID_RATIO    = 0.1    # min fraction of non-nodata pixels per tile
MIN_COVERAGE_RATIO = 0.05   # min fraction overlapping a polygon (hard floor)

# ─── Disk budget ─────────────────────────────────────────────────────────────
# Absolute ceiling on ALL pipeline-generated data on disk simultaneously.
# (processed tiles) + (replay buffer) must never exceed this value.
MAX_DISK_GB = float(os.environ.get("MAX_DISK_GB", "50"))

# Per-shard processed-tile budget (GB).
# Injected by the shell script each shard as:
#   MAX_PROCESSED_GB = MAX_DISK_GB - <replay bytes already on disk in GB>
# Defaults to MAX_DISK_GB so running 01_preprocess.py standalone is safe.
MAX_PROCESSED_GB = float(os.environ.get("MAX_PROCESSED_GB", str(MAX_DISK_GB)))

# ─── Replay buffer ────────────────────────────────────────────────────────────
REPLAY_DIR             = os.environ.get("REPLAY_DIR", "data/replay")
REPLAY_TILES_PER_SHARD = int(os.environ.get("REPLAY_TILES_PER_SHARD", "300"))
REPLAY_RATIO           = 0.25   # fraction of each batch from replay

# ─── Training ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "nvidia/mit-b5"
BATCH_SIZE   = 2
NUM_EPOCHS   = 150
LR           = 3e-5
WEIGHT_DECAY = 0.05
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42

CLASS_WEIGHTS = [0.3, 1.2, 5.0, 2.0, 4.0, 6.0, 3.0]
PATIENCE      = 30

# ─── Advanced training ────────────────────────────────────────────────────────
USE_AMP          = True
GRAD_ACCUM_STEPS = 16
WARMUP_EPOCHS    = 8
USE_EMA          = False
EMA_DECAY        = 0.9998

# ─── Loss ─────────────────────────────────────────────────────────────────────
FOCAL_GAMMA  = 2.0
FOCAL_WEIGHT = 0.5
DICE_WEIGHT  = 0.5

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUGMENT_TRAIN = True

# ─── Class-balanced sampling ──────────────────────────────────────────────────
USE_WEIGHTED_SAMPLER  = True
SAMPLER_CLASS_WEIGHTS = {
    0: 0.2,   # background  — abundant, down-weight strongly
    1: 1.5,   # built-up
    2: 5.0,   # road        — thin linear feature, boost
    3: 2.0,   # water body
    4: 6.0,   # bridge      — rare, thin
    5: 7.0,   # railway     — rare, very thin linear
    6: 4.0,   # utility     — mix of points and small polygons
}

# ─── Inference ────────────────────────────────────────────────────────────────
INFERENCE_OVERLAP = 192
INFERENCE_BATCH   = 2
USE_TTA           = True

# ─── Notifications ────────────────────────────────────────────────────────────
NOTIFY_INTERVAL_HOURS = 2