"""
config.py — All pipeline settings in one place.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_RAW_DIR       = os.environ.get("RAW_DATA_DIR", "data/raw")
DATA_PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "data/processed")
CHECKPOINT_DIR     = os.environ.get("CHECKPOINT_DIR", "checkpoints")
OUTPUT_DIR         = os.environ.get("OUTPUT_DIR", "outputs")

# ─── Shapefile directory ──────────────────────────────────────────────────────
# Shapefiles are shared across all TIFF folders (one CG_SHP dir for everything)
SHP_DIR = os.environ.get("SHP_DIR", "data/raw/CG_SHP")

# ─── Exact shapefile filenames → class name ───────────────────────────────────
SHAPEFILE_MAP = {
    "builtup":   "Built_Up_Area_type.shp",
    "road":      "Road.shp",
    "waterbody": "Water_Body.shp",
}

# ─── Class Definitions ────────────────────────────────────────────────────────
CLASSES = {
    "background": 0,
    "builtup":    1,
    "road":       2,
    "waterbody":  3,
}
NUM_CLASSES = len(CLASSES)

CLASS_LABELS = ["Background", "Built-up", "Road", "Water Body"]

CLASS_COLORS = {
    0: (50,  50,  50),
    1: (210, 140,  80),
    2: (255, 220,  80),
    3: (80,  160, 230),
}

CLASS_PRIORITY = {
    "builtup":   1,
    "waterbody": 2,
    "road":      3,   # road drawn last → wins overlaps
}

# ─── Band Configuration ───────────────────────────────────────────────────────
BAND_INDICES = [1, 2, 3, 4]

# ─── Tiling ───────────────────────────────────────────────────────────────────
TILE_SIZE       = 512
TILE_OVERLAP    = 64
MIN_VALID_RATIO = 0.1   # fraction of non-zero pixels required

# Minimum fraction of a tile that must intersect annotated polygons.
# Tiles entirely outside ALL shapefile extents are unannotated noise — skip them.
# Set to 0 to disable (keep all tiles that pass MIN_VALID_RATIO).
MIN_COVERAGE_RATIO = 0.05

# ─── Training ─────────────────────────────────────────────────────────────────
# mit-b3 gives a better accuracy/compute tradeoff vs b2 for multi-band geospatial data
MODEL_NAME   = "nvidia/mit-b3"
BATCH_SIZE   = 6           # per-GPU batch; effective = BATCH_SIZE × GRAD_ACCUM_STEPS
NUM_EPOCHS   = 60
LR           = 5e-5
WEIGHT_DECAY = 0.01
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42

# Class weights: down-weight background, up-weight rare road pixels
CLASS_WEIGHTS = [0.4, 1.2, 4.0, 2.0]

PATIENCE = 12   # early-stop patience (epochs without val-mIoU improvement)

# ─── Advanced Training Tricks ─────────────────────────────────────────────────
# Mixed-precision training (FP16) — cuts memory ~40%, speeds up ~1.5×
USE_AMP = True

# Gradient accumulation — effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS
GRAD_ACCUM_STEPS = 4

# LR warmup before cosine decay
WARMUP_EPOCHS = 5

# Exponential Moving Average of weights — improves val stability / generalization
USE_EMA   = True
EMA_DECAY = 0.9998

# ─── Loss ────────────────────────────────────────────────────────────────────
# Focal + Dice combination
# Focal: focuses learning on hard/misclassified pixels
# Dice: directly optimises overlap metric, handles class imbalance
FOCAL_GAMMA   = 2.0
FOCAL_WEIGHT  = 0.5
DICE_WEIGHT   = 0.5

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUGMENT_TRAIN = True

# ─── Class-balanced sampling ──────────────────────────────────────────────────
# Oversample tiles that contain rare classes during training
USE_WEIGHTED_SAMPLER = True

# Per-class tile-weight multiplier (applied on top of class frequency)
# Background-only tiles get weight 1; tiles with road/water get boosted
SAMPLER_CLASS_WEIGHTS = {
    0: 0.3,   # background — de-emphasise all-background tiles
    1: 1.5,   # builtup
    2: 5.0,   # road      — most rare → highest boost
    3: 3.0,   # waterbody
}

# ─── Inference ────────────────────────────────────────────────────────────────
INFERENCE_OVERLAP = 192   # larger overlap → smoother borders
INFERENCE_BATCH   = 4

# Test-time augmentation: average predictions over flips & rotations
USE_TTA = True

# ─── Replay Buffer (continual learning / anti-forgetting) ────────────────────
# Persistent directory for replay tiles — NEVER wiped between shards.
# Separate from data/processed/ which is cleared after each shard.
REPLAY_DIR = os.environ.get("REPLAY_DIR", "data/replay")

# How many tiles to keep from each shard.
# Budget: 300 tiles × 4 bands × 512² × float32 ≈ 300 MB per shard
# 4 shards × 300 = 1200 tiles ≈ 1.2 GB total — adjust down if disk is tight.
REPLAY_TILES_PER_SHARD = 300

# Fraction of each training batch drawn from the replay buffer.
# 0.25 = 25% of every batch is old-shard tiles.  This is the primary
# mechanism preventing catastrophic forgetting.
REPLAY_RATIO = 0.25

# ─── Notifications ────────────────────────────────────────────────────────────
# Send a progress email every N hours during training (0 = disable)
NOTIFY_INTERVAL_HOURS = 2