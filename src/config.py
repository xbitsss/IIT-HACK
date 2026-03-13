"""
config.py — All pipeline settings in one place.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_RAW_DIR       = os.environ.get("RAW_DATA_DIR", "data/raw")
DATA_PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "data/processed")
CHECKPOINT_DIR     = os.environ.get("CHECKPOINT_DIR", "checkpoints")
OUTPUT_DIR         = os.environ.get("OUTPUT_DIR", "outputs")

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

# Render priority — higher = painted last = wins overlaps
CLASS_PRIORITY = {
    "builtup":   1,
    "waterbody": 2,
    "road":      3,
}

# ─── Band Configuration ───────────────────────────────────────────────────────
BAND_INDICES = [1, 2, 3, 4]

# ─── Tiling ───────────────────────────────────────────────────────────────────
TILE_SIZE       = 512
TILE_OVERLAP    = 64
MIN_VALID_RATIO = 0.1

# ─── Training ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "nvidia/mit-b2"
BATCH_SIZE   = 8
NUM_EPOCHS   = 50
LR           = 6e-5
WEIGHT_DECAY = 0.01
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42

CLASS_WEIGHTS = [0.5, 1.0, 3.0, 1.5]

PATIENCE = 10

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUGMENT_TRAIN = True

# ─── Inference ────────────────────────────────────────────────────────────────
INFERENCE_OVERLAP = 128
INFERENCE_BATCH   = 4