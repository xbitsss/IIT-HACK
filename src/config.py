"""
config.py — All pipeline settings in one place.
Edit this file to match your data before running anything.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
# DATA_RAW_DIR is read from the environment variable RAW_DATA_DIR if set.
# This lets you point Docker at an external drive without copying anything.
#
# Docker usage:
#   docker compose run -v /external/drive/path:/data_ext --rm preprocess
#   and set RAW_DATA_DIR=/data_ext in docker-compose.yml or .env
#
# Local usage: just set RAW_DATA_DIR in your shell, or leave as default.
DATA_RAW_DIR       = os.environ.get("RAW_DATA_DIR", "data/raw")
DATA_PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "data/processed")
CHECKPOINT_DIR     = os.environ.get("CHECKPOINT_DIR", "checkpoints")
OUTPUT_DIR         = os.environ.get("OUTPUT_DIR", "outputs")

# ─── Class Definitions ────────────────────────────────────────────────────────
# Keys must be substrings of your actual shapefile filenames (case-insensitive).
# e.g. "Water_Body" shapefile matches key "water_body"
#
# From your QGIS layers:
#   shp-file — Waterbody_Point      → waterbody (point features)
#   shp-file — Water_Body_Line      → waterbody (line features)
#   shp-file — Road_Centre_Line     → road
#   shp-file — Railway              → railway
#   shp-file — Utility              → utility
#   shp-file — Water_Body           → waterbody (polygon)
#   shp-file — Utility_Poly         → utility
#   shp-file — Road                 → road (polygon)
#   shp-file — Built_Up_Area_type   → builtup
#   shp-file — Bridge               → road (bridges are road class)

CLASSES = {
    "background":        0,
    "built_up_area":     1,   # matches: Built_Up_Area_type
    "road":              2,   # matches: Road, Road_Centre_Line, Bridge
    "water_body":        3,   # matches: Water_Body, Water_Body_Line, Waterbody_Point
    "railway":           4,   # matches: Railway
    "utility":           5,   # matches: Utility, Utility_Poly
}
NUM_CLASSES = len(CLASSES)

# Human-readable labels (must be in same order as CLASSES values above)
CLASS_LABELS = ["Background", "Built-up", "Road", "Water Body", "Railway", "Utility"]

# Visualization colors (RGB tuples)
CLASS_COLORS = {
    0: (50,  50,  50),    # background  — dark grey
    1: (210, 140,  80),   # built-up    — terracotta
    2: (255, 220,  80),   # road        — yellow
    3: (80,  160, 230),   # water body  — blue
    4: (180,  80, 180),   # railway     — purple
    5: (80,  200, 130),   # utility     — green
}

# ─── Shapefile → Class mapping ────────────────────────────────────────────────
# Explicit mapping: shapefile filename keyword → class key in CLASSES above.
# The preprocessor will match shapefiles whose names CONTAIN these keywords
# (case-insensitive). Add or remove entries to match your exact filenames.
SHAPEFILE_CLASS_MAP = {
    "built_up_area":    "built_up_area",
    "road_centre_line": "road",
    "road":             "road",
    "bridge":           "road",         # bridges are road class
    "water_body_line":  "water_body",
    "water_body":       "water_body",
    "waterbody_point":  "water_body",
    "waterbody":        "water_body",
    "railway":          "railway",
    "utility_poly":     "utility",
    "utility":          "utility",
}

# Render priority — higher number wins when masks overlap
# (e.g. a road on top of built-up area → road wins)
CLASS_PRIORITY = {
    "background":    0,
    "utility":       1,
    "built_up_area": 2,
    "water_body":    3,
    "railway":       4,
    "road":          5,   # road always wins
}

# ─── Band Configuration ───────────────────────────────────────────────────────
# None = use all bands from the TIFF (recommended — auto-detected)
# [0,1,2]   = RGB only
# [0,1,2,3] = RGB + NIR
BAND_INDICES = None

# ─── Tiling ───────────────────────────────────────────────────────────────────
TILE_SIZE       = 512   # reduce to 256 if you run out of GPU memory
TILE_OVERLAP    = 64
MIN_VALID_RATIO = 0.1   # skip tiles where >90% pixels are NoData

# ─── Training ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "nvidia/mit-b2"
BATCH_SIZE   = 8
NUM_EPOCHS   = 50
LR           = 6e-5
WEIGHT_DECAY = 0.01
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42

# One weight per class — increase for underrepresented classes
# Order: background, built_up, road, water_body, railway, utility
CLASS_WEIGHTS = [0.5, 1.0, 3.0, 1.5, 3.0, 2.0]

PATIENCE = 10

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUGMENT_TRAIN = True

# ─── Inference ────────────────────────────────────────────────────────────────
INFERENCE_OVERLAP = 128
INFERENCE_BATCH   = 4
