import sys
from pathlib import Path

# inherit everything from the main config
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import *  # noqa: F401, F403

# ── paths ──────────────────────────────────────────────────────────────────
CHECKPOINT_DIR   = str(Path(__file__).parent / "checkpoints")
TILES_META_PATH  = "data/processed/tiles_meta_specialist.json"

# ── model ──────────────────────────────────────────────────────────────────
# mit-b2 generalises better than mit-b5 on a smaller specialist dataset
MODEL_NAME       = "nvidia/mit-b2"

# ── 7-class overrides ──────────────────────────────────────────────────────
# Base config has 4 classes (Background, Built-up, Road, Water Body).
# Specialist adds Bridge=4, Railway=5, Utility=6.
# ALL class-count constants must be overridden here so FocalLoss,
# DiceLoss, and the DataLoaders see 7 classes, not 4.

NUM_CLASSES  = 7

CLASS_LABELS = [
    "Background",   # 0
    "Built-up",     # 1
    "Road",         # 2
    "Water Body",   # 3
    "Bridge",       # 4  ← minor
    "Railway",      # 5  ← minor
    "Utility",      # 6  ← minor
]

CLASSES = {
    "background": 0,
    "builtup":    1,
    "road":       2,
    "waterbody":  3,
    "bridge":     4,
    "railway":    5,
    "utility":    6,
}

# BGR color tuples (OpenCV convention, same as base config)
CLASS_COLORS = {
    0: (50,  50,  50),   # Background  – dark grey
    1: (210, 140,  80),  # Built-up    – tan
    2: (255, 220,  80),  # Road        – yellow
    3: (80,  160, 230),  # Water Body  – blue
    4: (50,   50, 220),  # Bridge      – red (BGR)
    5: (200,  50, 150),  # Railway     – purple
    6: (180, 200,  50),  # Utility     – teal
}

# ── loss weights  (Back  Bup   Road  Watr  Brid   Rail  Util) ─────────────
# Must have exactly NUM_CLASSES=7 entries.
CLASS_WEIGHTS    = [0.2,  0.8,  1.0,  0.8,  8.0,  10.0,  6.0]

# ── sampler weights ────────────────────────────────────────────────────────
SAMPLER_CLASS_WEIGHTS = {
    0: 0.1,    # background  – context only
    1: 0.5,    # built-up    – context
    2: 0.8,    # road        – easy to confuse with railway
    3: 0.5,    # water       – context
    4: 8.0,    # bridge      ← primary target
    5: 10.0,   # railway     ← primary target (was IoU = 0.000)
    6: 6.0,    # utility     ← primary target
}

# ── specialist inference ───────────────────────────────────────────────────
MINOR_CLASSES        = [4, 5, 6]   # Bridge, Railway, Utility
SPECIALIST_THRESHOLD = 0.45        # global default; tune per-class if needed
BRIDGE_THRESHOLD     = 0.45
RAILWAY_THRESHOLD    = 0.35        # lower: railway IoU was 0, be aggressive
UTILITY_THRESHOLD    = 0.45

CLASS_THRESHOLDS = {
    4: BRIDGE_THRESHOLD,
    5: RAILWAY_THRESHOLD,
    6: UTILITY_THRESHOLD,
}

# ── NOTE ───────────────────────────────────────────────────────────────────
# To actually label Bridge / Railway / Utility pixels during preprocessing,
# add entries to SHAPEFILE_MAP in this file (overriding the base config):
#
SHAPEFILE_MAP = {
      "builtup":   "Built_Up_Area_type.shp",
      "road":      "Road.shp",
      "waterbody": "Water_Body.shp",
      "bridge":    "Bridge.shp",      #← add your shapefiles
      "railway":   "Railway.shp",
      "utility":   "Utility.shp",
  }