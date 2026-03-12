"""
config.py — All pipeline settings in one place.
Edit this file to match your data before running anything.
"""

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_RAW_DIR       = "data/raw"          # folder with your .tif + .shp files
DATA_PROCESSED_DIR = "data/processed"   # auto-created tiles + masks
CHECKPOINT_DIR     = "checkpoints"
OUTPUT_DIR         = "outputs"

# ─── Class Definitions ────────────────────────────────────────────────────────
# Map each shapefile name (without extension) to a class ID.
# Adjust keys to match YOUR actual shapefile filenames.
CLASSES = {
    "background": 0,
    "builtup":    1,
    "road":       2,
    "waterbody":  3,
}
NUM_CLASSES = len(CLASSES)

# Human-readable labels (for visualization)
CLASS_LABELS = ["Background", "Built-up", "Road", "Waterbody"]

# Visualization colors per class (BGR for OpenCV)
CLASS_COLORS = {
    0: (50,  50,  50),    # background — dark grey
    1: (180, 100, 60),    # built-up   — blue-grey
    2: (60,  180, 255),   # road       — yellow
    3: (255, 120, 50),    # waterbody  — blue
}

# ─── Band Configuration ───────────────────────────────────────────────────────
# Set to None to auto-detect from TIFF.
# Set to e.g. [0, 1, 2] to use only first 3 bands (RGB).
# Set to [0, 1, 2, 3] for RGB+NIR.
BAND_INDICES = None   # None = use all bands

# ─── Tiling ───────────────────────────────────────────────────────────────────
TILE_SIZE    = 512    # pixels — reduce to 256 if you run out of GPU memory
TILE_OVERLAP = 64     # overlap between tiles (helps avoid edge artifacts)
MIN_VALID_RATIO = 0.1 # skip tiles where >90% pixels are NoData

# ─── Training ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "nvidia/mit-b2"   # SegFormer backbone. Options: mit-b0 (fast) → mit-b5 (accurate)
BATCH_SIZE   = 8
NUM_EPOCHS   = 50
LR           = 6e-5
WEIGHT_DECAY = 0.01
VAL_SPLIT    = 0.2      # fraction of tiles held out for validation
RANDOM_SEED  = 42

# Loss weights — increase weight for underrepresented classes (roads are thin!)
CLASS_WEIGHTS = [0.5, 1.0, 3.0, 1.5]   # [background, builtup, road, waterbody]

# Early stopping
PATIENCE = 10   # stop if val loss doesn't improve for this many epochs

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUGMENT_TRAIN = True

# ─── Inference ────────────────────────────────────────────────────────────────
INFERENCE_OVERLAP = 128    # larger overlap = smoother stitching, slower
INFERENCE_BATCH   = 4
