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

# ── loss weights  (Back  Bup   Road  Watr  Brid   Rail  Util) ─────────────
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
SPECIALIST_THRESHOLD = 0.45        # global default; tune per-class if needed:
BRIDGE_THRESHOLD     = 0.45
RAILWAY_THRESHOLD    = 0.35        # lower: railway IoU was 0, be aggressive
UTILITY_THRESHOLD    = 0.45

CLASS_THRESHOLDS = {
    4: BRIDGE_THRESHOLD,
    5: RAILWAY_THRESHOLD,
    6: UTILITY_THRESHOLD,
}
