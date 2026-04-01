"""
Build tiles_meta_specialist.json
  50 % minor-class tiles  (Bridge=4, Railway=5, Utility=6)
  50 % major-class tiles  (Built-up=1, Road=2, Water=3)
       -- pure-background-only tiles are excluded from the major pool

The major pool gives the specialist context: it must learn
"this is a road / building, NOT a railway / bridge".

Fallback (specialist-only mode):
  If tiles_meta.json has been wiped (e.g. running --specialist-only after
  generalist training is complete), the script reads tile metadata from
  every replay-buffer manifest instead.  The replay buffer is never wiped,
  so minor-class tiles saved during generalist shards will still be found.

Exit codes:
  0 - success
  1 - no tiles found at all (nothing to work with)
  2 - tiles found but ZERO have class_ids 4/5/6 (needs re-preprocess)
      The shell script catches this specific code and triggers a bootstrap.

Usage:
    python specialist/01_build_specialist_meta.py
"""

import json
import sys
import random
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from config_specialist import DATA_PROCESSED_DIR, CLASS_LABELS, RANDOM_SEED
from config import REPLAY_DIR, SHAPEFILE_MAP as BASE_SHAPEFILE_MAP

MINOR_CLASSES = {4, 5, 6}
MAJOR_CLASSES = {1, 2, 3}

proc     = Path(DATA_PROCESSED_DIR)
src_meta = proc / "tiles_meta.json"
dst_meta = proc / "tiles_meta_specialist.json"


# ── Load tile list -- current shard OR replay fallback -------------------------
def _load_from_replay():
    replay_root = Path(REPLAY_DIR)
    if not replay_root.exists():
        return [], {}

    tiles    = []
    seen_ids = set()
    num_bands = 4

    for shard_dir in sorted(replay_root.iterdir()):
        manifest_path = shard_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        nb = manifest.get("num_bands", num_bands)
        if nb:
            num_bands = nb
        for tile in manifest.get("tiles", []):
            tid = tile["id"]
            if tid not in seen_ids:
                seen_ids.add(tid)
                tiles.append(tile)

    base_meta = {"num_bands": num_bands, "source": "replay_fallback"}
    return tiles, base_meta


def _check_shapefile_map():
    """Check whether config_specialist.py actually defines bridge/railway/utility."""
    try:
        from config_specialist import SHAPEFILE_MAP as SPEC_MAP
        minor_keys = {"bridge", "railway", "utility"}
        found = minor_keys & set(SPEC_MAP.keys())
        missing = minor_keys - set(SPEC_MAP.keys())
        return found, missing
    except ImportError:
        return set(), {"bridge", "railway", "utility"}


# ── Load tiles -----------------------------------------------------------------
if src_meta.exists():
    with open(src_meta) as f:
        meta = json.load(f)
    all_tiles = meta["tiles"]
    print(f"[INFO] Loaded {len(all_tiles)} tiles from {src_meta}")
else:
    print(f"[WARN] {src_meta} not found -- falling back to replay buffer manifests.")
    print(f"       (This is expected in --specialist-only mode.)")
    all_tiles, meta = _load_from_replay()
    if not all_tiles:
        print("[ERROR] No tiles found in replay buffer either.")
        print("        Run generalist shards first, or ensure REPLAY_DIR is correct.")
        sys.exit(1)
    print(f"[INFO] Loaded {len(all_tiles)} tiles from replay buffer.")

# ── Partition -----------------------------------------------------------------
minor_tiles = [t for t in all_tiles if MINOR_CLASSES & set(t.get("class_ids", []))]
major_tiles = [
    t for t in all_tiles
    if (MAJOR_CLASSES & set(t.get("class_ids", [])))
    and not (MINOR_CLASSES & set(t.get("class_ids", [])))
]

rng = random.Random(RANDOM_SEED)
rng.shuffle(minor_tiles)
rng.shuffle(major_tiles)

n_minor = len(minor_tiles)
n_major = min(len(major_tiles), n_minor)

# ── Guard: zero minor-class tiles ---------------------------------------------
if n_minor == 0:
    found_shp, missing_shp = _check_shapefile_map()

    print("")
    print("=" * 62)
    print("[ERROR] Zero minor-class tiles found (class_ids 4/5/6 not present).")
    print("=" * 62)
    print("")
    print("WHY THIS HAPPENS:")
    print("  The generalist preprocessing only rasterised classes 0-3")
    print("  (Background, Built-up, Road, Water). Bridge/Railway/Utility")
    print("  (classes 4/5/6) require their own shapefiles to be defined")
    print("  in config_specialist.py and then a specialist preprocess run.")
    print("")

    if missing_shp:
        print("STEP 1 -- Add shapefile entries to specialist/config_specialist.py:")
        print("")
        print("  SHAPEFILE_MAP = {")
        print('      "builtup":   "Built_Up_Area_type.shp",')
        print('      "road":      "Road.shp",')
        print('      "waterbody": "Water_Body.shp",')
        for key in sorted(missing_shp):
            print(f'      "{key}":    "YourActual{key.title()}.shp",  # <-- add this')
        print("  }")
        print("")
        print("  Replace 'YourActual*.shp' with the real filename from your SHP folder.")
        print("")
    else:
        print("STEP 1 -- config_specialist.py already has entries for:")
        for k in sorted(found_shp):
            print(f"  {k}")
        print("")

    print("STEP 2 -- Re-run specialist with bootstrap preprocessing:")
    print("")
    print("  docker compose run --rm specialist --data-dir /raw_data/ALL")
    print("")
    print("  This runs preprocessing with SPECIALIST_PREPROCESS=1 so that")
    print("  Bridge/Railway/Utility pixels are rasterised from your shapefiles.")
    print("")
    print("NOTE: You do NOT need to re-train the generalist. Only the specialist")
    print("      tile set needs to be regenerated.")
    print("=" * 62)
    print("")
    # Exit code 2 = tiles exist but no minor classes.
    # The shell script catches this to trigger automatic bootstrap.
    sys.exit(2)

if n_major < n_minor:
    ratio = n_major / (n_minor + n_major) * 100 if (n_minor + n_major) > 0 else 0
    print(f"[WARN] Only {n_major} major tiles available (wanted {n_minor}). "
          f"Dataset will be {n_minor + n_major} tiles ({ratio:.0f}% major).")

selected_major = major_tiles[:n_major]
combined       = minor_tiles + selected_major
rng.shuffle(combined)

# ── Per-class breakdown --------------------------------------------------------
counts = Counter(cid for t in combined for cid in t.get("class_ids", []))
print("")
print("-" * 45)
print(f"  Minor-class tiles : {n_minor:>6}")
print(f"  Major-class tiles : {n_major:>6}")
print(f"  Total specialist  : {len(combined):>6}")
print("-" * 45)
print("  Per-class tile counts in specialist set:")
max_count = max(counts.values(), default=1)
for cid in range(len(CLASS_LABELS)):
    bar = "#" * (counts.get(cid, 0) * 30 // max_count)
    print(f"    {CLASS_LABELS[cid]:12s}  {counts.get(cid, 0):5d}  {bar}")
print("-" * 45)
print("")

# ── Write output ---------------------------------------------------------------
proc.mkdir(parents=True, exist_ok=True)

out = {
    **meta,
    "tiles":         combined,
    "total_tiles":   len(combined),
    "specialist":    True,
    "minor_classes": list(MINOR_CLASSES),
    "n_minor_tiles": n_minor,
    "n_major_tiles": n_major,
}
with open(dst_meta, "w") as f:
    json.dump(out, f, indent=2)

print(f"[OK] Written -> {dst_meta}")