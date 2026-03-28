"""
Build tiles_meta_specialist.json
  50 % minor-class tiles  (Bridge=4, Railway=5, Utility=6)
  50 % major-class tiles  (Built-up=1, Road=2, Water=3)
       — pure-background-only tiles are excluded from the major pool

The major pool gives the specialist context: it must learn
"this is a road / building, NOT a railway / bridge".

Fallback (specialist-only mode):
  If tiles_meta.json has been wiped (e.g. running --specialist-only after
  generalist training is complete), the script reads tile metadata from
  every replay-buffer manifest instead.  The replay buffer is never wiped,
  so minor-class tiles saved during generalist shards will still be found.

Usage:
    python specialist/01_build_specialist_meta.py
"""

import json
import sys
import random
from collections import Counter
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Import from config_specialist so CLASS_LABELS has all 7 entries.
from config_specialist import DATA_PROCESSED_DIR, CLASS_LABELS, RANDOM_SEED
from config import REPLAY_DIR  # replay dir lives in the base config

MINOR_CLASSES = {4, 5, 6}
MAJOR_CLASSES = {1, 2, 3}

proc     = Path(DATA_PROCESSED_DIR)
src_meta = proc / "tiles_meta.json"
dst_meta = proc / "tiles_meta_specialist.json"

# ── Load tile list — current shard OR replay fallback ─────────────────────
def _load_from_replay() -> tuple[list, dict]:
    """
    Read all replay-buffer manifests and reconstruct a tile list.
    Returns (tiles, base_meta_dict) where base_meta_dict carries the
    fields that tiles_meta_specialist.json expects (num_bands, etc.).
    """
    replay_root = Path(REPLAY_DIR)
    if not replay_root.exists():
        return [], {}

    tiles    = []
    seen_ids = set()
    num_bands = 4  # default; overridden from first manifest that has it

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


if src_meta.exists():
    with open(src_meta) as f:
        meta = json.load(f)
    all_tiles = meta["tiles"]
    print(f"[INFO] Loaded {len(all_tiles)} tiles from {src_meta}")
else:
    print(f"[WARN] {src_meta} not found — falling back to replay buffer manifests.")
    print(f"       (This is expected in --specialist-only mode.)")
    all_tiles, meta = _load_from_replay()
    if not all_tiles:
        print("[ERROR] No tiles found in replay buffer either.")
        print("        Run generalist shards first, or ensure REPLAY_DIR is correct.")
        sys.exit(1)
    print(f"[INFO] Loaded {len(all_tiles)} tiles from replay buffer.")

# ── Partition into minor / major pools ────────────────────────────────────
minor_tiles = [
    t for t in all_tiles
    if MINOR_CLASSES & set(t.get("class_ids", []))
]

major_tiles = [
    t for t in all_tiles
    if (MAJOR_CLASSES & set(t.get("class_ids", [])))
    and not (MINOR_CLASSES & set(t.get("class_ids", [])))
]

rng = random.Random(RANDOM_SEED)
rng.shuffle(minor_tiles)
rng.shuffle(major_tiles)

n_minor = len(minor_tiles)
n_major = min(len(major_tiles), n_minor)   # strict 50 / 50

if n_minor == 0:
    print("[ERROR] Zero minor-class tiles found.")
    print("        Check that Bridge/Railway/Utility shapefiles were used in preprocessing,")
    print("        and that 'class_ids' fields in the tile metadata include values 4/5/6.")
    sys.exit(1)

if n_major < n_minor:
    ratio = n_major / (n_minor + n_major) * 100 if (n_minor + n_major) > 0 else 0
    print(f"[WARN] Only {n_major} major tiles available (wanted {n_minor}). "
          f"Dataset will be {n_minor + n_major} tiles ({ratio:.0f}% major).")

selected_major = major_tiles[:n_major]
combined       = minor_tiles + selected_major
rng.shuffle(combined)

# ── Per-class breakdown ────────────────────────────────────────────────────
counts = Counter(cid for t in combined for cid in t.get("class_ids", []))
print(f"\n{'─'*45}")
print(f"  Minor-class tiles : {n_minor:>6}")
print(f"  Major-class tiles : {n_major:>6}")
print(f"  Total specialist  : {len(combined):>6}")
print(f"{'─'*45}")
print("  Per-class tile counts in specialist set:")
max_count = max(counts.values(), default=1)
for cid in range(len(CLASS_LABELS)):
    bar = "█" * (counts.get(cid, 0) * 30 // max_count)
    print(f"    {CLASS_LABELS[cid]:12s}  {counts.get(cid, 0):5d}  {bar}")
print(f"{'─'*45}\n")

# ── Write output ──────────────────────────────────────────────────────────
proc.mkdir(parents=True, exist_ok=True)

out = {
    **meta,
    "tiles":          combined,
    "total_tiles":    len(combined),
    "specialist":     True,
    "minor_classes":  list(MINOR_CLASSES),
    "n_minor_tiles":  n_minor,
    "n_major_tiles":  n_major,
}
with open(dst_meta, "w") as f:
    json.dump(out, f, indent=2)

print(f"[OK] Written → {dst_meta}")