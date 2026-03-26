"""
Build tiles_meta_specialist.json
  50 % minor-class tiles  (Bridge=4, Railway=5, Utility=6)
  50 % major-class tiles  (Built-up=1, Road=2, Water=3)
       — pure-background-only tiles are excluded from the major pool

The major pool gives the specialist context: it must learn
"this is a road / building, NOT a railway / bridge".

Usage:
    python specialist/01_build_meta.py
"""

import json
import sys
import random
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import DATAPROCESSEDDIR, CLASSLABELS, RANDOMSEED

MINOR_CLASSES = {4, 5, 6}
MAJOR_CLASSES = {1, 2, 3}

proc     = Path(DATAPROCESSEDDIR)
src_meta = proc / "tiles_meta.json"
dst_meta = proc / "tiles_meta_specialist.json"

if not src_meta.exists():
    print(f"[ERROR] {src_meta} not found — run 01_preprocess.py first.")
    sys.exit(1)

with open(src_meta) as f:
    meta = json.load(f)

all_tiles = meta["tiles"]

# tiles that contain at least one minor class
minor_tiles = [
    t for t in all_tiles
    if MINOR_CLASSES & set(t.get("classids", []))
]

# tiles that contain at least one major class but NO minor class
major_tiles = [
    t for t in all_tiles
    if (MAJOR_CLASSES & set(t.get("classids", [])))
    and not (MINOR_CLASSES & set(t.get("classids", [])))
]

rng = random.Random(RANDOMSEED)
rng.shuffle(minor_tiles)
rng.shuffle(major_tiles)

n_minor = len(minor_tiles)
n_major = min(len(major_tiles), n_minor)   # strict 50 / 50

if n_minor == 0:
    print("[ERROR] Zero minor-class tiles found. Check classids in tiles_meta.json.")
    sys.exit(1)

if n_major < n_minor:
    print(f"[WARN] Only {n_major} major tiles available (wanted {n_minor}). "
          f"Dataset will be {n_minor + n_major} tiles ({n_major / (n_minor + n_major) * 100:.0f}% major).")

selected_major = major_tiles[:n_major]
combined       = minor_tiles + selected_major
rng.shuffle(combined)

# ── per-class breakdown ────────────────────────────────────────────────────
counts = Counter(cid for t in combined for cid in t.get("classids", []))
print(f"\n{'─'*45}")
print(f"  Minor-class tiles : {n_minor:>6}")
print(f"  Major-class tiles : {n_major:>6}")
print(f"  Total specialist  : {len(combined):>6}")
print(f"{'─'*45}")
print("  Per-class tile counts in specialist set:")
for cid in range(len(CLASSLABELS)):
    bar = "█" * (counts.get(cid, 0) * 30 // max(counts.values(), default=1))
    print(f"    {CLASSLABELS[cid]:12s}  {counts.get(cid, 0):5d}  {bar}")
print(f"{'─'*45}\n")

# ── write ──────────────────────────────────────────────────────────────────
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
