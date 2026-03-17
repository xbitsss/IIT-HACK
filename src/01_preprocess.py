"""
01_preprocess.py — Windowed preprocessing for large GeoTIFFs.

Key additions vs original:
  • Coverage mask: tiles where NO shapefile polygon exists at all are skipped.
    This handles TIFFs that only have partial polygon annotation coverage —
    unannotated regions would otherwise create spurious all-background tiles
    that confuse the model about what "background" really means.
  • Per-tile class statistics saved to metadata for class-balanced sampling.
  • Shapefiles are loaded once from SHP_DIR and reprojected to each TIFF CRS.
"""

import os
import sys
import json
import shutil
import traceback
import numpy as np
import rasterio
import rasterio.windows
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, SHP_DIR, CLASSES, SHAPEFILE_MAP,
    CLASS_PRIORITY, TILE_SIZE, TILE_OVERLAP, BAND_INDICES,
    MIN_VALID_RATIO, MIN_COVERAGE_RATIO, RANDOM_SEED,
)

CHUNK_SIZE = 8192   # pixels processed per chunk pass


# ── Shapefile helpers ─────────────────────────────────────────────────────────

def find_shapefiles(shp_dir: Path) -> dict:
    shp_map = {}
    for class_name, filename in SHAPEFILE_MAP.items():
        shp_path = shp_dir / filename
        if shp_path.exists():
            shp_map[class_name] = shp_path
            print(f"  ✓ {class_name} → {shp_path}", flush=True)
        else:
            print(f"  ✗ Not found: {shp_path}", flush=True)
    return shp_map


def load_gdfs(shp_map: dict, tif_crs) -> dict:
    """Load & reproject all shapefiles to the TIFF CRS."""
    gdfs = {}
    for class_name, shp_path in shp_map.items():
        gdf = gpd.read_file(shp_path)
        if gdf.crs is not None and gdf.crs != tif_crs:
            gdf = gdf.to_crs(tif_crs)
        gdfs[class_name] = gdf
        print(f"    {class_name}: {len(gdf)} features", flush=True)
    return gdfs


# ── Rasterisation ─────────────────────────────────────────────────────────────

def rasterize_labels(gdfs: dict, win_transform, chunk_h: int, chunk_w: int) -> np.ndarray:
    """
    Rasterise class labels into a (chunk_h, chunk_w) uint8 mask.
    Higher-priority classes overwrite lower-priority ones.
    """
    label = np.zeros((chunk_h, chunk_w), dtype=np.uint8)
    for class_name in sorted(gdfs.keys(), key=lambda c: CLASS_PRIORITY.get(c, 0)):
        gdf = gdfs[class_name]
        if gdf.empty:
            continue
        shapes = [(g.__geo_interface__, 1) for g in gdf.geometry if g is not None]
        if not shapes:
            continue
        binary = rasterize(
            shapes=shapes,
            out_shape=(chunk_h, chunk_w),
            transform=win_transform,
            fill=0, dtype=np.uint8, all_touched=True,
        )
        label[binary == 1] = CLASSES[class_name]
    return label


def rasterize_coverage(gdfs: dict, win_transform, chunk_h: int, chunk_w: int) -> np.ndarray:
    """
    Build a binary (chunk_h, chunk_w) mask that is 1 wherever ANY polygon
    from ANY class exists.  Used to exclude tiles with zero annotation coverage.
    """
    coverage = np.zeros((chunk_h, chunk_w), dtype=np.uint8)
    for gdf in gdfs.values():
        if gdf.empty:
            continue
        shapes = [(g.__geo_interface__, 1) for g in gdf.geometry if g is not None]
        if not shapes:
            continue
        binary = rasterize(
            shapes=shapes,
            out_shape=(chunk_h, chunk_w),
            transform=win_transform,
            fill=0, dtype=np.uint8, all_touched=True,
        )
        coverage = np.maximum(coverage, binary)
    return coverage


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize(bands: np.ndarray) -> np.ndarray:
    bands = bands.astype(np.float32)
    for i in range(bands.shape[0]):
        b = bands[i]
        mn, mx = b.min(), b.max()
        bands[i] = (b - mn) / (mx - mn) if mx > mn else 0.0
    return bands


# ── Per-TIFF processing ───────────────────────────────────────────────────────

def process_tif(tif_path: Path, shp_map: dict, proc_dir: Path):
    stride    = TILE_SIZE - TILE_OVERLAP
    all_meta  = []
    tif_tiles = 0
    skipped_valid   = 0   # failed MIN_VALID_RATIO
    skipped_coverage = 0  # failed MIN_COVERAGE_RATIO

    with rasterio.open(tif_path) as tif:
        W, H = tif.width, tif.height
        print(f"  {W}×{H}, {tif.count} bands, CRS={tif.crs}", flush=True)
        print(f"  Loading & reprojecting shapefiles...", flush=True)
        gdfs = load_gdfs(shp_map, tif.crs)

        col_starts = list(range(0, W - TILE_SIZE + 1, CHUNK_SIZE))
        row_starts = list(range(0, H - TILE_SIZE + 1, CHUNK_SIZE))
        n_chunks   = len(row_starts) * len(col_starts)
        stem       = tif_path.stem
        print(f"  {n_chunks} chunks to process...", flush=True)

        chunk_num = 0
        for row_off in row_starts:
            for col_off in col_starts:
                chunk_h = min(CHUNK_SIZE + TILE_SIZE, H - row_off)
                chunk_w = min(CHUNK_SIZE + TILE_SIZE, W - col_off)
                if chunk_h < TILE_SIZE or chunk_w < TILE_SIZE:
                    continue

                chunk_num += 1
                print(f"  Chunk {chunk_num}/{n_chunks} "
                      f"row={row_off} col={col_off} "
                      f"({chunk_h}×{chunk_w})", flush=True)

                try:
                    window        = Window(col_off, row_off, chunk_w, chunk_h)
                    win_transform = rasterio.windows.transform(window, tif.transform)

                    bands    = tif.read(BAND_INDICES, window=window)
                    valid    = ~np.all(bands == 0, axis=0)   # non-nodata mask
                    label    = rasterize_labels(gdfs, win_transform, chunk_h, chunk_w)
                    coverage = rasterize_coverage(gdfs, win_transform, chunk_h, chunk_w)
                    bands    = normalize(bands)

                    chunk_tiles = 0
                    for r in range(0, chunk_h - TILE_SIZE + 1, stride):
                        for c in range(0, chunk_w - TILE_SIZE + 1, stride):
                            tile_valid    = valid[r:r+TILE_SIZE, c:c+TILE_SIZE]
                            tile_coverage = coverage[r:r+TILE_SIZE, c:c+TILE_SIZE]

                            # ── Filter 1: enough non-nodata pixels ────────────
                            if tile_valid.mean() < MIN_VALID_RATIO:
                                skipped_valid += 1
                                continue

                            # ── Filter 2: tile must intersect annotated region ─
                            # Tiles completely outside all shapefile extents have
                            # no ground-truth annotation — skip them to avoid
                            # teaching the model that "unannotated = background".
                            if MIN_COVERAGE_RATIO > 0 and tile_coverage.mean() < MIN_COVERAGE_RATIO:
                                skipped_coverage += 1
                                continue

                            # ── Save tile ─────────────────────────────────────
                            gr  = row_off + r
                            gc  = col_off + c
                            tid = f"{stem}_r{gr:06d}_c{gc:06d}"

                            tile_img  = bands[:, r:r+TILE_SIZE, c:c+TILE_SIZE].astype(np.float32)
                            tile_mask = label[r:r+TILE_SIZE, c:c+TILE_SIZE].astype(np.uint8)

                            np.save(proc_dir / "images" / f"{tid}.npy", tile_img)
                            np.save(proc_dir / "masks"  / f"{tid}.npy", tile_mask)

                            # Collect class IDs present in tile (for weighted sampler)
                            class_ids = [int(u) for u in np.unique(tile_mask)]

                            all_meta.append({
                                "id":               tid,
                                "source":           tif_path.name,
                                "row":              gr,
                                "col":              gc,
                                "coverage_ratio":   float(tile_coverage.mean()),
                                "class_ids":        class_ids,
                            })
                            chunk_tiles += 1
                            tif_tiles   += 1

                    print(f"    → {chunk_tiles} tiles saved  "
                          f"(total: {tif_tiles}, "
                          f"skipped valid={skipped_valid} "
                          f"coverage={skipped_coverage})", flush=True)

                except Exception as e:
                    print(f"  [WARN] Chunk {chunk_num} failed: {e}", flush=True)
                    traceback.print_exc()
                    continue

    print(f"  Skipped: {skipped_valid} (low pixel validity) + "
          f"{skipped_coverage} (no annotation coverage)", flush=True)
    return tif_tiles, all_meta


# ── Entry point ───────────────────────────────────────────────────────────────

def preprocess():
    raw_dir  = Path(DATA_RAW_DIR)
    shp_dir  = Path(SHP_DIR)
    proc_dir = Path(DATA_PROCESSED_DIR)

    # Clear previous processed data
    print(f"Clearing previous processed data at {proc_dir}...", flush=True)
    for subdir in ["images", "masks"]:
        d = proc_dir / subdir
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    meta_path = proc_dir / "tiles_meta.json"
    if meta_path.exists():
        meta_path.unlink()
    print("✓ Output dirs ready", flush=True)

    tif_files = list(raw_dir.glob("**/*.tif")) + list(raw_dir.glob("**/*.tiff"))
    if not tif_files:
        print(f"[ERROR] No .tif files in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(tif_files)} TIFF(s) in {raw_dir}", flush=True)
    print(f"Shapefiles from: {shp_dir}", flush=True)
    shp_map = find_shapefiles(shp_dir)
    if not shp_map:
        print("[ERROR] No shapefiles found — check SHP_DIR and SHAPEFILE_MAP in config.py")
        sys.exit(1)
    print(f"  Using classes: {list(shp_map.keys())}\n", flush=True)

    total_tiles = 0
    all_meta    = []

    for tif_path in tif_files:
        print(f"\n{'='*50}", flush=True)
        print(f"Processing: {tif_path.name}", flush=True)
        try:
            tile_count, meta = process_tif(tif_path, shp_map, proc_dir)
            total_tiles += tile_count
            all_meta    += meta
            print(f"  → {tile_count} tiles  (running total: {total_tiles})", flush=True)
        except Exception as e:
            print(f"[ERROR] {tif_path.name}: {e}", flush=True)
            traceback.print_exc()
            continue

    # ── Shuffle metadata so tiles are not in spatial scan order ──────────────
    # Tiles are produced row-by-row across the TIFF, so without shuffling the
    # metadata list is spatially sorted.  The DataLoader shuffle handles this
    # at training time, but shuffling here means:
    #   • The train/val split for the single-TIFF case draws from the full
    #     spatial extent rather than a contiguous geographic block.
    #   • The replay buffer selection samples evenly across the image rather
    #     than favouring one spatial corner.
    #   • Anything reading metadata sequentially (debugging, inspection) gets
    #     a representative sample rather than one corner of the image.
    import random as _random
    rng = _random.Random(RANDOM_SEED)
    rng.shuffle(all_meta)
    print(f"  Shuffled {len(all_meta)} tile metadata entries (seed={RANDOM_SEED})", flush=True)

    # ── Count class distribution across all tiles ─────────────────────────────
    from collections import Counter
    class_tile_counts = Counter()
    for m in all_meta:
        for cid in m["class_ids"]:
            class_tile_counts[cid] += 1

    print("\n── Class tile distribution ──────────────────────────────", flush=True)
    from config import CLASS_LABELS
    for cid, cnt in sorted(class_tile_counts.items()):
        label = CLASS_LABELS[cid] if cid < len(CLASS_LABELS) else f"class_{cid}"
        print(f"  {label}: {cnt} tiles ({cnt/max(total_tiles,1)*100:.1f}%)", flush=True)

    with open(proc_dir / "tiles_meta.json", "w") as f:
        json.dump({
            "tiles":              all_meta,
            "num_bands":          len(BAND_INDICES),
            "tile_size":          TILE_SIZE,
            "class_tile_counts":  dict(class_tile_counts),
        }, f, indent=2)

    print(f"\n  Done. {total_tiles} total tiles saved to {proc_dir}", flush=True)


if __name__ == "__main__":
    preprocess()