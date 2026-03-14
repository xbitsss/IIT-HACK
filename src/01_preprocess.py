"""
01_preprocess.py — Single-core windowed preprocessing, chunk-by-chunk.
Processes and saves each chunk immediately — no memory accumulation.
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
    DATA_RAW_DIR, DATA_PROCESSED_DIR, CLASSES, SHAPEFILE_MAP, CLASS_PRIORITY,
    TILE_SIZE, TILE_OVERLAP, BAND_INDICES, MIN_VALID_RATIO
)

CHUNK_SIZE = 8192


def find_shapefiles(raw_dir):
    shp_map = {}
    for class_name, filename in SHAPEFILE_MAP.items():
        shp_path = raw_dir / filename
        if shp_path.exists():
            shp_map[class_name] = shp_path
            print(f"  ✓ {class_name} → {filename}", flush=True)
        else:
            print(f"  ✗ Not found: {filename}", flush=True)
    return shp_map


def load_gdfs(shp_map, tif_crs):
    gdfs = {}
    for class_name, shp_path in shp_map.items():
        gdf = gpd.read_file(shp_path)
        if gdf.crs is not None and gdf.crs != tif_crs:
            gdf = gdf.to_crs(tif_crs)
        gdfs[class_name] = gdf
        print(f"    {class_name}: {len(gdf)} features", flush=True)
    return gdfs


def rasterize_chunk(gdfs, win_transform, chunk_h, chunk_w):
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


def normalize(bands):
    bands = bands.astype(np.float32)
    for i in range(bands.shape[0]):
        b = bands[i]
        mn, mx = b.min(), b.max()
        bands[i] = (b - mn) / (mx - mn) if mx > mn else 0.0
    return bands


def process_tif(tif_path, shp_map, proc_dir):
    stride    = TILE_SIZE - TILE_OVERLAP
    all_meta  = []
    tif_tiles = 0

    with rasterio.open(tif_path) as tif:
        W, H = tif.width, tif.height
        print(f"  {W}×{H}, {tif.count} bands, CRS={tif.crs}", flush=True)
        print(f"  Loading shapefiles...", flush=True)
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
                print(f"  Chunk {chunk_num}/{n_chunks} row={row_off} col={col_off}", flush=True)

                try:
                    window        = Window(col_off, row_off, chunk_w, chunk_h)
                    win_transform = rasterio.windows.transform(window, tif.transform)
                    bands = tif.read(BAND_INDICES, window=window)
                    valid = ~np.all(bands == 0, axis=0)
                    label = rasterize_chunk(gdfs, win_transform, chunk_h, chunk_w)
                    bands = normalize(bands)

                    # Save tiles immediately
                    chunk_tiles = 0
                    for r in range(0, chunk_h - TILE_SIZE + 1, stride):
                        for c in range(0, chunk_w - TILE_SIZE + 1, stride):
                            if valid[r:r+TILE_SIZE, c:c+TILE_SIZE].mean() < MIN_VALID_RATIO:
                                continue
                            gr  = row_off + r
                            gc  = col_off + c
                            tid = f"{stem}_r{gr:06d}_c{gc:06d}"
                            np.save(proc_dir / "images" / f"{tid}.npy",
                                    bands[:, r:r+TILE_SIZE, c:c+TILE_SIZE].astype(np.float32))
                            np.save(proc_dir / "masks" / f"{tid}.npy",
                                    label[r:r+TILE_SIZE, c:c+TILE_SIZE].astype(np.uint8))
                            all_meta.append({"id": tid, "source": tif_path.name,
                                             "row": gr, "col": gc})
                            chunk_tiles += 1
                            tif_tiles   += 1

                    print(f"    → {chunk_tiles} tiles (total: {tif_tiles})", flush=True)

                except Exception as e:
                    print(f"  [WARN] Chunk {chunk_num} failed: {e}", flush=True)
                    traceback.print_exc()
                    continue

    return tif_tiles, all_meta


def preprocess():
    raw_dir  = Path(DATA_RAW_DIR)
    proc_dir = Path(DATA_PROCESSED_DIR)

    # Clear previous processed data
    print(f"Clearing previous processed data...", flush=True)
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

    print(f"Found {len(tif_files)} TIFF(s)", flush=True)
    print("Locating shapefiles...", flush=True)
    shp_map = find_shapefiles(raw_dir)
    print(f"  Using: {list(shp_map.keys())}\n", flush=True)

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

    with open(proc_dir / "tiles_meta.json", "w") as f:
        json.dump({
            "tiles":     all_meta,
            "num_bands": len(BAND_INDICES),
            "tile_size": TILE_SIZE
        }, f, indent=2)

    print(f"\n✓ Done. {total_tiles} total tiles saved to {proc_dir}", flush=True)


if __name__ == "__main__":
    preprocess()