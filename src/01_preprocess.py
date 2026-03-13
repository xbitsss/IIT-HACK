"""
01_preprocess.py — Rasterizes shapefiles and tiles TIFFs into training patches.
"""

import os
import sys
import json
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, CLASSES, SHAPEFILE_MAP, CLASS_PRIORITY,
    TILE_SIZE, TILE_OVERLAP, BAND_INDICES, MIN_VALID_RATIO, RANDOM_SEED
)


def find_shapefiles(raw_dir: Path) -> dict:
    shp_map = {}
    for class_name, filename in SHAPEFILE_MAP.items():
        shp_path = raw_dir / filename
        if shp_path.exists():
            shp_map[class_name] = shp_path
            print(f"  ✓ {class_name:12s} → {filename}")
        else:
            print(f"  ✗ [WARN] Not found: {filename}")
    return shp_map


def rasterize_shapefile(shp_path: Path, ref_tif: rasterio.DatasetReader) -> np.ndarray:
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        print(f"    [WARN] {shp_path.name} has no CRS")
    elif gdf.crs != ref_tif.crs:
        gdf = gdf.to_crs(ref_tif.crs)
    if gdf.empty:
        return np.zeros((ref_tif.height, ref_tif.width), dtype=np.uint8)
    shapes = [(geom.__geo_interface__, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        return np.zeros((ref_tif.height, ref_tif.width), dtype=np.uint8)
    return rasterize(
        shapes=shapes,
        out_shape=(ref_tif.height, ref_tif.width),
        transform=ref_tif.transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )


def build_label_mask(shp_map: dict, tif: rasterio.DatasetReader) -> np.ndarray:
    H, W  = tif.height, tif.width
    label = np.zeros((H, W), dtype=np.uint8)
    sorted_classes = sorted(shp_map.keys(), key=lambda c: CLASS_PRIORITY.get(c, 0))
    for class_name in sorted_classes:
        class_id = CLASSES[class_name]
        binary   = rasterize_shapefile(shp_map[class_name], tif)
        label[binary == 1] = class_id
        print(f"    {class_name}: {int(binary.sum()):,} pixels")
    return label


def normalize_bands(bands: np.ndarray) -> np.ndarray:
    bands = bands.astype(np.float32)
    for i in range(bands.shape[0]):
        b = bands[i]
        bmin, bmax = b.min(), b.max()
        bands[i] = (b - bmin) / (bmax - bmin) if bmax > bmin else 0.0
    return bands


def tile_image_and_mask(image, mask, valid, tile_size, overlap):
    C, H, W = image.shape
    stride  = tile_size - overlap
    for row in range(0, H - tile_size + 1, stride):
        for col in range(0, W - tile_size + 1, stride):
            if valid[row:row+tile_size, col:col+tile_size].mean() < MIN_VALID_RATIO:
                continue
            yield image[:, row:row+tile_size, col:col+tile_size], mask[row:row+tile_size, col:col+tile_size], row, col


def preprocess():
    raw_dir  = Path(DATA_RAW_DIR)
    proc_dir = Path(DATA_PROCESSED_DIR)
    (proc_dir / "images").mkdir(parents=True, exist_ok=True)
    (proc_dir / "masks").mkdir(parents=True, exist_ok=True)

    tif_files = list(raw_dir.glob("**/*.tif")) + list(raw_dir.glob("**/*.tiff"))
    if not tif_files:
        print(f"[ERROR] No .tif files found in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(tif_files)} TIFF file(s)")
    print("Locating shapefiles...")
    shp_map = find_shapefiles(raw_dir)
    print(f"  Using: {list(shp_map.keys())}\n")

    tile_count = 0
    meta       = []

    for tif_path in tif_files:
        print(f"Processing: {tif_path.name}")
        with rasterio.open(tif_path) as tif:
            print(f"  Size: {tif.width}x{tif.height}, Bands: {tif.count}")
            bands = tif.read(BAND_INDICES)
            valid = ~np.all(bands == 0, axis=0)
            print(f"  Building label mask...")
            label = build_label_mask(shp_map, tif)
            unique, counts = np.unique(label, return_counts=True)
            dist = {int(u): f"{int(c)/label.size*100:.1f}%" for u, c in zip(unique, counts)}
            print(f"  Distribution: {dist}")
            bands = normalize_bands(bands)
            stem  = tif_path.stem
            tiles = list(tile_image_and_mask(bands, label, valid, TILE_SIZE, TILE_OVERLAP))
            print(f"  {len(tiles)} tiles — saving...")
            for img_tile, msk_tile, row, col in tqdm(tiles, desc="  Tiles"):
                tid = f"{stem}_r{row:05d}_c{col:05d}"
                np.save(proc_dir / "images" / f"{tid}.npy", img_tile.astype(np.float32))
                np.save(proc_dir / "masks"  / f"{tid}.npy", msk_tile.astype(np.uint8))
                meta.append({"id": tid, "source": tif_path.name, "row": row, "col": col})
                tile_count += 1
        print(f"  Running total: {tile_count} tiles\n")

    with open(proc_dir / "tiles_meta.json", "w") as f:
        json.dump({"tiles": meta, "num_bands": len(BAND_INDICES), "tile_size": TILE_SIZE}, f, indent=2)

    print(f"✓ Done. {tile_count} tiles saved.")


if __name__ == "__main__":
    preprocess()