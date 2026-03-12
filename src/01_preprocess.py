"""
01_preprocess.py
────────────────
Converts raw GeoTIFF + Shapefile pairs into fixed-size tiles ready for training.

What it does:
  1. Discovers all .tif files in DATA_RAW_DIR
  2. For each TIFF, finds matching shapefiles by class name
  3. Rasterizes each shapefile → per-class binary mask
  4. Merges class masks into a single multi-class label mask
  5. Tiles both image and mask into TILE_SIZE patches
  6. Saves tiles to DATA_PROCESSED_DIR/images/ and /masks/

Expected folder layout in data/raw/:
  data/raw/
  ├── area1.tif
  ├── builtup.shp   (+ .dbf, .prj, .shx)
  ├── road.shp
  ├── waterbody.shp
  └── background.shp   ← optional; pixels not covered by any class default to 0

  Multiple TIFF regions are supported — just add more .tif files.
  Each TIFF will use the SAME shapefiles (if they cover that region).
  If you have per-region shapefiles, use subfolders (see multi-region note below).
"""

import os
import sys
import json
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from pathlib import Path
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, CLASSES,
    TILE_SIZE, TILE_OVERLAP, BAND_INDICES, MIN_VALID_RATIO, RANDOM_SEED
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def find_shapefiles(raw_dir: Path) -> dict[str, Path]:
    """Find shapefiles matching class names (case-insensitive)."""
    shp_map = {}
    for class_name in CLASSES:
        if class_name == "background":
            continue  # background is the default; no shapefile needed
        matches = list(raw_dir.glob(f"**/*{class_name}*.shp"))
        if not matches:
            print(f"  [WARN] No shapefile found for class '{class_name}' — pixels not covered will be background")
        else:
            shp_map[class_name] = matches[0]
            if len(matches) > 1:
                print(f"  [WARN] Multiple matches for '{class_name}', using {matches[0]}")
    return shp_map


def rasterize_shapefile(shp_path: Path, ref_tif: rasterio.DatasetReader) -> np.ndarray:
    """
    Burns shapefile polygons into a binary mask aligned to ref_tif's grid.
    Returns uint8 array of shape (H, W): 1 inside polygons, 0 outside.
    """
    gdf = gpd.read_file(shp_path)

    # Reproject to match the TIFF's CRS if needed
    if gdf.crs is None:
        print(f"    [WARN] {shp_path.name} has no CRS — assuming it matches the TIFF")
    elif gdf.crs != ref_tif.crs:
        gdf = gdf.to_crs(ref_tif.crs)

    if gdf.empty:
        return np.zeros((ref_tif.height, ref_tif.width), dtype=np.uint8)

    shapes = [(geom.__geo_interface__, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        return np.zeros((ref_tif.height, ref_tif.width), dtype=np.uint8)

    mask = rasterize(
        shapes=shapes,
        out_shape=(ref_tif.height, ref_tif.width),
        transform=ref_tif.transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,   # important for thin roads — catches edge pixels
    )
    return mask


def build_label_mask(shp_map: dict, tif: rasterio.DatasetReader) -> np.ndarray:
    """
    Builds a single H×W label mask where each pixel = class ID.
    Later classes overwrite earlier ones if they overlap (road > builtup > background).
    """
    H, W = tif.height, tif.width
    label = np.zeros((H, W), dtype=np.uint8)   # 0 = background

    # Paint in priority order: builtup first, road last (roads should win)
    priority_order = ["builtup", "waterbody", "road"]
    for class_name in priority_order:
        if class_name not in shp_map:
            continue
        class_id = CLASSES[class_name]
        binary = rasterize_shapefile(shp_map[class_name], tif)
        label[binary == 1] = class_id

    return label


def get_nodata_mask(tif: rasterio.DatasetReader, bands: np.ndarray) -> np.ndarray:
    """Returns True where pixel is valid (not NoData)."""
    nodata = tif.nodata
    if nodata is not None:
        valid = ~np.all(bands == nodata, axis=0)
    else:
        valid = ~np.all(bands == 0, axis=0)
    return valid


def tile_image_and_mask(image: np.ndarray, mask: np.ndarray, valid: np.ndarray,
                         tile_size: int, overlap: int):
    """
    Yields (tile_image, tile_mask, row, col) for each tile.
    image: (C, H, W), mask: (H, W), valid: (H, W)
    """
    C, H, W = image.shape
    stride = tile_size - overlap

    for row in range(0, H - tile_size + 1, stride):
        for col in range(0, W - tile_size + 1, stride):
            img_tile  = image[:, row:row+tile_size, col:col+tile_size]
            mask_tile = mask[row:row+tile_size, col:col+tile_size]
            valid_tile = valid[row:row+tile_size, col:col+tile_size]

            valid_ratio = valid_tile.mean()
            if valid_ratio < MIN_VALID_RATIO:
                continue   # skip mostly-nodata tiles

            yield img_tile, mask_tile, row, col


def normalize_bands(bands: np.ndarray) -> np.ndarray:
    """
    Per-band min-max normalization → float32 [0, 1].
    Handles uint8, uint16, float32 inputs.
    """
    bands = bands.astype(np.float32)
    for i in range(bands.shape[0]):
        b = bands[i]
        bmin, bmax = b.min(), b.max()
        if bmax > bmin:
            bands[i] = (b - bmin) / (bmax - bmin)
        else:
            bands[i] = 0.0
    return bands


# ── Main ─────────────────────────────────────────────────────────────────────

def preprocess():
    raw_dir  = Path(DATA_RAW_DIR)
    proc_dir = Path(DATA_PROCESSED_DIR)

    img_dir  = proc_dir / "images"
    msk_dir  = proc_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    tif_files = list(raw_dir.glob("**/*.tif")) + list(raw_dir.glob("**/*.tiff"))
    if not tif_files:
        print(f"[ERROR] No .tif files found in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(tif_files)} TIFF file(s)")

    # Find shapefiles (shared across all TIFFs — adjust if per-region)
    print("Locating shapefiles...")
    shp_map = find_shapefiles(raw_dir)
    print(f"  Loaded: {list(shp_map.keys())}")

    tile_count = 0
    meta = []   # store tile metadata for dataset

    for tif_path in tif_files:
        print(f"\nProcessing: {tif_path.name}")
        with rasterio.open(tif_path) as tif:
            print(f"  Size: {tif.width}×{tif.height}, Bands: {tif.count}, CRS: {tif.crs}")

            # Read bands
            band_idx = BAND_INDICES if BAND_INDICES else list(range(1, tif.count + 1))
            bands = tif.read(band_idx)   # (C, H, W)
            print(f"  Using {len(band_idx)} band(s): {band_idx}")

            # Build masks
            valid_mask  = get_nodata_mask(tif, bands)
            label_mask  = build_label_mask(shp_map, tif)

            unique, counts = np.unique(label_mask, return_counts=True)
            print(f"  Label distribution: { {int(u): int(c) for u, c in zip(unique, counts)} }")

            # Normalize
            bands = normalize_bands(bands)

            # Tile
            stem = tif_path.stem
            tiles = list(tile_image_and_mask(bands, label_mask, valid_mask, TILE_SIZE, TILE_OVERLAP))
            print(f"  Generating {len(tiles)} tiles...")

            for img_tile, msk_tile, row, col in tqdm(tiles, desc="  Tiling"):
                tile_id = f"{stem}_r{row:05d}_c{col:05d}"
                np.save(img_dir / f"{tile_id}.npy", img_tile.astype(np.float32))
                np.save(msk_dir / f"{tile_id}.npy", msk_tile.astype(np.uint8))
                meta.append({"id": tile_id, "source": tif_path.name, "row": row, "col": col})
                tile_count += 1

    # Save metadata
    meta_path = proc_dir / "tiles_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"tiles": meta, "num_bands": len(band_idx), "tile_size": TILE_SIZE}, f, indent=2)

    print(f"\n✓ Done. {tile_count} tiles saved to {proc_dir}")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    preprocess()
