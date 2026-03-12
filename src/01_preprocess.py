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
    DATA_RAW_DIR, DATA_PROCESSED_DIR, CLASSES, SHAPEFILE_CLASS_MAP, CLASS_PRIORITY,
    TILE_SIZE, TILE_OVERLAP, BAND_INDICES, MIN_VALID_RATIO, RANDOM_SEED
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def find_shapefiles(raw_dir: Path) -> dict[str, list[Path]]:
    """
    Find shapefiles and map them to class IDs using SHAPEFILE_CLASS_MAP.
    A single class (e.g. 'water_body') can have MULTIPLE shapefiles
    (e.g. Water_Body polygon + Water_Body_Line + Waterbody_Point).
    Returns: { class_name: [shp_path1, shp_path2, ...] }
    """
    all_shps = list(raw_dir.glob("**/*.shp"))
    shp_map  = {}   # class_name → list of matching shp paths

    print(f"  Found {len(all_shps)} shapefile(s) in {raw_dir}")

    for shp_path in all_shps:
        stem_lower = shp_path.stem.lower().replace("-", "_").replace(" ", "_")
        matched = False
        # Check against every keyword in SHAPEFILE_CLASS_MAP
        # Use longest-match to avoid 'road' matching 'road_centre_line' wrong
        best_key, best_class = None, None
        for keyword, class_name in SHAPEFILE_CLASS_MAP.items():
            if keyword.lower() in stem_lower:
                if best_key is None or len(keyword) > len(best_key):
                    best_key   = keyword
                    best_class = class_name
                matched = True

        if best_class:
            shp_map.setdefault(best_class, []).append(shp_path)
            print(f"    {shp_path.name:40s} → class '{best_class}'")
        else:
            print(f"    {shp_path.name:40s} → [UNMATCHED — add to SHAPEFILE_CLASS_MAP in config.py]")

    # Report which classes have no shapefiles
    for class_name in CLASSES:
        if class_name == "background":
            continue
        if class_name not in shp_map:
            print(f"  [WARN] No shapefile matched class '{class_name}' — will default to background")

    return shp_map


def rasterize_shapefile(shp_path: Path, ref_tif: rasterio.DatasetReader) -> np.ndarray:
    """
    Burns shapefile polygons/lines/points into a binary mask aligned to ref_tif.
    Returns uint8 array (H, W): 1 inside features, 0 outside.
    Handles all geometry types: Polygon, LineString, Point.
    """
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        print(f"    [WARN] {shp_path.name} has no CRS — assuming it matches the TIFF")
    elif gdf.crs != ref_tif.crs:
        gdf = gdf.to_crs(ref_tif.crs)

    if gdf.empty:
        return np.zeros((ref_tif.height, ref_tif.width), dtype=np.uint8)

    # Buffer lines and points to give them pixel width
    geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else "Unknown"
    if "Line" in geom_type:
        # Buffer road/railway lines by ~1 pixel width in CRS units
        pixel_size = abs(ref_tif.res[0])
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(pixel_size * 1.5)
    elif "Point" in geom_type:
        pixel_size = abs(ref_tif.res[0])
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(pixel_size * 3)

    shapes = [(geom.__geo_interface__, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        return np.zeros((ref_tif.height, ref_tif.width), dtype=np.uint8)

    mask = rasterize(
        shapes=shapes,
        out_shape=(ref_tif.height, ref_tif.width),
        transform=ref_tif.transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    return mask


def build_label_mask(shp_map: dict, tif: rasterio.DatasetReader) -> np.ndarray:
    """
    Builds a single H×W label mask where each pixel = class ID.
    Renders classes in priority order so higher-priority classes overwrite lower.
    shp_map: { class_name: [shp_path1, shp_path2, ...] }
    """
    H, W  = tif.height, tif.width
    label = np.zeros((H, W), dtype=np.uint8)   # 0 = background

    # Sort classes by render priority (low → high), so high priority paints last
    sorted_classes = sorted(
        [c for c in shp_map if c in CLASS_PRIORITY],
        key=lambda c: CLASS_PRIORITY.get(c, 0)
    )

    for class_name in sorted_classes:
        class_id  = CLASSES[class_name]
        shp_paths = shp_map[class_name]
        # Merge all shapefiles for this class into one mask
        combined = np.zeros((H, W), dtype=np.uint8)
        for shp_path in shp_paths:
            binary   = rasterize_shapefile(shp_path, tif)
            combined = np.maximum(combined, binary)
        label[combined == 1] = class_id

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
