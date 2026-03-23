"""
00_inspect.py
─────────────
Run this FIRST to understand your data before preprocessing.
Tells you: band count, CRS, resolution, shapefile classes, class pixel coverage.

Band statistics are computed from a random sample of tiles rather than reading
the full raster — safe for very large TIFFs (100k+ pixels wide).

Usage:
    python src/00_inspect.py
"""

import sys
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW_DIR, CLASSES, ALL_RAW_DIRS, SHP_DIR, SHAPEFILE_MAP

# Number of random tiles sampled per band to estimate statistics.
# Each tile is 512×512 — 20 tiles reads ~20 MB regardless of TIFF size.
_STAT_TILES   = 20
_STAT_TILE_SZ = 512


def _band_stats_sampled(tif, band_idx: int, n_tiles: int, tile_size: int) -> dict:
    """
    Estimate band statistics by reading n_tiles random windows.
    Never loads more than n_tiles × tile_size² pixels into RAM.
    """
    W, H   = tif.width, tif.height
    rng    = np.random.default_rng(42)
    pixels = []

    for _ in range(n_tiles):
        col = int(rng.integers(0, max(1, W - tile_size)))
        row = int(rng.integers(0, max(1, H - tile_size)))
        w   = min(tile_size, W - col)
        h   = min(tile_size, H - row)
        data = tif.read(band_idx, window=Window(col, row, w, h)).astype(np.float32)
        if tif.nodata is not None:
            data = data[data != tif.nodata]
        pixels.append(data.flatten())

    if not pixels:
        return {}
    arr = np.concatenate(pixels)
    if arr.size == 0:
        return {}
    return {
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
    }


def _size_str(path: Path) -> str:
    b = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def inspect():
    # Collect all raw dirs (supports both single DATA_RAW_DIR and ALL_RAW_DIRS)
    raw_dirs = []
    try:
        for d in ALL_RAW_DIRS:
            p = Path(d)
            if p.exists():
                raw_dirs.append(p)
    except Exception:
        raw_dirs = [Path(DATA_RAW_DIR)]
    if not raw_dirs:
        raw_dirs = [Path(DATA_RAW_DIR)]

    print("=" * 60)
    print("GeoSeg Data Inspector")
    print("=" * 60)

    # ── TIFFs ─────────────────────────────────────────────────────────────────
    all_tifs = []
    for raw_dir in raw_dirs:
        all_tifs += list(raw_dir.glob("**/*.tif")) + list(raw_dir.glob("**/*.tiff"))

    print(f"\nFound {len(all_tifs)} TIFF file(s) across {len(raw_dirs)} folder(s):")

    for tif_path in all_tifs:
        with rasterio.open(tif_path) as tif:
            W, H         = tif.width, tif.height
            pixel_count  = W * H
            raw_mb       = pixel_count * tif.count * np.dtype(tif.dtypes[0]).itemsize / 1024**2
            large        = raw_mb > 4096   # > 4 GB uncompressed → use sampling

            print(f"\n  {tif_path.name}  ({_size_str(tif_path)} on disk)")
            print(f"    Size:       {W:,} × {H:,} pixels  "
                  f"(~{raw_mb/1024:.1f} GB uncompressed)")
            print(f"    Bands:      {tif.count}")
            print(f"    CRS:        {tif.crs}")
            print(f"    Resolution: {tif.res[0]:.4f} × {tif.res[1]:.4f}")
            print(f"    Bounds:     {tif.bounds}")
            print(f"    NoData:     {tif.nodata}")
            print(f"    Dtype:      {tif.dtypes[0]}")

            if large:
                print(f"    [INFO] Large TIFF — band stats from "
                      f"{_STAT_TILES} random {_STAT_TILE_SZ}px tiles (not full read)")

            for i in range(1, min(tif.count + 1, 6)):
                s = _band_stats_sampled(tif, i, _STAT_TILES, _STAT_TILE_SZ)
                if s:
                    print(f"    Band {i}: min={s['min']:.2f}  max={s['max']:.2f}  "
                          f"mean={s['mean']:.2f}  std={s['std']:.2f}"
                          + ("  (sampled)" if large else ""))
                else:
                    print(f"    Band {i}: (no valid pixels in sample)")

    # ── Auto-detect SHP subdirectory in each dataset folder ───────────────────
    def find_shp_dir(dataset_dir: Path):
        """Return the first subdirectory inside dataset_dir that contains .shp files."""
        for sub in sorted(dataset_dir.iterdir()):
            if sub.is_dir() and list(sub.glob("*.shp")):
                return sub
        # Fallback: shapefiles directly in the dataset folder
        if list(dataset_dir.glob("*.shp")):
            return dataset_dir
        return None

    # Build a map: dataset_folder → shp_dir
    shp_dir_map = {}
    for raw_dir in raw_dirs:
        detected = find_shp_dir(raw_dir)
        shp_dir_map[raw_dir] = detected
        if detected:
            print(f"\n  Auto-detected SHP dir for {raw_dir.name}: {detected}")
        else:
            print(f"\n  [WARN] No SHP subdir found in {raw_dir} — run inspect to check")

    # ── Shapefiles ────────────────────────────────────────────────────────────
    seen_shps = set()
    all_shps  = []
    for shp_dir in shp_dir_map.values():
        if shp_dir and shp_dir.exists():
            for p in shp_dir.glob("**/*.shp"):
                if p not in seen_shps:
                    seen_shps.add(p)
                    all_shps.append(p)

    print(f"\nFound {len(all_shps)} shapefile(s):")
    for shp_path in all_shps:
        try:
            gdf = gpd.read_file(shp_path)
            print(f"\n  {shp_path.name}  [{shp_path.parent}]")
            print(f"    Features:  {len(gdf)}")
            print(f"    CRS:       {gdf.crs}")
            print(f"    Geometry:  {gdf.geometry.geom_type.unique().tolist()}")
            print(f"    Bounds:    {gdf.total_bounds}")
            if len(gdf.columns) > 1:
                print(f"    Columns:   {list(gdf.columns)}")
        except Exception as e:
            print(f"\n  {shp_path.name}  [ERROR reading: {e}]")

    # ── Config check ──────────────────────────────────────────────────────────
    print(f"\nConfig class map: {CLASSES}")
    print(f"\nSHAPEFILE_MAP checked against each detected SHP folder:")
    for raw_dir, shp_dir in shp_dir_map.items():
        if not shp_dir:
            print(f"  [{raw_dir.name}] — no SHP dir found")
            continue
        print(f"  [{raw_dir.name}] → {shp_dir}")
        for class_name, filename in SHAPEFILE_MAP.items():
            path   = shp_dir / filename
            status = "✓ found" if path.exists() else "✗ MISSING"
            print(f"    {class_name:12s} → {filename:45s} {status}")

    # ── Overlap check ─────────────────────────────────────────────────────────
    print("\n── Shapefile ↔ TIFF overlap check ───────────────────────────────────")
    for tif_path in all_tifs:
        with rasterio.open(tif_path) as tif:
            # Use the SHP dir for whichever dataset folder this TIFF belongs to
            dataset_dir = next(
                (rd for rd in raw_dirs if str(tif_path).startswith(str(rd))),
                list(raw_dirs)[0] if raw_dirs else None,
            )
            shp_dir  = shp_dir_map.get(dataset_dir)
            chk_shps = list(shp_dir.glob("**/*.shp")) if shp_dir else all_shps
            tif_bounds = tif.bounds
            tif_crs    = tif.crs
            overlaps   = []
            for shp_path in chk_shps:
                try:
                    gdf = gpd.read_file(shp_path)
                    if gdf.crs and gdf.crs != tif_crs:
                        gdf = gdf.to_crs(tif_crs)
                    sb = gdf.total_bounds
                    if (sb[2] > tif_bounds.left  and sb[0] < tif_bounds.right and
                            sb[3] > tif_bounds.bottom and sb[1] < tif_bounds.top):
                        overlaps.append(shp_path.name)
                except Exception:
                    pass
            status = "✓ overlaps: " + ", ".join(overlaps) if overlaps else "✗ NO shapefile overlap — will be skipped"
            print(f"  {tif_path.name[:50]:50s}  {status}")

    print()


if __name__ == "__main__":
    inspect()