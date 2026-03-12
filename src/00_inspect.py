"""
00_inspect.py
─────────────
Run this FIRST to understand your data before preprocessing.
Tells you: band count, CRS, resolution, shapefile classes, class pixel coverage.

Usage:
    python src/00_inspect.py
"""

import sys
import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW_DIR, CLASSES

def inspect():
    raw_dir = Path(DATA_RAW_DIR)
    print("=" * 60)
    print("GeoSeg Data Inspector")
    print("=" * 60)

    # ── TIFFs ──
    tifs = list(raw_dir.glob("**/*.tif")) + list(raw_dir.glob("**/*.tiff"))
    print(f"\nFound {len(tifs)} TIFF file(s):")
    for tif_path in tifs:
        with rasterio.open(tif_path) as tif:
            print(f"\n  {tif_path.name}")
            print(f"    Size:       {tif.width} × {tif.height} pixels")
            print(f"    Bands:      {tif.count}")
            print(f"    CRS:        {tif.crs}")
            print(f"    Resolution: {tif.res[0]:.4f} × {tif.res[1]:.4f}")
            print(f"    Bounds:     {tif.bounds}")
            print(f"    NoData:     {tif.nodata}")
            print(f"    Dtype:      {tif.dtypes[0]}")

            # Band stats
            for i in range(1, min(tif.count + 1, 6)):   # show up to 5 bands
                band = tif.read(i).astype(np.float32)
                valid = band[band != tif.nodata] if tif.nodata else band.flatten()
                print(f"    Band {i}: min={valid.min():.2f}  max={valid.max():.2f}  "
                      f"mean={valid.mean():.2f}  std={valid.std():.2f}")

    # ── Shapefiles ──
    shps = list(raw_dir.glob("**/*.shp"))
    print(f"\nFound {len(shps)} shapefile(s):")
    for shp_path in shps:
        gdf = gpd.read_file(shp_path)
        print(f"\n  {shp_path.name}")
        print(f"    Features:  {len(gdf)}")
        print(f"    CRS:       {gdf.crs}")
        print(f"    Geometry:  {gdf.geometry.geom_type.unique().tolist()}")
        print(f"    Bounds:    {gdf.total_bounds}")
        if len(gdf.columns) > 1:
            print(f"    Columns:   {list(gdf.columns)}")

    # ── Config Check ──
    print(f"\nConfig class map: {CLASSES}")
    print("\nMake sure your shapefile filenames contain the class keys above.")
    print("Edit src/config.py → CLASSES if they don't match.\n")


if __name__ == "__main__":
    inspect()
