"""
01_preprocess.py — Parallel single-pass preprocessing with strict disk budget.

Architecture
────────────
Rasterisation is the bottleneck: ~1-2s per 8192² chunk × 435 chunks for a
large TIFF. On a 20-core machine, parallelising across chunks cuts this by
~16×.

Worker/collector design:
  • A multiprocessing.Pool processes chunks in parallel.
    Each worker: reads bands, rasterises labels+coverage, filters tiles,
    returns tile data (image arrays + mask arrays + metadata) to the
    main process via a result queue.
  • The main process (collector) receives results, enforces the disk budget
    and token buckets, and writes .npy files.
    All budget/token state lives only in the main process — no shared memory
    races possible.

This keeps disk I/O and budget accounting serial (safe) while making the
expensive rasterise calls parallel (fast).

RAM per worker: ~1 GB bands + ~256 MB label + ~256 MB coverage ≈ 1.5 GB.
With 16 workers: ~24 GB peak. Fine on a 64 GB machine.
Tune NUM_WORKERS down if RAM is tight.
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
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import random as _random

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, SHP_DIR,
    CLASSES, SHAPEFILE_MAP, CLASS_PRIORITY, CLASS_LABELS,
    TILE_SIZE, TILE_OVERLAP, BAND_INDICES,
    MIN_VALID_RATIO, MIN_COVERAGE_RATIO,
    MAX_PROCESSED_GB, RANDOM_SEED,
    SAMPLER_CLASS_WEIGHTS,
)

# Worker count: use up to 16 processes, always leaving at least 2 cores free
# for the OS and disk I/O.  We use max(cpus//2, cpus-4) so that on a
# 4-CPU Docker default we still get 2 workers instead of 0.
# To get full 16 workers: set Docker Desktop → Resources → CPU to 20
# AND ensure cpus:'20' is set in docker-compose.yml base anchor.
_available_cpus = multiprocessing.cpu_count()
NUM_WORKERS     = min(16, max(2, _available_cpus - max(2, _available_cpus // 8)))

CHUNK_SIZE      = 8192
_COVERAGE_RATIO = max(float(MIN_COVERAGE_RATIO), 0.05)
_BYTES_PER_TILE = (len(BAND_INDICES) * TILE_SIZE * TILE_SIZE * 4
                   + TILE_SIZE * TILE_SIZE * 1)
_BUDGET_BYTES   = int(MAX_PROCESSED_GB * 1024 ** 3)
_MAX_TILES      = _BUDGET_BYTES // _BYTES_PER_TILE


# ── Shapefile helpers ─────────────────────────────────────────────────────────

def find_shapefiles(shp_dirs) -> dict:
    """
    shp_dirs: Path or list of Paths — supports multiple SHP folders (Approach B).
    Searches all directories; later directories take precedence for the same
    class name, but in practice all dirs should have the same filenames.
    """
    if isinstance(shp_dirs, Path):
        shp_dirs = [shp_dirs]

    shp_map = {}
    for shp_dir in shp_dirs:
        print(f"\nShapefiles: {shp_dir}", flush=True)
        for class_name, filename in SHAPEFILE_MAP.items():
            shp_path = shp_dir / filename
            if shp_path.exists():
                shp_map[class_name] = shp_path
                print(f"  ✓ {class_name} → {shp_path}", flush=True)
            else:
                if class_name not in shp_map:
                    print(f"  ✗ Not found: {shp_path}", flush=True)
    return shp_map


def load_gdfs(shp_map: dict, tif_crs) -> dict:
    gdfs = {}
    for class_name, shp_path in shp_map.items():
        gdf = gpd.read_file(shp_path)
        if gdf.crs is not None and gdf.crs != tif_crs:
            gdf = gdf.to_crs(tif_crs)
        gdfs[class_name] = gdf
    return gdfs


# ── Rasterisation ─────────────────────────────────────────────────────────────

def rasterize_labels(gdfs, win_transform, h, w):
    label = np.zeros((h, w), dtype=np.uint8)
    for class_name in sorted(gdfs.keys(), key=lambda c: CLASS_PRIORITY.get(c, 0)):
        gdf    = gdfs[class_name]
        shapes = [(g.__geo_interface__, 1) for g in gdf.geometry if g is not None]
        if not shapes:
            continue
        binary = rasterize(shapes=shapes, out_shape=(h, w), transform=win_transform,
                           fill=0, dtype=np.uint8, all_touched=True)
        label[binary == 1] = CLASSES[class_name]
    return label


def rasterize_coverage(gdfs, win_transform, h, w):
    coverage = np.zeros((h, w), dtype=np.uint8)
    for gdf in gdfs.values():
        shapes = [(g.__geo_interface__, 1) for g in gdf.geometry if g is not None]
        if not shapes:
            continue
        binary = rasterize(shapes=shapes, out_shape=(h, w), transform=win_transform,
                           fill=0, dtype=np.uint8, all_touched=True)
        coverage = np.maximum(coverage, binary)
    return coverage


def normalize(bands: np.ndarray) -> np.ndarray:
    """
    Robust per-band scaling to [0, 1] using p2/p98 percentile clipping.

    Tiles are stored on disk in [0, 1].  ImageNet mean/std standardization
    is applied in GeoSegDataset.__getitem__ AFTER augmentation — doing it
    here would cause albumentations (HueSaturationValue, CLAHE, etc.) to
    receive out-of-range inputs and produce garbage outputs.
    """
    bands = bands.astype(np.float32)
    for i in range(bands.shape[0]):
        b        = bands[i]
        lo, hi   = np.percentile(b, 2), np.percentile(b, 98)
        bands[i] = np.clip((b - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return bands


# ── Token buckets ─────────────────────────────────────────────────────────────

def build_token_buckets(max_tiles: int) -> dict:
    all_cls      = list(set(CLASSES.values()))
    total_weight = sum(SAMPLER_CLASS_WEIGHTS.get(c, 1.0) for c in all_cls)
    buckets = {
        c: max(1, round(max_tiles * SAMPLER_CLASS_WEIGHTS.get(c, 1.0) / total_weight))
        for c in all_cls
    }
    diff = sum(buckets.values()) - max_tiles
    if diff > 0:
        buckets[0] = max(1, buckets[0] - diff)
    elif diff < 0:
        buckets[0] = buckets.get(0, 1) + abs(diff)
    return buckets


# ── Worker function (runs in subprocess) ─────────────────────────────────────

def process_chunk(args):
    """
    Called in a worker process. Does all the expensive CPU work:
      - Read bands from the TIFF window
      - Rasterise labels + coverage
      - Filter tiles by validity and coverage
      - Return passing tiles as (image_array, mask_array, meta_dict) tuples

    Does NOT write to disk and does NOT touch the budget/token state.
    The main process handles all of that after receiving results.

    Returns list of (img_arr, msk_arr, meta_dict) for passing tiles,
    or an empty list if the chunk has no coverage.
    """
    (tif_path_str, shp_map_paths, row_off, col_off,
     chunk_h, chunk_w, stem) = args

    stride = TILE_SIZE - TILE_OVERLAP
    results = []

    try:
        # Load shapefiles fresh in each worker (not picklable as GeoDataFrames)
        with rasterio.open(tif_path_str) as tif:
            window        = Window(col_off, row_off, chunk_w, chunk_h)
            win_transform = rasterio.windows.transform(window, tif.transform)

            # Load and reproject GDFs inside worker
            gdfs = {}
            for class_name, shp_path in shp_map_paths.items():
                gdf = gpd.read_file(shp_path)
                if gdf.crs is not None and gdf.crs != tif.crs:
                    gdf = gdf.to_crs(tif.crs)
                gdfs[class_name] = gdf

            raw      = tif.read(BAND_INDICES, window=window)
            valid    = ~np.all(raw == 0, axis=0)
            coverage = rasterize_coverage(gdfs, win_transform, chunk_h, chunk_w)

            # Cheap early exit — entire chunk outside all polygons
            if coverage.max() == 0:
                return []

            label = rasterize_labels(gdfs, win_transform, chunk_h, chunk_w)
            bands = normalize(raw)
            del raw

            for r in range(0, chunk_h - TILE_SIZE + 1, stride):
                for c in range(0, chunk_w - TILE_SIZE + 1, stride):
                    tv = valid[r:r+TILE_SIZE, c:c+TILE_SIZE]
                    tc = coverage[r:r+TILE_SIZE, c:c+TILE_SIZE]

                    if tv.mean() < MIN_VALID_RATIO:
                        continue
                    if tc.mean() < _COVERAGE_RATIO:
                        continue

                    tile_img   = bands[:, r:r+TILE_SIZE, c:c+TILE_SIZE].copy()
                    tile_mask  = label[r:r+TILE_SIZE, c:c+TILE_SIZE].copy()
                    class_ids  = [int(u) for u in np.unique(tile_mask)]
                    gr         = row_off + r
                    gc         = col_off + c
                    tid        = f"{stem}_r{gr:06d}_c{gc:06d}"

                    results.append((
                        tile_img,
                        tile_mask,
                        {
                            "id":             tid,
                            "source":         Path(tif_path_str).name,
                            "folder":         str(Path(tif_path_str).parent),
                            "row":            gr,
                            "col":            gc,
                            "coverage_ratio": float(tc.mean()),
                            "class_ids":      class_ids,
                        }
                    ))

    except Exception as e:
        print(f"  [WARN] Worker chunk row={row_off} col={col_off}: {e}", flush=True)
        traceback.print_exc()

    return results


# ── Per-TIFF parallel processing ─────────────────────────────────────────────

def process_tif_parallel(tif_path: Path, shp_map: dict, proc_dir: Path,
                          token_buckets: dict, bytes_written: list) -> tuple:
    """
    Submits all chunks of a TIFF to the process pool.
    Collects results as they complete and writes tiles to disk in the main
    process, enforcing budget and token limits.
    """
    stride = TILE_SIZE - TILE_OVERLAP
    saved  = 0
    meta   = []

    # Pass shapefile paths (strings) to workers — GeoDataFrames aren't picklable
    shp_map_paths = {k: str(v) for k, v in shp_map.items()}

    with rasterio.open(tif_path) as tif:
        W, H = tif.width, tif.height
        print(f"\n  Processing {tif_path.name}  {W:,}×{H:,}  {tif.count} bands  "
              f"workers={NUM_WORKERS}", flush=True)

        gdfs = load_gdfs(shp_map, tif.crs)
        if not any(not gdf.empty for gdf in gdfs.values()):
            print(f"  [SKIP] No shapefile features overlap this TIFF", flush=True)
            return 0, []

        stem = tif_path.stem

        # Build chunk list
        chunk_args = []
        for row_off in range(0, H - TILE_SIZE + 1, CHUNK_SIZE):
            for col_off in range(0, W - TILE_SIZE + 1, CHUNK_SIZE):
                chunk_h = min(CHUNK_SIZE + TILE_SIZE, H - row_off)
                chunk_w = min(CHUNK_SIZE + TILE_SIZE, W - col_off)
                if chunk_h < TILE_SIZE or chunk_w < TILE_SIZE:
                    continue
                chunk_args.append((
                    str(tif_path), shp_map_paths,
                    row_off, col_off, chunk_h, chunk_w, stem
                ))

    n_chunks   = len(chunk_args)
    done       = 0
    print(f"  Submitting {n_chunks} chunks to {NUM_WORKERS} workers...", flush=True)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(process_chunk, args): args for args in chunk_args}

        for future in as_completed(futures):
            done += 1
            if done % 20 == 0 or done == n_chunks:
                gb = bytes_written[0] / 1024**3
                print(f"  Chunks {done}/{n_chunks}  "
                      f"saved={saved}  "
                      f"disk={gb:.2f}/{MAX_PROCESSED_GB:.1f} GB",
                      flush=True)

            # Budget check before processing this result
            if bytes_written[0] >= _BUDGET_BYTES:
                # Cancel pending futures
                for f in futures:
                    f.cancel()
                print(f"  [BUDGET] Disk budget reached at chunk {done}/{n_chunks}",
                      flush=True)
                break

            try:
                tile_results = future.result()
            except Exception as e:
                print(f"  [WARN] Chunk result failed: {e}", flush=True)
                continue

            # Write tiles — serial in main process (safe for budget + tokens)
            for img_arr, msk_arr, tile_meta in tile_results:
                if bytes_written[0] >= _BUDGET_BYTES:
                    break

                class_ids  = tile_meta["class_ids"]
                rarest_cls = max(class_ids,
                                 key=lambda c: SAMPLER_CLASS_WEIGHTS.get(c, 0.0))

                if token_buckets.get(rarest_cls, 0) <= 0:
                    continue

                tid = tile_meta["id"]
                np.save(proc_dir / "images" / f"{tid}.npy", img_arr)
                np.save(proc_dir / "masks"  / f"{tid}.npy", msk_arr)

                token_buckets[rarest_cls] -= 1
                bytes_written[0] += _BYTES_PER_TILE
                saved += 1
                meta.append(tile_meta)

    print(f"  {tif_path.name}: {saved} tiles written", flush=True)
    return saved, meta


# ── Entry point ───────────────────────────────────────────────────────────────

def preprocess():
    raw_dir  = Path(DATA_RAW_DIR)
    proc_dir = Path(DATA_PROCESSED_DIR)

    print(f"Clearing {proc_dir}...", flush=True)
    for subdir in ["images", "masks"]:
        d = proc_dir / subdir
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    meta_path = proc_dir / "tiles_meta.json"
    if meta_path.exists():
        meta_path.unlink()

    # ── SHP directories ────────────────────────────────────────────────────────
    # SHP_DIRS_LIST env var: colon-separated list of SHP directories.
    # Used by Approach B where multiple SHP folders cover the same TIFF set.
    # Falls back to single SHP_DIR if not set.
    shp_dirs_env = os.environ.get("SHP_DIRS_LIST", "")
    if shp_dirs_env:
        shp_dirs = [Path(p.strip()) for p in shp_dirs_env.split(":") if p.strip()]
    else:
        shp_dirs = [Path(SHP_DIR)]

    shp_map = find_shapefiles(shp_dirs)
    if not shp_map:
        print("[ERROR] No shapefiles found.")
        sys.exit(1)

    # ── TIFF file list ─────────────────────────────────────────────────────────
    # TIFF_FILES env var: colon-separated list of specific TIFF paths to process.
    # Used by Approach B to process a random subset of files as one shard.
    # Falls back to scanning raw_dir recursively if not set.
    tiff_files_env = os.environ.get("TIFF_FILES", "")
    if tiff_files_env:
        tif_files = [Path(p.strip()) for p in tiff_files_env.split(":") if p.strip()]
        tif_files = [p for p in tif_files if p.exists()]
        shard_label = f"approach-b split ({len(tif_files)} TIFFs)"
    else:
        tif_files   = list(raw_dir.glob("**/*.tif")) + list(raw_dir.glob("**/*.tiff"))
        shard_label = f"{raw_dir.name}  ({len(tif_files)} TIFF(s))"

    if not tif_files:
        print(f"[ERROR] No .tif files found.")
        sys.exit(1)

    print(f"\nShard       : {shard_label}", flush=True)
    print(f"Workers     : {NUM_WORKERS}", flush=True)
    print(f"Budget      : {MAX_PROCESSED_GB:.2f} GB  "
          f"→ max {_MAX_TILES:,} tiles  "
          f"({_BYTES_PER_TILE/1024**2:.2f} MB/tile)", flush=True)
    print(f"Coverage    : ≥{_COVERAGE_RATIO*100:.0f}% polygon overlap per tile\n",
          flush=True)

    token_buckets = build_token_buckets(_MAX_TILES)
    print("Per-class tile budgets:", flush=True)
    for cid, tokens in sorted(token_buckets.items()):
        lbl = CLASS_LABELS[cid] if cid < len(CLASS_LABELS) else f"cls{cid}"
        print(f"  {lbl:12s}: {tokens:6d} tiles", flush=True)
    print(flush=True)

    bytes_written = [0]
    all_meta      = []
    total_saved   = 0

    for tif_path in tif_files:
        try:
            n, meta = process_tif_parallel(tif_path, shp_map, proc_dir,
                                           token_buckets, bytes_written)
            total_saved += n
            all_meta    += meta
        except Exception as e:
            print(f"[ERROR] {tif_path.name}: {e}", flush=True)
            traceback.print_exc()

        if bytes_written[0] >= _BUDGET_BYTES:
            print("\n[BUDGET] Disk budget reached — skipping remaining TIFFs.",
                  flush=True)
            break

    if not all_meta:
        print("[ERROR] No tiles written. Check shapefile coverage and paths.")
        sys.exit(1)

    _random.Random(RANDOM_SEED).shuffle(all_meta)

    class_tile_counts = Counter(
        cid for t in all_meta for cid in t["class_ids"]
    )
    print("\n── Class tile distribution ──────────────────────────────", flush=True)
    for cid, cnt in sorted(class_tile_counts.items()):
        lbl = CLASS_LABELS[cid] if cid < len(CLASS_LABELS) else f"cls{cid}"
        print(f"  {lbl:12s}: {cnt:6d} ({cnt/max(total_saved,1)*100:.1f}%)",
              flush=True)

    actual_gb = bytes_written[0] / 1024**3
    print(f"\nDisk used: {actual_gb:.2f} GB / {MAX_PROCESSED_GB:.2f} GB budget",
          flush=True)

    with open(proc_dir / "tiles_meta.json", "w") as f:
        json.dump({
            "tiles":             all_meta,
            "num_bands":         len(BAND_INDICES),
            "tile_size":         TILE_SIZE,
            "class_tile_counts": dict(class_tile_counts),
            "total_tiles":       total_saved,
            "budget_gb":         MAX_PROCESSED_GB,
        }, f, indent=2)

    print(f"\n✓ Done. {total_saved:,} tiles → {proc_dir}", flush=True)


if __name__ == "__main__":
    # Required on Windows and for ProcessPoolExecutor in general
    multiprocessing.freeze_support()
    preprocess()