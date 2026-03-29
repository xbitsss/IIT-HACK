"""
01_preprocess.py — Parallel single-pass preprocessing with strict disk budget.

Architecture
────────────
Rasterisation is the bottleneck: ~1-2s per 8192² chunk × 435 chunks for a
large TIFF. On a 20-core machine, parallelising across chunks cuts this by ~16×.

Worker/collector design:
  • A ProcessPoolExecutor processes chunks in parallel.
  • The main process (collector) receives results, enforces the disk budget
    and token buckets, and writes .npy files.
    All budget/token state lives only in the main process — no shared memory
    races possible.

Process B (the only approach):
  Spatial SHP matching — each TIFF is matched to the SHP directory whose
  shapefiles geographically overlap it. Works with 1 or more SHP dirs.

Specialist bootstrap mode (SPECIALIST_PREPROCESS=1):
  Merges config_specialist.py's SHAPEFILE_MAP (includes Bridge, Railway,
  Utility) so the specialist can self-bootstrap from raw data on a machine
  that has never run the generalist pipeline.

RAM per worker: ~1.5 GB. With 16 workers: ~24 GB peak.
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

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)

from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, SHP_DIR,
    TILE_SIZE, TILE_OVERLAP, BAND_INDICES,
    MIN_VALID_RATIO, MIN_COVERAGE_RATIO,
    MAX_PROCESSED_GB, RANDOM_SEED,
)

# ── Specialist bootstrap mode ─────────────────────────────────────────────────
# When SPECIALIST_PREPROCESS=1, use config_specialist's SHAPEFILE_MAP so
# Bridge/Railway/Utility tiles are rasterised in the same pass.
_SPECIALIST_MODE = os.environ.get("SPECIALIST_PREPROCESS", "0") == "1"

if _SPECIALIST_MODE:
    _spec_dir = str(Path(_SRC_DIR).parent / "specialist")
    sys.path.insert(0, _spec_dir)
    try:
        from config_specialist import (
            CLASSES, SHAPEFILE_MAP, CLASS_PRIORITY,
            CLASS_LABELS, SAMPLER_CLASS_WEIGHTS,
        )
        print(f"[PREPROCESS] Specialist bootstrap mode: "
              f"using config_specialist SHAPEFILE_MAP ({len(SHAPEFILE_MAP)} classes)",
              flush=True)
    except ImportError:
        print("[WARN] SPECIALIST_PREPROCESS=1 but config_specialist not found — "
              "falling back to standard config.", flush=True)
        from config import CLASSES, SHAPEFILE_MAP, CLASS_PRIORITY, CLASS_LABELS, SAMPLER_CLASS_WEIGHTS
else:
    from config import CLASSES, SHAPEFILE_MAP, CLASS_PRIORITY, CLASS_LABELS, SAMPLER_CLASS_WEIGHTS


_available_cpus = multiprocessing.cpu_count()
NUM_WORKERS     = min(16, max(2, _available_cpus - max(2, _available_cpus // 8)))

CHUNK_SIZE      = 8192
_COVERAGE_RATIO = max(float(MIN_COVERAGE_RATIO), 0.05)
_BYTES_PER_TILE = (len(BAND_INDICES) * TILE_SIZE * TILE_SIZE * 4
                   + TILE_SIZE * TILE_SIZE * 1)
_BUDGET_BYTES   = int(MAX_PROCESSED_GB * 1024 ** 3)
_MAX_TILES      = _BUDGET_BYTES // _BYTES_PER_TILE


# ── Shapefile helpers ─────────────────────────────────────────────────────────

def find_shapefiles(shp_dir) -> dict:
    shp_dir = Path(shp_dir)
    shp_map = {}
    print(f"\nShapefiles: {shp_dir}", flush=True)
    for class_name, filename in SHAPEFILE_MAP.items():
        shp_path = shp_dir / filename
        if shp_path.exists():
            shp_map[class_name] = shp_path
            print(f"  ✓ {class_name} → {shp_path}", flush=True)
        else:
            print(f"  ✗ Not found: {shp_path}", flush=True)
    if not shp_map:
        print(f"  [WARN] No shapefiles matched in {shp_dir}. "
              f"Check SHAPEFILE_MAP in config.py matches your filenames.", flush=True)
    return shp_map


def find_shp_dir_for_tif(tif_path: Path, shp_dirs: list) -> Path:
    """
    Process B — spatial matching.
    Returns the SHP dir whose shapefiles geographically overlap the TIFF.
    Falls back to shp_dirs[0] with a warning if no match found.
    """
    with rasterio.open(tif_path) as tif:
        tif_bounds = tif.bounds
        tif_crs    = tif.crs

    for shp_dir in shp_dirs:
        shp_dir = Path(shp_dir)
        for filename in SHAPEFILE_MAP.values():
            shp_path = shp_dir / filename
            if not shp_path.exists():
                continue
            try:
                gdf = gpd.read_file(shp_path)
                if gdf.crs is not None and gdf.crs != tif_crs:
                    gdf = gdf.to_crs(tif_crs)
                sb = gdf.total_bounds
                if (sb[2] > tif_bounds.left  and sb[0] < tif_bounds.right and
                        sb[3] > tif_bounds.bottom and sb[1] < tif_bounds.top):
                    return shp_dir
            except Exception:
                continue

    print(f"  [WARN] No SHP dir overlaps {tif_path.name} — "
          f"using fallback {Path(shp_dirs[0]).name}", flush=True)
    return Path(shp_dirs[0])


def _clean_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    import warnings
    n_before = len(gdf)
    gdf = gdf[~gdf.geometry.isna()].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid].copy()
    n_dropped = n_before - len(gdf)
    if n_dropped:
        print(f"  [GEOM] Cleaned {n_dropped} invalid geometries "
              f"({n_before} → {len(gdf)})", flush=True)
    return gdf


def load_gdfs(shp_map: dict, tif_crs) -> dict:
    gdfs = {}
    for class_name, shp_path in shp_map.items():
        gdf = gpd.read_file(shp_path)
        if gdf.crs is not None and gdf.crs != tif_crs:
            gdf = gdf.to_crs(tif_crs)
        gdfs[class_name] = _clean_gdf(gdf)
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
    """p2/p98 clip to [0,1]. ImageNet normalization applied later in Dataset."""
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
    Runs in a worker process. Reads the TIFF chunk, rasterises labels and
    coverage, filters tiles, returns (img_arr, msk_arr, meta) tuples.
    Does NOT write to disk.
    """
    (tif_path_str, shp_map_paths, row_off, col_off,
     chunk_h, chunk_w, stem, specialist_mode) = args

    # Re-import correct config inside worker
    _src = str(Path(tif_path_str).parent)  # unused — just avoiding stale closure
    if specialist_mode:
        _spec_dir = str(Path(__file__).resolve().parent.parent / "specialist")
        sys.path.insert(0, _spec_dir)
        try:
            from config_specialist import (
                CLASSES as _CLS, CLASS_PRIORITY as _CP,
                SAMPLER_CLASS_WEIGHTS as _SCW,
            )
        except ImportError:
            from config import CLASSES as _CLS, CLASS_PRIORITY as _CP, SAMPLER_CLASS_WEIGHTS as _SCW
    else:
        from config import CLASSES as _CLS, CLASS_PRIORITY as _CP, SAMPLER_CLASS_WEIGHTS as _SCW

    stride  = TILE_SIZE - TILE_OVERLAP
    results = []

    try:
        with rasterio.open(tif_path_str) as tif:
            window        = Window(col_off, row_off, chunk_w, chunk_h)
            win_transform = rasterio.windows.transform(window, tif.transform)

            gdfs = {}
            for class_name, shp_path in shp_map_paths.items():
                gdf = gpd.read_file(shp_path)
                if gdf.crs is not None and gdf.crs != tif.crs:
                    gdf = gdf.to_crs(tif.crs)
                gdfs[class_name] = _clean_gdf(gdf)

            raw      = tif.read(BAND_INDICES, window=window)
            valid    = ~np.all(raw == 0, axis=0)
            coverage = rasterize_coverage(gdfs, win_transform, chunk_h, chunk_w)

            if coverage.max() == 0:
                return []

            # Use worker-local _CLS and _CP for correct class IDs in specialist mode
            label = np.zeros((chunk_h, chunk_w), dtype=np.uint8)
            for class_name in sorted(gdfs.keys(), key=lambda c: _CP.get(c, 0)):
                gdf    = gdfs[class_name]
                shapes = [(g.__geo_interface__, 1) for g in gdf.geometry if g is not None]
                if not shapes:
                    continue
                binary = rasterize(shapes=shapes, out_shape=(chunk_h, chunk_w),
                                   transform=win_transform, fill=0,
                                   dtype=np.uint8, all_touched=True)
                if class_name in _CLS:
                    label[binary == 1] = _CLS[class_name]

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
                    gr, gc     = row_off + r, col_off + c
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
    stride = TILE_SIZE - TILE_OVERLAP
    saved  = 0
    meta   = []

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

        chunk_args = []
        for row_off in range(0, H - TILE_SIZE + 1, CHUNK_SIZE):
            for col_off in range(0, W - TILE_SIZE + 1, CHUNK_SIZE):
                chunk_h = min(CHUNK_SIZE + TILE_SIZE, H - row_off)
                chunk_w = min(CHUNK_SIZE + TILE_SIZE, W - col_off)
                if chunk_h < TILE_SIZE or chunk_w < TILE_SIZE:
                    continue
                chunk_args.append((
                    str(tif_path), shp_map_paths,
                    row_off, col_off, chunk_h, chunk_w, stem,
                    _SPECIALIST_MODE,
                ))

    n_chunks = len(chunk_args)
    done     = 0
    print(f"  Submitting {n_chunks} chunks to {NUM_WORKERS} workers...", flush=True)

    CHUNK_TIMEOUT = 300

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(process_chunk, args): args for args in chunk_args}

        for future in as_completed(futures, timeout=CHUNK_TIMEOUT * n_chunks):
            done += 1
            if done % 20 == 0 or done == n_chunks:
                gb = bytes_written[0] / 1024**3
                print(f"  Chunks {done}/{n_chunks}  saved={saved}  "
                      f"disk={gb:.2f}/{MAX_PROCESSED_GB:.1f} GB", flush=True)

            if bytes_written[0] >= _BUDGET_BYTES:
                for f in futures:
                    f.cancel()
                print(f"  [BUDGET] Disk budget reached at chunk {done}/{n_chunks}",
                      flush=True)
                break

            try:
                tile_results = future.result(timeout=CHUNK_TIMEOUT)
            except TimeoutError:
                args = futures[future]
                print(f"  [WARN] Chunk timed out (row={args[2]} col={args[3]}) — skipping",
                      flush=True)
                continue
            except Exception as e:
                print(f"  [WARN] Chunk result failed: {e}", flush=True)
                continue

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

    mode_str = "SPECIALIST BOOTSTRAP" if _SPECIALIST_MODE else "standard"
    print(f"\nPreprocessing [{mode_str}] — clearing {proc_dir}...", flush=True)
    for subdir in ["images", "masks"]:
        d = proc_dir / subdir
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    meta_path = proc_dir / "tiles_meta.json"
    if meta_path.exists():
        meta_path.unlink()

    # ── SHP directories ────────────────────────────────────────────────────────
    shp_dirs_env = os.environ.get("SHP_DIRS_LIST", "")
    if shp_dirs_env:
        shp_dirs = [Path(p.strip()) for p in shp_dirs_env.split(":") if p.strip()]
    else:
        shp_dirs = [Path(SHP_DIR)]

    missing = [d for d in shp_dirs if not d.exists()]
    if missing:
        print(f"[ERROR] SHP directory/directories not found:", flush=True)
        for m in missing:
            print(f"  {m}", flush=True)
        if _SPECIALIST_MODE:
            print("  Tip: Add Bridge/Railway/Utility shapefiles and list them in", flush=True)
            print("       config_specialist.py → SHAPEFILE_MAP", flush=True)
        sys.exit(1)

    # ── TIFF file list ─────────────────────────────────────────────────────────
    tiff_files_env = os.environ.get("TIFF_FILES", "")
    if tiff_files_env:
        tif_files = [Path(p.strip()) for p in tiff_files_env.split(":") if p.strip()]
        tif_files = [p for p in tif_files if p.exists()]
        shard_label = f"shard split ({len(tif_files)} TIFFs)"
    else:
        tif_files   = list(raw_dir.glob("**/*.tif")) + list(raw_dir.glob("**/*.tiff"))
        shard_label = f"{raw_dir.name}  ({len(tif_files)} TIFF(s))"

    if not tif_files:
        print(f"[ERROR] No .tif files found in {raw_dir}.", flush=True)
        print(f"  Check RAW_DATA_DIR={DATA_RAW_DIR} and TIFF_FILES env vars.", flush=True)
        sys.exit(1)

    # ── Process B: spatial SHP matching for every TIFF ─────────────────────────
    n_shp = len(shp_dirs)
    print(f"\nProcess B [{mode_str}]: "
          f"{n_shp} SHP dir(s) — spatially matching each TIFF...", flush=True)

    tif_shp_maps: dict = {}
    for tif_path in tif_files:
        matched = find_shp_dir_for_tif(tif_path, shp_dirs)
        tif_shp_maps[tif_path] = find_shapefiles(matched)
        print(f"  {tif_path.name:50s} → {matched.name}", flush=True)

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
        print(f"  {lbl:14s}: {tokens:6d} tiles", flush=True)
    print(flush=True)

    bytes_written = [0]
    all_meta      = []
    total_saved   = 0

    for tif_path in tif_files:
        tif_shp_map = tif_shp_maps[tif_path]
        if not tif_shp_map:
            print(f"[SKIP] {tif_path.name} — no shapefiles matched", flush=True)
            continue
        try:
            n, meta = process_tif_parallel(tif_path, tif_shp_map, proc_dir,
                                           token_buckets, bytes_written)
            total_saved += n
            all_meta    += meta
        except Exception as e:
            print(f"[ERROR] {tif_path.name}: {e}", flush=True)
            traceback.print_exc()

        if bytes_written[0] >= _BUDGET_BYTES:
            print("\n[BUDGET] Disk budget reached — skipping remaining TIFFs.", flush=True)
            break

    if not all_meta:
        print("[ERROR] No tiles written.", flush=True)
        print("  Check:", flush=True)
        print("    (1) Shapefile filenames match SHAPEFILE_MAP in config.py", flush=True)
        print("    (2) TIFF and shapefile CRS are compatible", flush=True)
        if _SPECIALIST_MODE:
            print("    (3) Bridge/Railway/Utility shapefiles exist in your SHP folder", flush=True)
            print("        and their filenames match config_specialist.py → SHAPEFILE_MAP", flush=True)
        print("    Run: python src/00_inspect.py to diagnose", flush=True)
        sys.exit(1)

    _random.Random(RANDOM_SEED).shuffle(all_meta)

    class_tile_counts = Counter(cid for t in all_meta for cid in t["class_ids"])
    print("\n── Class tile distribution ──────────────────────────────", flush=True)
    for cid, cnt in sorted(class_tile_counts.items()):
        lbl = CLASS_LABELS[cid] if cid < len(CLASS_LABELS) else f"cls{cid}"
        print(f"  {lbl:14s}: {cnt:6d} ({cnt/max(total_saved,1)*100:.1f}%)", flush=True)

    actual_gb = bytes_written[0] / 1024**3
    print(f"\nDisk used: {actual_gb:.2f} GB / {MAX_PROCESSED_GB:.2f} GB budget", flush=True)

    with open(proc_dir / "tiles_meta.json", "w") as f:
        json.dump({
            "tiles":             all_meta,
            "num_bands":         len(BAND_INDICES),
            "tile_size":         TILE_SIZE,
            "class_tile_counts": dict(class_tile_counts),
            "total_tiles":       total_saved,
            "budget_gb":         MAX_PROCESSED_GB,
            "specialist_mode":   _SPECIALIST_MODE,
        }, f, indent=2)

    print(f"\n✓ Done. {total_saved:,} tiles → {proc_dir}", flush=True)

    if _SPECIALIST_MODE:
        minor_ids   = {4, 5, 6}
        minor_count = sum(1 for t in all_meta
                          if minor_ids & set(t.get("class_ids", [])))
        print(f"  Minor-class tiles (Bridge/Railway/Utility): {minor_count}", flush=True)
        if minor_count == 0:
            print("  [WARN] Zero minor-class tiles found — specialist will train on major classes only.", flush=True)
            print("         Add Bridge/Railway/Utility shapefiles to config_specialist.py → SHAPEFILE_MAP", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    preprocess()