# GeoSeg Pipeline Guide

Everything you need to run, resume, fix, and extend the pipeline.

---

## Quick Reference

| What you want | Command |
|---|---|
| Full run from scratch | `docker compose run --rm train-all --data-dir /raw_data/ALL` |
| Resume from shard 2 | `docker compose run --rm train-all --data-dir /raw_data/ALL --from 2` |
| Specialist only | `docker compose run --rm specialist` |
| Generalist only (no specialist) | `docker compose run --rm train-all --data-dir /raw_data/ALL --skip-specialist` |
| Run with a message | `docker compose run --rm train-all --data-dir /raw_data/ALL --message "Testing NIR"` |
| Generalist inference | `docker compose run --rm infer --input /raw_data/img.tif --output /app/outputs/mask.tif` |
| Combined inference | `docker compose run --rm combined-infer --input /raw_data/img.tif --output /app/outputs/mask.tif` |
| Inspect your data | `docker compose run --rm inspect` |
| Test email notification | `docker compose run --rm notify-test` |
| View version history | `python src/run_version.py --list` |

---

## Data Layout

Your raw data folder must look like this:

```
/raw_data/ALL/
    image_001.tif
    image_002.tif
    image_003.tif
    ...
    SHP1/
        Road.shp          ← same filenames as in config.py → SHAPEFILE_MAP
        Water_Body.shp
        Built_Up_Area_type.shp
    SHP2/
        Road.shp          ← duplicate names are fine — Process B spatially
        Water_Body.shp    ← matches each TIFF to its correct SHP folder
        Built_Up_Area_type.shp
```

**Single SHP folder** also works — just put all shapefiles in one subfolder.

**What `--data-dir` means:** The folder containing both your TIFFs and your
SHP subfolders. The pipeline discovers TIFFs at depth 1 and SHP dirs by
looking for `*.shp` files in immediate subdirectories.

---

## How Versioning Works

Every run auto-increments a version number stored in `run_registry.json`.

- Generalist runs: `v1`, `v2`, `v3` …
- Specialist runs: `v1`, `v2`, `v3` … (independent counter)

Each run saves a permanently versioned checkpoint copy:
- `checkpoints/generalist_v3_best.pt`
- `specialist/checkpoints/specialist_v2_best.pt`

The `best_model.pt` always points to the latest best. You can safely roll
back by copying any versioned file over `best_model.pt`.

### Adding a Run Message

```bash
docker compose run --rm train-all \
    --data-dir /raw_data/ALL \
    --message "Added NIR band (band 4), testing mit-b3 backbone"
```

The message is stored in:
- The checkpoint file (`ckpt["run_message"]`)
- The version registry (`run_registry.json`)
- All notification emails

### View Version History

```bash
python src/run_version.py --list
```

Output example:
```json
{
  "generalist": {
    "version": 3,
    "runs": [
      {"version": 1, "timestamp": "2024-01-01T10:00:00", "message": "initial run", "best_val_miou": 0.68},
      {"version": 2, "timestamp": "2024-01-05T14:00:00", "message": "Added NIR band", "best_val_miou": 0.71},
      {"version": 3, "timestamp": "2024-01-10T09:00:00", "message": "mit-b3 backbone", "best_val_miou": 0.73}
    ]
  },
  "specialist": {
    "version": 2,
    "runs": [...]
  }
}
```

---

## Entry Points — Start From Anywhere

### 1. Full Pipeline From Scratch

```bash
docker compose run --rm train-all \
    --data-dir /raw_data/ALL \
    --message "Initial training run"
```

Steps executed:
1. Wipe stale data/processed and data/replay
2. Shuffle all TIFFs and split 50/50 into shard 1 and shard 2
3. Preprocess shard 1 → save tiles → save replay buffer
4. Train generalist on shard 1
5. Wipe shard 1 processed tiles
6. Preprocess shard 2 → save tiles → save replay buffer
7. Train generalist on shard 2 (resume from shard 1 checkpoint)
8. Build specialist metadata from shard 2 tiles + replay
9. Train specialist on specialist tiles

### 2. Resume From Shard 2

Use this when shard 1 trained successfully but the pipeline crashed during
shard 2 (or you want to re-train shard 2 with different settings).

```bash
docker compose run --rm train-all \
    --data-dir /raw_data/ALL \
    --from 2
```

What this does:
- Skips shard 1 entirely (checkpoint from shard 1 is preserved)
- Uses existing `data/replay/` from shard 1 (NOT wiped)
- Preprocesses shard 2 with a fresh random split
- Trains generalist on shard 2 in RESUME mode (LR × 0.3)
- Runs specialist afterwards

> **Note:** The TIFF split is re-randomized each run. Shard 2 will be a
> different set of TIFFs than the previous attempt. This is intentional —
> you get different training diversity.

### 3. Specialist Only

Use this when generalist training is complete but you want to re-train or
tune the specialist without re-doing any generalist work.

```bash
docker compose run --rm specialist
```

Or with a message:
```bash
docker compose run --rm train-all \
    --specialist-only \
    --message "Lowering railway threshold to 0.30"
```

**Requirements:**
- Tiles must exist in `data/processed/` or `data/replay/`
- No generalist checkpoint required (trains from HuggingFace by default)

**Optional warm-start from generalist weights:**
```bash
python specialist/03_train_specialist.py \
    --init-weights checkpoints/best_model.pt \
    --run-message "Warm-started from generalist v3"
```

### 4. Generalist Only (Skip Specialist)

```bash
docker compose run --rm train-all \
    --data-dir /raw_data/ALL \
    --skip-specialist
```

### 5. Fine-tune From a Pretrained Checkpoint

```bash
docker compose run --rm train-all \
    --data-dir /raw_data/ALL \
    --pretrained /raw_data/my_previous_model.pth \
    --message "Fine-tuning from external checkpoint"
```

### 6. Load Weights Only (Fresh Training)

Loads weights from a file but resets epoch to 0 and uses full LR.
Use this to start fresh training from a custom backbone.

```bash
docker compose run --rm train-all \
    --data-dir /raw_data/ALL \
    --init-weights /raw_data/backbone.pth
```

---

## Notifications

You receive emails at these events:

| Event | Timing |
|---|---|
| Training started | Immediately when training begins |
| 2-hour progress | Every 2 hours (configurable in config.py) |
| Epoch 10, 20, 30… | With training curves attached |
| Early stopping | When patience runs out |
| Shard complete | After each shard finishes |
| Error / crash | Immediately on any failure |

### Setup

Add to your `.env` file:
```
RESEND_API_KEY=re_xxxxxxxxxxxxxxxxxxxx
NOTIFY_TO=you@yourcompany.com
```

Test:
```bash
docker compose run --rm notify-test
```

---

## Inference

### Generalist Only

```bash
docker compose run --rm infer \
    --input /raw_data/test_image.tif \
    --output /app/outputs/mask.tif

# With visualization
docker compose run --rm infer \
    --input /raw_data/test_image.tif \
    --output /app/outputs/mask.tif \
    --visualize

# Fast (disable TTA — ~8× faster, slightly lower accuracy)
docker compose run --rm infer \
    --input /raw_data/test_image.tif \
    --output /app/outputs/mask.tif \
    --no-tta
```

### Combined (Generalist + Specialist Overlay)

```bash
docker compose run --rm combined-infer \
    --input /raw_data/test_image.tif \
    --output /app/outputs/combined_mask.tif \
    --visualize
```

---

## Common Problems and Fixes

### "No TIFF files found"

```
[ERROR] No TIFF files found in /raw_data/ALL
```

Check:
1. Is the host path mounted correctly in `.env`? `RAW_DATA_DIR=/path/on/host`
2. Do TIFFs sit at the top level of that folder (not in subdirectories)?
3. `ls -la /raw_data/ALL/*.tif` inside the container shell

```bash
docker compose run --rm shell
ls -la /raw_data/ALL/*.tif
```

### "No SHP subdirectories found"

```
[ERROR] No SHP subdirectories found in /raw_data/ALL
```

The pipeline looks for immediate subdirectories that contain `*.shp` files.

Check:
1. SHP files exist: `ls /raw_data/ALL/SHP1/*.shp`
2. Names match `SHAPEFILE_MAP` in `config.py`
3. Or specify explicitly: `--shp-dirs /raw_data/ALL/SHP1:/raw_data/ALL/SHP2`

### "No tiles written"

```
[ERROR] No tiles written. Check shapefile coverage and paths.
```

Run the inspector first to diagnose:
```bash
docker compose run --rm inspect
```

The inspector tells you:
- TIFF band count, CRS, resolution
- Which SHP files are found / missing
- Whether each TIFF geographically overlaps its matched SHP dir

Common causes:
- TIFF and shapefile are in different CRS (the pipeline reprojects automatically,
  but malformed CRS strings can fail silently)
- `SHAPEFILE_MAP` in `config.py` has wrong filenames
- Coverage threshold too high (`MIN_COVERAGE_RATIO` in config.py, default 0.05)

### "No tile data found for specialist"

```
[ERROR] No tile data found for specialist training.
```

The specialist needs tiles from either `data/processed/` or `data/replay/`.

Solutions:
1. Run generalist shards first so replay is populated
2. Manually copy an existing `data/replay/` from another machine
3. Preprocess tiles with specialist shapefiles (Bridge, Railway, Utility) then retry

### "Replay buffer large — processed budget clamped"

```
[WARN] Replay buffer large — processed budget clamped to 1 GB
```

Your replay buffer is eating most of your `MAX_DISK_GB` budget.
Options:
- Increase `MAX_DISK_GB` (in `.env` or docker compose command)
- Reduce `REPLAY_TILES_PER_SHARD` in `config.py`
- Clear stale replay: `rm -rf data/replay/`

### Training crashes mid-epoch

The checkpoint is always safe — it's only written when val_mIoU improves.
You will also receive an email with the exact resume command.

Resume from the last good shard:
```bash
# If shard 1 finished, resume shard 2:
docker compose run --rm train-all --data-dir /raw_data/ALL --from 2

# If both shards finished but specialist crashed:
docker compose run --rm specialist
```

### "CUDA out of memory"

Reduce batch size in `config.py`:
```python
BATCH_SIZE   = 1       # down from 2
GRAD_ACCUM_STEPS = 16  # up from 8 to keep effective batch size
```

Or switch to a lighter backbone:
```python
MODEL_NAME = "nvidia/mit-b2"   # lighter than mit-b3
```

### Specialist IoU for bridge/railway is 0.000

This means the model never predicts those classes. Causes:
1. **No specialist shapefiles configured** — add `bridge`, `railway`, `utility`
   to `SHAPEFILE_MAP` in `config_specialist.py`
2. **Threshold too high** — lower `RAILWAY_THRESHOLD` in `config_specialist.py`
   (currently 0.35, try 0.25)
3. **Not enough minority tiles** — check tile counts in specialist metadata:
   `cat data/processed/tiles_meta_specialist.json | python -m json.tool | head -30`

---

## Key Configuration Files

### `src/config.py` — Generalist settings
- `MODEL_NAME` — backbone (`nvidia/mit-b3` default)
- `NUM_EPOCHS`, `LR`, `BATCH_SIZE`, `PATIENCE`
- `CLASS_WEIGHTS` — per-class loss weighting
- `SHAPEFILE_MAP` — shapefile filename → class name
- `TILE_SIZE`, `TILE_OVERLAP` — tile geometry
- `MAX_DISK_GB` — hard disk ceiling
- `REPLAY_TILES_PER_SHARD` — continual learning buffer size

### `specialist/config_specialist.py` — Specialist settings
Inherits everything from `config.py` and overrides:
- `MODEL_NAME = "nvidia/mit-b2"` — lighter model
- `NUM_CLASSES = 7` (adds Bridge=4, Railway=5, Utility=6)
- `CLASS_WEIGHTS` — heavily up-weighted minor classes
- `CLASS_THRESHOLDS` — per-class confidence threshold for overlay
- `SHAPEFILE_MAP` — add your bridge/railway/utility shapefiles here

---

## File Structure Reference

```
project/
├── src/
│   ├── 00_inspect.py              ← Run first — diagnoses your data
│   ├── 01_preprocess.py           ← TIFF → .npy tiles (Process B only)
│   ├── 02_dataset.py              ← PyTorch Dataset + replay mixing
│   ├── 03_train.py                ← Generalist training (versioned)
│   ├── 04_inference.py            ← Generalist inference
│   ├── 05_visualize.py            ← Visualize tiles
│   ├── 07_notify.py               ← Email notifications
│   ├── config.py                  ← All generalist settings
│   ├── replay_buffer.py           ← Continual learning buffer
│   └── run_version.py             ← Run version registry (NEW)
│
├── specialist/
│   ├── 01_build_specialist_meta.py
│   ├── 02_dataset_specialist.py
│   ├── 03_train_specialist.py     ← Specialist training (independent)
│   ├── 04_inference_combined.py   ← Combined inference
│   └── config_specialist.py
│
├── checkpoints/
│   ├── best_model.pt              ← Always the latest best generalist
│   ├── generalist_v1_best.pt      ← Permanently versioned copies
│   ├── generalist_v2_best.pt
│   └── training_curves.png
│
├── specialist/checkpoints/
│   ├── best_model.pt              ← Always the latest best specialist
│   ├── specialist_v1_best.pt      ← Permanently versioned copies
│   └── training_curves.png
│
├── data/
│   ├── processed/                 ← Current shard tiles (wiped between shards)
│   └── replay/                    ← Persistent replay buffer (never wiped)
│
├── outputs/                       ← Inference masks + visualizations
│
├── 06_train_incremental.sh        ← Main entry point
├── run_registry.json              ← Version history (auto-created)
├── pipeline_state.json            ← Last run state (auto-created)
└── .env                           ← RAW_DATA_DIR, RESEND_API_KEY, NOTIFY_TO
```

---

## Rolling Back to a Previous Version

```bash
# See what versions exist
python src/run_version.py --list

# Roll back generalist to v2
cp checkpoints/generalist_v2_best.pt checkpoints/best_model.pt

# Roll back specialist to v1
cp specialist/checkpoints/specialist_v1_best.pt specialist/checkpoints/best_model.pt

# Now run combined inference with the rolled-back models
docker compose run --rm combined-infer \
    --input /raw_data/test.tif \
    --output /app/outputs/rollback_test.tif
```

---

## Specialist Pipeline — Standalone Run Guide

The specialist is **fully independent** — you do not need to re-run the
generalist to retrain it.

### Prerequisites

You need tiles. One of:
- `data/processed/` populated (from the last shard)  
- `data/replay/` populated (from any previous shard run)

### Steps

1. **Check tiles exist:**
   ```bash
   ls data/replay/ 2>/dev/null || echo "Empty"
   ls data/processed/tiles_meta.json 2>/dev/null || echo "Not found"
   ```

2. **Build specialist metadata:**
   ```bash
   python specialist/01_build_specialist_meta.py
   ```

3. **Train:**
   ```bash
   # From scratch (HuggingFace pretrained weights)
   python specialist/03_train_specialist.py --run-message "Fresh specialist run"

   # Resume from previous specialist checkpoint
   python specialist/03_train_specialist.py --resume

   # Warm-start from generalist weights (loads weights only, resets epoch)
   python specialist/03_train_specialist.py \
       --init-weights checkpoints/best_model.pt \
       --run-message "Warm-started from generalist v3"
   ```

4. **Run combined inference:**
   ```bash
   docker compose run --rm combined-infer \
       --input /raw_data/test.tif \
       --output /app/outputs/combined_mask.tif
   ```

---

## Environment Variables Reference

Set in `.env` or `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `RAW_DATA_DIR` | (required) | Host path to your data folder |
| `MAX_DISK_GB` | `50` | Hard ceiling on pipeline data |
| `REPLAY_TILES_PER_SHARD` | `300` | Tiles kept per shard in replay |
| `RESEND_API_KEY` | (optional) | Resend API key for email alerts |
| `NOTIFY_TO` | (optional) | Email address for notifications |
| `HF_TOKEN` | (optional) | HuggingFace token (if using private models) |

---

## Useful One-Liners

```bash
# Check current disk usage
du -sh data/processed data/replay checkpoints 2>/dev/null

# Count tiles in processed dir
python3 -c "import json; d=json.load(open('data/processed/tiles_meta.json')); print(len(d['tiles']))"

# Check replay buffer
python3 -c "
import sys; sys.path.insert(0,'src')
from replay_buffer import load_all_replay
tiles, _, _ = load_all_replay()
print(f'{len(tiles)} replay tiles')
"

# View version registry
python src/run_version.py --list

# Load and inspect a checkpoint
python3 -c "
import torch
ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
print('epoch:', ckpt.get('epoch'))
print('val_miou:', ckpt.get('val_miou'))
print('version:', ckpt.get('run_version'))
print('message:', ckpt.get('run_message'))
print('per_class_iou:', ckpt.get('per_class_iou'))
"

# Open a shell inside the container
docker compose run --rm shell
```
