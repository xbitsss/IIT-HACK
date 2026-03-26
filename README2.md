# GeoSeg — Incremental Geospatial Semantic Segmentation Pipeline

A production-grade pipeline for training and running semantic segmentation
models on large GeoTIFF satellite/aerial imagery, with a specialist overlay
for rare infrastructure classes.

---

## What This Pipeline Does

The pipeline takes raw GeoTIFF raster images and paired shapefiles as input
and produces a pixel-level classified map with 7 classes:

| ID | Class      | Description                          |
|----|------------|--------------------------------------|
| 0  | Background | Everything not labelled              |
| 1  | Built-up   | Buildings, urban structures          |
| 2  | Road       | Roads and road centre lines          |
| 3  | Water Body | Lakes, rivers, reservoirs            |
| 4  | Bridge     | Bridges (rare, thin)                 |
| 5  | Railway    | Railway lines (rare, very thin)      |
| 6  | Utility    | Utility polygons and point features  |

### Two-Model Architecture

**Generalist model** (`nvidia/mit-b5` SegFormer):
- Trained on all classes across all shards
- Strong on Background, Built-up, Road, Water Body
- Saved to `checkpoints/best_model.pt`

**Specialist model** (`nvidia/mit-b2` SegFormer):
- Trained on a 50/50 balanced dataset:
  - 50% tiles containing Bridge / Railway / Utility
  - 50% tiles containing only Built-up / Road / Water (context)
- Focused on detecting rare infrastructure classes
- Saved to `specialist/checkpoints/best_model.pt`

At inference time both models run on the same image. The specialist overrides
the generalist wherever its confidence for Bridge / Railway / Utility exceeds
a per-class threshold (default: Bridge=0.45, Railway=0.35, Utility=0.45).

### Incremental Training with Replay Buffer

Large datasets are split into shards. Each shard is preprocessed, trained,
then wiped to stay within a disk budget. A replay buffer retains 300 tiles
per shard so the model does not catastrophically forget earlier shards.

---

## Project Structure

project/
├── src/
│ ├── config.py # All pipeline settings
│ ├── 00_inspect.py # Pre-flight data inspection tool
│ ├── 01_preprocess.py # Parallel tile extraction → .npy files
│ ├── 02_dataset.py # PyTorch Dataset, augmentation, sampler
│ ├── 03_train.py # SegFormer fine-tuning (Focal+Dice, AMP)
│ ├── 04_inference.py # Generalist-only sliding window inference
│ ├── 05_visualize.py # Visualization utilities
│ ├── 06_train_incremental.sh # Master pipeline orchestration script
│ ├── 07_notify.py # Email notifications
│ └── replay_buffer.py # Anti-forgetting replay mechanism
│
├── specialist/
│ ├── config_specialist.py # Specialist overrides (model, weights, thresholds)
│ ├── 01_build_specialist_meta.py # Builds 50/50 balanced specialist dataset
│ ├── 02_dataset_specialist.py # Dataset wrapper for specialist training
│ ├── 03_train_specialist.py # Specialist training script
│ └── 04_inference_combined.py # Combined generalist + specialist inference
│
├── data/
│ ├── raw/ # Your GeoTIFF files go here
│ ├── processed/ # Auto-generated tile cache (wiped per shard)
│ └── replay/ # Persistent replay buffer across shards
│
├── checkpoints/ # Generalist model saved here
└── docker-compose.yml

text

---

## Dataset Requirements

### File Formats

- **Imagery**: GeoTIFF (`.tif` / `.tiff`), any number of bands
  - The pipeline uses bands 1, 2, 3, 4 by default (configurable in `config.py` → `BAND_INDICES`)
  - Band 4 (NIR) is left in [0,1]; bands 1–3 are ImageNet-normalised
  - Minimum recommended resolution: 0.5m/px. Works with any CRS.
  - Any raster size — the pipeline tiles it at 512×512 px with 64 px overlap

- **Labels**: ESRI Shapefiles (`.shp` + `.dbf` + `.shx` + `.prj`)
  - One shapefile per class — filenames must match `SHAPEFILE_MAP` in `config.py`
  - Default expected filenames:

| Class    | Expected Shapefile(s)                                  |
|----------|--------------------------------------------------------|
| Built-up | `BuiltUpAreatype.shp`                                  |
| Road     | `Road.shp`, `RoadCentreLine.shp`                       |
| Water    | `WaterBody.shp`, `WaterBodyLine.shp`, `WaterbodyPoint.shp` |
| Bridge   | `Bridge.shp`                                           |
| Railway  | `Railway.shp`                                          |
| Utility  | `UtilityPoly.shp`, `Utility.shp`                       |

  - Shapefiles must be in the same CRS as the GeoTIFF, or have a valid `.prj`
    file so the pipeline can reproject automatically
  - Point and line features are auto-buffered:
    - Points: 8m radius disc
    - Lines: 3m half-width strip

### Approach A — Folder-per-Shard (Recommended)

Each dataset folder is one shard. Place shapefiles in a `SHP/` subfolder:

/raw_data/
CG/
imagery_2023.tif
SHP/
BuiltUpAreatype.shp
Road.shp
RoadCentreLine.shp
WaterBody.shp
Bridge.shp
Railway.shp
UtilityPoly.shp
PB/
imagery_2024.tif
SHP/
BuiltUpAreatype.shp
...

text

Every immediate subfolder of `/raw_data/` that contains a `.tif` file is
auto-discovered as a shard — no configuration needed.

### Approach B — Single Folder, Two SHP Dirs

All TIFFs in one folder, two SHP directories (e.g. from different annotation
teams). The pipeline splits TIFFs randomly 50/50 into two shards each run:

/raw_data/ALL/
image1.tif
image2.tif
image3.tif
SHP1/
BuiltUpAreatype.shp ...
SHP2/
BuiltUpAreatype.shp ...

text

---

## Quick Start

### 1. Inspect Your Data First

Always run this before preprocessing to verify band counts, CRS, shapefile
coverage and class mapping:

```bash
docker compose run --rm train-all python src/00_inspect.py
2. Run the Full Pipeline
bash
# Approach A — auto-discover shards under /raw_data
docker compose run --rm train-all

# Approach B — single folder with two SHP dirs
docker compose run --rm train-all \
    --approach b \
    --data-dir /raw_data/ALL \
    --shp-dirs /raw_data/ALL/SHP1:/raw_data/ALL/SHP2
This runs in order:

Preprocess each shard → 512×512 tiles

Train generalist model on each shard (with replay)

Build specialist 50/50 dataset metadata

Train specialist model

Notify via email at each stage

3. Run Inference
Combined (generalist + specialist overlay) — recommended:

bash
docker compose run --rm train-all \
    python specialist/04_inference_combined.py \
    --input  data/raw/test.tif \
    --output outputs/combined_mask.tif \
    --visualize
Generalist only:

bash
docker compose run --rm train-all \
    python src/04_inference.py \
    --input  data/raw/test.tif \
    --output outputs/mask.tif
Disable TTA for faster inference (at some accuracy cost):

bash
python specialist/04_inference_combined.py \
    --input test.tif --output mask.tif --no-tta
All Pipeline Flags
06_train_incremental.sh
Flag	Default	Description
--from N	1	Resume from shard N, skip shards 1..N-1
--approach a|b	a	Approach A (folder-per-shard) or B (single folder)
--data-dir /path	/raw_data/ALL	Root folder for Approach B
--shp-dirs d1:d2	auto-detect	Colon-separated SHP dirs for Approach B
--pretrained /path.pth	—	Fine-tune from checkpoint at full LR×0.3
--init-weights /path.pth	—	Load weights only, train from epoch 0, full LR
--skip-specialist	off	Bypass the specialist training stage entirely
Environment Variables
Variable	Default	Description
RAW_DATA_ROOT	/raw_data	Root folder for Approach A shard discovery
MAX_DISK_GB	50	Hard disk ceiling for processed tiles + replay
REPLAY_TILES_PER_SHARD	300	Tiles kept per shard in the replay buffer
04_inference_combined.py
Flag	Description
--input /path.tif	Input GeoTIFF
--output /path.tif	Output classified mask GeoTIFF
--visualize	Save a side-by-side RGB + mask PNG
--no-tta	Disable 8-fold Test-Time Augmentation (faster, slightly lower accuracy)
Configuration
All pipeline parameters live in src/config.py. Key settings:

python
TILE_SIZE        = 512        # tile size in pixels
TILE_OVERLAP     = 64         # overlap between adjacent tiles
BAND_INDICES     =   # which raster bands to use (1-indexed)[1][2][3][4]
MAX_DISK_GB      = 50         # total disk budget (processed + replay)
NUM_EPOCHS       = 150        # epochs per shard
BATCH_SIZE       = 2          # per-GPU batch size
GRAD_ACCUM_STEPS = 16         # effective batch = 2 × 16 = 32
LR               = 3e-5       # peak learning rate
PATIENCE         = 30         # early stopping patience
Specialist overrides in specialist/config_specialist.py:

python
MODEL_NAME   = "nvidia/mit-b2"   # lighter backbone for smaller dataset
CHECKPOINT_DIR = "specialist/checkpoints"
CLASS_WEIGHTS  = [0.2, 0.8, 1.0, 0.8, 8.0, 10.0, 6.0]  # boost minor classes

# Per-class confidence thresholds for the overlay
BRIDGE_THRESHOLD  = 0.45
RAILWAY_THRESHOLD = 0.35   # lower: railway was IoU=0, be aggressive
UTILITY_THRESHOLD = 0.45
Disk Budget Guarantee
The pipeline guarantees total disk usage never exceeds MAX_DISK_GB:

Processed tiles for the current shard are wiped after each training step

The replay buffer keeps only REPLAY_TILES_PER_SHARD tiles per previous shard

The budget for new processed tiles is dynamically calculated as:
budget = MAX_DISK_GB − current_replay_bytes

With 2 shards × 300 replay tiles = ~2.5 GB replay, leaving ~47.5 GB for
processing on a 50 GB ceiling.

Output Files
Path	Description
checkpoints/best_model.pt	Best generalist checkpoint
checkpoints/training_curves.png	Loss and mIoU curves
checkpoints/history.json	Full training history
specialist/checkpoints/best_model.pt	Best specialist checkpoint
specialist/checkpoints/training_curves.png	Specialist training curves
data/replay/	Persistent replay tiles (do not delete between runs)
outputs/combined_mask.tif	Final georeferenced prediction (uint8 GeoTIFF)
outputs/combined_mask.viz.png	RGB + mask side-by-side visualization
Resuming After a Crash
If training crashes mid-shard, resume from where it left off:

bash
# If shard 2 crashed, resume from shard 2
docker compose run --rm train-all --from 2
The checkpoint at checkpoints/best_model.pt is always the best validated
epoch — it is never overwritten with a worse checkpoint.

Requirements
Docker with NVIDIA GPU support (nvidia-container-toolkit)

At least 24 GB GPU VRAM recommended for mit-b5 + AMP

At least 64 GB system RAM (16 parallel preprocessing workers × ~1.5 GB each)

Disk space per the MAX_DISK_GB setting (default 50 GB)

text