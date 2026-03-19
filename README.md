# GeoSeg — Geospatial Semantic Segmentation Pipeline

End-to-end pipeline: GeoTIFF + Shapefile → trained SegFormer → predicted mask GeoTIFF

## Project Structure
```
geoseg/
├── data/
│   ├── raw/          ← put your .tif and .shp files here
│   └── processed/    ← auto-generated tiles + masks
├── src/
│   ├── 01_preprocess.py      ← rasterize shapefiles, tile TIFFs
│   ├── 02_dataset.py         ← PyTorch Dataset class
│   ├── 03_train.py           ← SegFormer fine-tuning
│   ├── 04_inference.py       ← sliding window inference → GeoTIFF output
│   └── config.py             ← all settings in one place
├── outputs/                  ← predicted masks saved here
├── requirements.txt
└── README.md
```

## Quickstart
```bash
pip install -r requirements.txt

# 1. Put your TIFFs + SHPs in data/raw/
# 2. Run preprocessing
python src/01_preprocess.py

# 3. Train
python src/03_train.py

# 4. Predict on new TIFFs
python src/04_inference.py --input data/raw/test.tif --output outputs/test_mask.tif
```

## Classes
| ID | Class       |
|----|-------------|
| 0  | Background  |
| 1  | Built-up    |
| 2  | Road        |
| 3  | Waterbody   |
