# ─────────────────────────────────────────────────────────────────────────────
# GeoSeg — Geospatial Segmentation Pipeline
# Base: PyTorch + CUDA 11.8 on Ubuntu 22.04
#
# CPU-only build? Replace the FROM line with:
#   FROM python:3.11-slim
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# System deps — GDAL + GEOS needed by rasterio / geopandas
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        libspatialindex-dev \
        build-essential \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*

# GDAL env vars (required for rasterio to find the system GDAL)
ENV GDAL_VERSION=3.4.1
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Install Python deps first (cached layer — only re-runs if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        GDAL==$(gdal-config --version) \
        rasterio>=1.3.0 \
        geopandas>=0.14.0 \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create expected data directories
RUN mkdir -p data/raw data/processed outputs checkpoints

# Default entrypoint — can be overridden at runtime
ENTRYPOINT ["python", "src/04_inference.py"]
CMD ["--help"]
