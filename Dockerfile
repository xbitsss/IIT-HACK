# ─────────────────────────────────────────────────────────────────────────────
# GeoSeg — Geospatial Segmentation Pipeline
# Base: PyTorch + CUDA 11.8 on Ubuntu 22.04
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

# 1. System dependencies — GDAL + GEOS needed by rasterio / geopandas
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

# 2. GDAL env vars (required for rasterio to find the system GDAL)
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# 3. Optimization: Copy ONLY requirements first to cache the heavy install step.
# This prevents re-installing everything when you change your .py code.
COPY src/requirements.txt ./src/requirements.txt

# 4. Install Python deps
# We downgrade setuptools to <58 to fix the "use_2to3 is invalid" GDAL error.
# We install numpy first because the GDAL Python bindings require it during setup.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools<58.0.0" wheel numpy && \
    pip install --no-cache-dir \
        GDAL==$(gdal-config --version) \
        rasterio>=1.3.0 \
        geopandas>=0.14.0 \
    && pip install --no-cache-dir -r src/requirements.txt

# 5. Copy the rest of the source code
COPY src/ ./src/

# 6. Create expected data directories
RUN mkdir -p data/raw data/processed outputs checkpoints

# Default entrypoint
ENTRYPOINT ["python", "src/04_inference.py"]
CMD ["--help"]