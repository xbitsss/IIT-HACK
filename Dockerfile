FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install geo stack via conda — use strict channel priority to avoid conflicts
RUN conda update -n base -c defaults conda -y && \
    conda install -c conda-forge --strict-channel-priority \
        gdal \
        rasterio \
        geopandas \
        libgdal \
        libsqlite \
        -y --quiet

# Copy requirements first for layer caching
COPY src/requirements.txt ./src/requirements.txt
RUN pip install --no-cache-dir -r src/requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
# NOTE: src/ and specialist/ are also volume-mounted at runtime (see
# docker-compose.yml) so code edits are reflected without a rebuild.
# These COPY instructions bake a working image for production deployments
# where volume mounts may not be used.
COPY src/       ./src/
COPY specialist/ ./specialist/

# Create required directories
RUN mkdir -p data/raw data/processed data/relay outputs checkpoints \
             specialist/checkpoints

# Default entrypoint is inference — override with docker compose services
ENTRYPOINT ["python", "src/04_inference.py"]
CMD ["--help"]