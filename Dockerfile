FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

# All system libraries needed — libtiff5, libgl1, libglib2 fix PIL/cv2/rasterio errors
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libtiff5 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Conda geo stack — pre-compiled, no source build issues
RUN conda update -n base -c defaults conda -y && \
    conda install -c conda-forge \
        gdal \
        rasterio \
        geopandas \
        libgdal \
        -y --quiet

# Copy requirements first for layer caching
COPY src/requirements.txt ./src/requirements.txt
RUN pip install --no-cache-dir -r src/requirements.txt

# Copy source code
COPY src/ ./src/

RUN mkdir -p data/raw data/processed outputs checkpoints

ENTRYPOINT ["python", "src/04_inference.py"]
CMD ["--help"]