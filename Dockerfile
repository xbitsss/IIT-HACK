FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN conda update -n base -c defaults conda -y && \
    conda install -c conda-forge \
        gdal \
        rasterio \
        geopandas \
        libgdal \
        -y --quiet

COPY src/requirements.txt ./src/requirements.txt
RUN pip install --no-cache-dir -r src/requirements.txt

COPY src/ ./src/
RUN mkdir -p data/raw data/processed outputs checkpoints

ENTRYPOINT ["python", "src/04_inference.py"]
CMD ["--help"]