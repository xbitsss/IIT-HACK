# ─────────────────────────────────────────────────────────────────────────────
# GeoSeg Makefile — shorthand for common Docker commands
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

IMAGE  := geoseg
COMPOSE := docker compose

.PHONY: build inspect preprocess train infer shell clean help

## Build the Docker image
build:
	docker build -t $(IMAGE) .

## Inspect your raw data (bands, CRS, shapefile coverage)
inspect:
	$(COMPOSE) run --rm inspect

## Preprocess: rasterize shapefiles + tile TIFFs
preprocess:
	$(COMPOSE) run --rm preprocess

## Train SegFormer on your tiles
train:
	$(COMPOSE) run --rm train

## Run inference — set INPUT and OUTPUT, e.g.:
##   make infer INPUT=data/raw/test.tif OUTPUT=outputs/mask.tif
infer:
	$(COMPOSE) run --rm infer \
		--input $(INPUT) \
		--output $(OUTPUT) \
		--visualize

## Open an interactive shell inside the container
shell:
	$(COMPOSE) run --rm shell

## Full pipeline: build → preprocess → train
pipeline: build preprocess train

## Remove generated tiles (keeps raw data + checkpoints)
clean-tiles:
	rm -rf data/processed/

## Remove everything generated (tiles, checkpoints, outputs)
clean:
	rm -rf data/processed/ outputs/ checkpoints/

## Show help
help:
	@grep -E '^##' Makefile | sed 's/## //'
