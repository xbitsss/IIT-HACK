# ─────────────────────────────────────────────────────────────────────────────
# GeoSeg Makefile — shorthand for common Docker Compose commands
# Usage: make <target>   or   make <target> KEY=value
# ─────────────────────────────────────────────────────────────────────────────

IMAGE   := geoseg
COMPOSE := docker compose

# Inference paths — override on the command line:
#   make infer INPUT=/raw_data/test.tif OUTPUT=outputs/mask.tif
INPUT  ?= /raw_data/image.tif
OUTPUT ?= /app/outputs/mask.tif

# Data dir for the incremental pipeline (Process B):
#   make train-all DATA_DIR=/raw_data/ALL
DATA_DIR ?= /raw_data/ALL

# Optional run message — describe what this run tests:
#   make train-all DATA_DIR=/raw_data/ALL MSG="Testing NIR band + mit-b3"
MSG ?=

# Internal helper: only pass --message if MSG is set
_MSG_FLAG = $(if $(MSG),--message "$(MSG)",)

.PHONY: build inspect preprocess \
        train train-all specialist resume \
        infer combined-infer \
        visualize notify-test shell \
        pipeline versions clean-tiles clean-outputs clean help

## ── Build ─────────────────────────────────────────────────────────────────

## Build the Docker image
build:
	docker build -t $(IMAGE) .

## ── Data ──────────────────────────────────────────────────────────────────

## Inspect raw data (bands, CRS, shapefile coverage)
inspect:
	$(COMPOSE) run --rm inspect

## Preprocess: rasterize shapefiles + tile TIFFs
preprocess:
	$(COMPOSE) run --rm preprocess

## ── Training ──────────────────────────────────────────────────────────────

## Train SegFormer on preprocessed tiles (single shard, no replay)
train:
	$(COMPOSE) run --rm train $(_MSG_FLAG)

## Full incremental pipeline: shard 1 → shard 2 → specialist
##   make train-all DATA_DIR=/raw_data/ALL MSG="Initial production run"
train-all:
	$(COMPOSE) run --rm train-all --data-dir $(DATA_DIR) $(_MSG_FLAG)

## Resume incremental pipeline from shard 2 (shard 1 already done)
##   make resume DATA_DIR=/raw_data/ALL
resume:
	$(COMPOSE) run --rm train-all --data-dir $(DATA_DIR) --from 2 $(_MSG_FLAG)

## Train specialist only (tiles in processed/ or replay/ must exist)
##   make specialist MSG="Lowering railway threshold"
specialist:
	$(COMPOSE) run --rm specialist $(_MSG_FLAG)

## ── Inference ─────────────────────────────────────────────────────────────

## Generalist inference (4-class mask)
##   make infer INPUT=/raw_data/test.tif OUTPUT=outputs/mask.tif
infer:
	$(COMPOSE) run --rm infer \
		--input $(INPUT) \
		--output $(OUTPUT)

## Combined generalist + specialist inference (7-class mask with Bridge/Railway/Utility)
##   make combined-infer INPUT=/raw_data/test.tif OUTPUT=outputs/combined.tif
combined-infer:
	$(COMPOSE) run --rm combined-infer \
		--input $(INPUT) \
		--output $(OUTPUT)

## ── Utilities ─────────────────────────────────────────────────────────────

## Visualize preprocessed tiles
visualize:
	$(COMPOSE) run --rm visualize

## Send a test notification (check email/webhook config)
notify-test:
	$(COMPOSE) run --rm notify-test

## Open an interactive shell inside the container
shell:
	$(COMPOSE) run --rm shell

## Show version history for all models
versions:
	$(COMPOSE) run --rm shell -c "python src/run_version.py --list"

## ── Pipelines ─────────────────────────────────────────────────────────────

## Full pipeline from scratch: build image → run incremental training
##   make pipeline DATA_DIR=/raw_data/ALL MSG="First production run"
pipeline: build train-all

## ── Cleanup ───────────────────────────────────────────────────────────────

## Remove processed tiles only (keeps checkpoints + outputs + replay)
clean-tiles:
	rm -rf data/processed/

## Remove outputs only (keeps checkpoints + tiles + replay)
clean-outputs:
	rm -rf outputs/

## Remove tiles + outputs + replay (keeps checkpoints — your trained model is SAFE)
clean:
	rm -rf data/processed/ data/replay/ outputs/

## ── Help ──────────────────────────────────────────────────────────────────

## Show this help
help:
	@echo ""
	@echo "GeoSeg — available make targets:"
	@echo ""
	@grep -E '^##' Makefile | sed 's/^## /  /'
	@echo ""
	@echo "Examples:"
	@echo "  make train-all DATA_DIR=/raw_data/ALL MSG=\"Testing NIR band\""
	@echo "  make resume    DATA_DIR=/raw_data/ALL"
	@echo "  make specialist MSG=\"Lowering railway threshold to 0.30\""
	@echo "  make versions"
	@echo ""