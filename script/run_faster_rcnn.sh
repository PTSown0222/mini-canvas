#!/bin/bash

set -e

echo "=== Test Run Model ==="
uv run python -m src.faster_rcnn.train --num_epochs 1