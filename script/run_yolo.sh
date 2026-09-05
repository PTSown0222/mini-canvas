#!/bin/bash

set -e

echo "=== Test Run Model YOLO ==="
uv run python -m src.yolo11.yolo11_train --num_epochs 1