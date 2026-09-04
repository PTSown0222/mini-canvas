#!/bin/bash

set -e

# echo "=== Test Run Image Enhancement ==="
# uv run python -m src.basic_adjust

# echo "=== Test Run Image Enhancement ==="
# uv run python -m src.enhance_image

echo "==== Run CLI ===="
uv run streamlit run app.py

