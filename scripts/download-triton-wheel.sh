#!/bin/bash
set -e

TRITON_VERSION="${1:-v2.70.0}"
DOWNLOAD_URL="https://github.com/triton-inference-server/server/releases/download"
ZIP_NAME="tritonserver-2.70.0%2Bnv26.06-cu132-cp312-manylinux_2_28-x86_64.zip"
WHEEL_DIR="/workspace/hear-ai/.venv-triton-build"

if [ ! -f "$WHEEL_DIR/tritonserver-2.70.0-cp312-cp312-manylinux_2_28_x86_64.whl" ]; then
    mkdir -p "$WHEEL_DIR"
    echo "Downloading Triton Server wheel from GitHub releases..."
    curl -sL -o "/tmp/$ZIP_NAME" "$DOWNLOAD_URL/$TRITON_VERSION/$ZIP_NAME"
    echo "Extracting..."
    unzip -o "/tmp/$ZIP_NAME" -d "$WHEEL_DIR"
    echo "Wheel extracted to $WHEEL_DIR"
fi

echo "Installing Triton Server Python package..."
pip install "$WHEEL_DIR/tritonserver/python/tritonserver-2.70.0-cp312-cp312-manylinux_2_28_x86_64.whl"
echo "Triton Server installed."
