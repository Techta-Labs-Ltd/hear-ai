#!/usr/bin/env bash

# Source this file before installing dependencies or starting Hear AI on RunPod.
# It prepares root-level model and cache directories. Ray provisions any
# missing model artifacts when the server starts.

set -e

export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/models}"
export FISH_SPEECH_HOME="${FISH_SPEECH_HOME:-/fish-speech}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache}"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export UV_CACHE_DIR="${XDG_CACHE_HOME}/uv"

mkdir -p \
  "${MODEL_CACHE_DIR}" \
  "${FISH_SPEECH_HOME}" \
  "${HF_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${UV_CACHE_DIR}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '%s\n' \
    "Workspace directories prepared. Source this script to retain its environment variables:" \
    "  source scripts/runpod-workspace-env.sh"
fi
