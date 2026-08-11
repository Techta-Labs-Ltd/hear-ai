#!/usr/bin/env bash

# Source this file before installing dependencies, downloading model assets, or
# starting Hear AI on RunPod. It only prepares persistent directories and
# environment variables; it never installs packages or downloads models.

set -e

HEAR_WORKSPACE_ROOT="${HEAR_WORKSPACE_ROOT:-/workspace}"

export MODEL_CACHE_DIR="${HEAR_WORKSPACE_ROOT}/models"
export FISH_SPEECH_HOME="${HEAR_WORKSPACE_ROOT}/fish-speech"
export TRAINING_CHECKPOINT_DIR="${HEAR_WORKSPACE_ROOT}/checkpoints"
export XDG_CACHE_HOME="${HEAR_WORKSPACE_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export UV_CACHE_DIR="${XDG_CACHE_HOME}/uv"

mkdir -p \
  "${MODEL_CACHE_DIR}" \
  "${FISH_SPEECH_HOME}" \
  "${TRAINING_CHECKPOINT_DIR}" \
  "${HF_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${UV_CACHE_DIR}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '%s\n' \
    "Workspace directories prepared. Source this script to retain its environment variables:" \
    "  source scripts/runpod-workspace-env.sh"
fi
