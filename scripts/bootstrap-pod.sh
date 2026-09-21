#!/usr/bin/env bash
set -euo pipefail

readonly project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly venv_python="${project_root}/.venv/bin/python"
readonly fish_speech_root="/fish-speech"
local_service_key="${HEAR_LOCAL_SERVICE_KEY:-}"

start_supervisor=false
if [[ "${1:-}" == "--start" ]]; then
  start_supervisor=true
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--start]" >&2
  exit 2
fi

if [[ $EUID -ne 0 ]]; then
  echo "Run this bootstrap script as root." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  ca-certificates \
  curl \
  ffmpeg \
  git \
  libsox-dev \
  portaudio19-dev \
  postgresql \
  postgresql-client \
  supervisor

if ! command -v uv >/dev/null 2>&1; then
  curl --fail --silent --show-error --location https://astral.sh/uv/install.sh | sh
  export PATH="/root/.local/bin:${PATH}"
fi

install -d -m 0755 \
  /audio \
  /cache \
  /cache/huggingface/hub \
  /cache/huggingface/transformers \
  /cache/torch \
  /cache/uv \
  /models \
  /postgres

cd "$project_root"
if [[ ! -f .env ]]; then
  cp .env.example .env
fi

if grep -q 'replace-with-64-char-sha256' .env; then
  if [[ -z "$local_service_key" ]]; then
    local_service_key="$(openssl rand -hex 32)"
  fi
  local_service_hash="$(printf '%s' "$local_service_key" | sha256sum | cut -d ' ' -f1)"
  sed -i "s/replace-with-64-char-sha256/${local_service_hash}/g" .env
fi
if grep -q '^STORAGE_CONTEXT_ENCRYPTION_KEY=replace-with-fernet-key$' .env; then
  fernet_key="$(openssl rand -base64 32 | tr '+/' '-_')"
  sed -i "s|^STORAGE_CONTEXT_ENCRYPTION_KEY=.*$|STORAGE_CONTEXT_ENCRYPTION_KEY=${fernet_key}|" .env
fi
if grep -q '^DATABASE_URL=postgresql+psycopg2://user:password@database:5432/hear$' .env; then
  database_password="$(openssl rand -hex 24)"
  sed -i "s|^DATABASE_URL=.*$|DATABASE_URL=postgresql+psycopg2://hear:${database_password}@127.0.0.1:5432/hear|" .env
fi
chmod 0600 .env

UV_CACHE_DIR=/cache/uv uv sync --frozen --group dev --inexact
if [[ ! -d "$fish_speech_root/.git" ]]; then
  git clone --depth 1 https://github.com/fishaudio/fish-speech.git "$fish_speech_root"
fi
UV_CACHE_DIR=/cache/uv uv pip install --python "$venv_python" --no-deps -e "$fish_speech_root"
fish_runtime_packages=(
  absl-py
  argbind
  cachetools
  datasets
  descript-audio-codec
  descript-audiotools
  docstring-parser
  einx
  ffmpy
  flatten-dict
  hydra-core
  importlib-resources
  ipython
  julius
  kui
  lightning
  lightning-utilities
  loguru
  loralib
  markdown2
  modelscope
  natsort
  opencc-python-reimplemented
  ormsgpack
  pyaudio
  pyrootutils
  pystoi
  randomname
  resampy
  safetensors
  tensorboard
  tensorboard-data-server
  tiktoken
  torch-stoi
  torchmetrics
  wandb
  werkzeug
  zstandard
)
UV_CACHE_DIR=/cache/uv uv pip install --python "$venv_python" --no-deps "${fish_runtime_packages[@]}"
"$venv_python" - <<'PY'
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

print("Fish Speech inference imports are ready")
PY

bash -n scripts/start-postgres.sh scripts/start-hear-ray-server.sh
chmod +x scripts/bootstrap-pod.sh scripts/start-postgres.sh scripts/start-hear-ray-server.sh

if [[ -n "$local_service_key" ]]; then
  echo "Bootstrap complete. Save this API key in the backend secret store: ${local_service_key}"
else
  echo "Bootstrap complete. The existing backend service-key digest was preserved."
fi
echo "Start the managed stack with: supervisord -c ${project_root}/deploy/supervisord.conf"
if $start_supervisor; then
  exec supervisord -c "${project_root}/deploy/supervisord.conf"
fi
