#!/bin/bash
set -e

WORKSPACE="${WORKSPACE:-/workspace/hear-ai}"
VENV_DIR="$WORKSPACE/.venv"

echo "=== Phase 1: Install Python 3.12 ==="
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv python3.12-dev cargo unzip 2>&1 | tail -3

echo "=== Phase 2: Create venv ==="
python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools 2>&1 | tail -3

echo "=== Phase 3: Install PyTorch ==="
pip install torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3

echo "=== Phase 4: Install core deps ==="
pip install numpy 2>&1 | tail -3
pip install "transformers>=4.48.0,<5.0.0" sentencepiece scipy \
  librosa soundfile 2>&1 | tail -3

echo "=== Phase 5: Install ML deps ==="
pip install accelerate bitsandbytes \
  "pydantic-settings==2.5.0" "demucs==4.0.1" "omegaconf>=2.3.0" \
  "pyloudnorm>=0.1.1" "silero-vad>=5.1.2" "redis[hiredis]>=5.0.0" 2>&1 | tail -3

echo "=== Phase 6: Install audio enhancement ==="
pip install deepfilternet clearvoice 2>&1 | tail -3

echo "=== Phase 7: Install WhisperX (custom fork) ==="
pip install whisperx@git+https://github.com/Ahelsamahy/whisperX.git@qwenasr-and-Forced-aligner \
  --no-deps 2>&1 | tail -3
pip install qwen-asr 2>&1 | tail -3

echo "=== Phase 8: Install FastAPI and services ==="
pip install "fastapi==0.115.0" "python-multipart==0.0.12" \
  "uvicorn[standard]==0.30.0" "httpx==0.27.0" "sqlalchemy==2.0.35" \
  "psycopg2-binary==2.9.9" "sentry-sdk[fastapi]==2.14.0" \
  "boto3==1.35.0" "openai>=1.0.0" 2>&1 | tail -3

echo "=== Phase 9: Install Triton Server wheel ==="
bash "$WORKSPACE/scripts/download-triton-wheel.sh" 2>&1 | tail -5

echo "=== Setup complete ==="
python -c "import torch; print(f'torch {torch.__version__}, cuda: {torch.cuda.is_available()}')"
