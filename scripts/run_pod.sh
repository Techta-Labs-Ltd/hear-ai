#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$project_root/scripts/ensure_rabbitmq.sh"

python_bin="${HEAR_PYTHON_BIN:-/opt/hear-ai-v11/venvs/${HEAR_WORKER_ROLE:-transcription}/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  printf 'Runtime environment is missing: %s\n' "$python_bin" >&2
  exit 1
fi
exec "$python_bin" -m hear.entrypoints.pod
