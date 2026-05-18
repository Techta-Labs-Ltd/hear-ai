#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  . "$ROOT/.env"
  set +a
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "[run-server] DATABASE_URL is not set — check $ROOT/.env" >&2
  exit 1
fi

exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
