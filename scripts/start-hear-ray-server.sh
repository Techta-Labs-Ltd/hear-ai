#!/usr/bin/env bash
set -euo pipefail

readonly hear_root="/workspace/hear-ai"
readonly ray_ready_attempts="${RAY_READY_ATTEMPTS:-60}"

cd "$hear_root"
supervisor_ray_address="${RAY_ADDRESS:-}"
env_file="${HEAR_ENV_FILE:-${hear_root}/.env}"
if [[ -f "$env_file" ]]; then
  # shellcheck disable=SC1091
  source "$hear_root/scripts/load-env.sh" "$env_file"
fi
export HEAR_ENV_FILE="$env_file"
if [[ -n "$supervisor_ray_address" ]]; then
  export RAY_ADDRESS="$supervisor_ray_address"
fi
for ((attempt = 1; attempt <= ray_ready_attempts; attempt++)); do
  if pg_isready --host=127.0.0.1 --port=5432 --dbname=hear >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! pg_isready --host=127.0.0.1 --port=5432 --dbname=hear >/dev/null 2>&1; then
  echo "PostgreSQL did not become ready after ${ray_ready_attempts}s" >&2
  exit 1
fi
for ((attempt = 1; attempt <= ray_ready_attempts; attempt++)); do
  if uv run --no-project ray status --address="${RAY_ADDRESS:-auto}" >/dev/null 2>&1; then
    exec uv run --no-project python main.py
  fi
  sleep 1
done

echo "Ray head did not become ready after ${ray_ready_attempts}s" >&2
exit 1
