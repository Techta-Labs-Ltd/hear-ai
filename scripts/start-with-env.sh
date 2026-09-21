#!/usr/bin/env bash
set -euo pipefail

readonly project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly env_file="${HEAR_ENV_FILE:-${project_root}/.env}"
supervisor_ray_address="${RAY_ADDRESS:-}"

if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
fi
if [[ -n "$supervisor_ray_address" ]]; then
  export RAY_ADDRESS="$supervisor_ray_address"
fi

exec "$@"
