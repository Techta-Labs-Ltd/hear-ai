#!/usr/bin/env bash
set -euo pipefail

readonly project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly env_file="${HEAR_ENV_FILE:-${project_root}/.env}"
supervisor_ray_address="${RAY_ADDRESS:-}"

if [[ -f "$env_file" ]]; then
  # shellcheck disable=SC1091
  source "$project_root/scripts/load-env.sh" "$env_file"
fi
export HEAR_ENV_FILE="$env_file"
if [[ -n "$supervisor_ray_address" ]]; then
  export RAY_ADDRESS="$supervisor_ray_address"
fi

exec "$@"
