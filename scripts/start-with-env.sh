#!/usr/bin/env bash
set -euo pipefail

readonly project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly env_file="${HEAR_ENV_FILE:-${project_root}/.env}"

if [[ -f "$env_file" ]]; then
  source "$project_root/scripts/load-env.sh" "$env_file"
fi

export HEAR_ENV_FILE="$env_file"
exec "$@"
