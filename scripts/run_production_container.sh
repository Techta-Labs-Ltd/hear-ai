#!/usr/bin/env bash
set -euo pipefail

image=hear-ai:production
name=hear-ai-production
env_file=/root/hear-ai-config/production.env
publish=127.0.0.1:8000:8000
gpus=all
detach=false
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) image="${2:?image required}"; shift 2 ;;
    --name) name="${2:?name required}"; shift 2 ;;
    --env-file) env_file="${2:?env file required}"; shift 2 ;;
    --publish) publish="${2:?port mapping required}"; shift 2 ;;
    --gpus) gpus="${2:?GPU selection required; use none for CPU diagnostics}"; shift 2 ;;
    --detach) detach=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

# Paths refer to the Docker host, including when using a remote Docker context.
[[ "$env_file" == /* && "$env_file" != *,* ]] || {
  echo 'Use an absolute env-file path without commas on the Docker host.' >&2
  exit 2
}

command=(docker run --init --name "$name")
if [[ "$detach" == true ]]; then
  command+=(--detach)
fi
if [[ "$gpus" != none ]]; then
  command+=(--gpus "$gpus")
fi
command+=(
  --publish "$publish"
  --mount "type=bind,src=$env_file,dst=/root/hear-ai-config/production.env,readonly"
  --env HEAR_ENV_FILE=/root/hear-ai-config/production.env
  --env HEAR_ENV_PARSER_PYTHON=/opt/hear-ai-v11/venvs/pipeline/bin/python
  --env HEAR_RUNTIME_MODE=production
  "$image" bash -c '
set -euo pipefail
source /app/scripts/load-env.sh "$HEAR_ENV_FILE"
[[ "${HEAR_RUNTIME_MODE:-}" == "production" ]] || {
  echo "Production container requires HEAR_RUNTIME_MODE=production" >&2
  exit 1
}
exec bash /app/scripts/run_pod_stack.sh
')

if [[ "$dry_run" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

command -v docker >/dev/null || { echo 'Docker is required.' >&2; exit 1; }
docker info >/dev/null
exec "${command[@]}"
