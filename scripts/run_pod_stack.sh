#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_env_file="${project_root}/.env"
if [[ -f /root/hear-ai-v11/runtime.env ]]; then
  default_env_file=/root/hear-ai-v11/runtime.env
fi
env_file="${HEAR_ENV_FILE:-${default_env_file}}"
if [[ -f "$env_file" ]]; then
  source "$project_root/scripts/load-env.sh" "$env_file"
fi

roles="${HEAR_POD_STACK_ROLES:-pipeline}"
declare -a children=()
declare -a started_roles=()
"$project_root/scripts/ensure_rabbitmq.sh"

cleanup() {
  for pid in "${children[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

IFS=',' read -ra role_list <<< "$roles"
pipeline_available=false
for configured_role in "${role_list[@]}"; do
  configured_role="${configured_role//[[:space:]]/}"
  if [[ "$configured_role" == "pipeline" ]]; then
    pipeline_python="${HEAR_ROLE_PYTHON_BIN:-/opt/hear-ai-v11/venvs/pipeline/bin/python}"
    if [[ -x "$pipeline_python" ]]; then
      pipeline_available=true
    fi
  fi
done
declare -A seen_roles=()
for role in "${role_list[@]}"; do
  role="${role//[[:space:]]/}"
  [[ -n "$role" ]] || continue
  case "$role" in
    magic_clean_sam_audio)
      printf 'Retired SAM Audio worker is not started; queued separation jobs are not remapped.\n' >&2
      continue
      ;;
    pipeline|transcription|reconstruction|magic_clean_natural) ;;
    *)
      printf 'Unsupported worker role: %s\n' "$role" >&2
      exit 1
      ;;
  esac
  if [[ -n "${seen_roles[$role]:-}" ]]; then
    continue
  fi
  seen_roles[$role]=1
  if [[ "$role" == "transcription" && "$pipeline_available" == "true" ]]; then
    printf 'Using the pipeline worker for transcription jobs.\n'
    continue
  fi
  role_python="${HEAR_ROLE_PYTHON_BIN:-/opt/hear-ai-v11/venvs/${role}/bin/python}"
  if [[ ! -x "$role_python" ]]; then
    printf 'Skipping %s: runtime environment is missing at %s\n' "$role" "$role_python" >&2
    continue
  fi
  (
    export HEAR_WORKER_ROLE="$role"
    export HEAR_WORKER_ID="runpod-${RUNPOD_POD_ID:-local}-${role}-01"
    export HEAR_PYTHON_BIN="$role_python"
    "$role_python" -m hear.entrypoints.consumer
  ) &
  children+=("$!")
  started_roles+=("$role")
done

if [[ "${#started_roles[@]}" -eq 0 ]]; then
  printf 'No configured worker runtime could be started.\n' >&2
  exit 1
fi

HEAR_POD_STACK_ROLES="$(IFS=,; printf '%s' "${started_roles[*]}")"
export HEAR_POD_STACK_ROLES

export HTTP_HOST="${HEAR_GATEWAY_HOST:-0.0.0.0}"
export HTTP_PORT="${HEAR_GATEWAY_PORT:-8000}"
gateway_python="${HEAR_GATEWAY_PYTHON_BIN:-/opt/hear-ai-v11/venvs/gateway/bin/python}"
if [[ ! -x "$gateway_python" ]]; then
  gateway_python="${HEAR_GATEWAY_PYTHON_BIN:-/opt/hear-ai-v11/venvs/pipeline/bin/python}"
fi
if [[ ! -x "$gateway_python" ]]; then
  printf 'Gateway runtime environment is missing.\n' >&2
  exit 1
fi
"$gateway_python" -m hear.entrypoints.gateway &
gateway_pid=$!
children+=("$gateway_pid")
set +e
wait -n "${children[@]}"
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  status=1
fi
exit "$status"
