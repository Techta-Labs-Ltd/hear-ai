#!/usr/bin/env bash
set -euo pipefail

# Docker and source checkouts use the same gateway/consumer entrypoint.
project_root="${HEAR_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ ! -f "$project_root/scripts/run_pod_stack.sh" ]]; then
  printf 'Runtime scripts not found under %s\n' "$project_root" >&2
  exit 1
fi
python_bin="${HEAR_PYTHON_BIN:-/opt/hear-ai-v11/venvs/${HEAR_WORKER_ROLE:-transcription}/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  printf 'Runtime environment is missing: %s\n' "$python_bin" >&2
  exit 1
fi
cd "$project_root"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
export HEAR_ROLE_PYTHON_BIN="$python_bin"
export HEAR_GATEWAY_PYTHON_BIN="$python_bin"
export HEAR_POD_STACK_ROLES="${HEAR_POD_STACK_ROLES:-${HEAR_WORKER_ROLE:-transcription}}"
exec bash "$project_root/scripts/run_pod_stack.sh"
