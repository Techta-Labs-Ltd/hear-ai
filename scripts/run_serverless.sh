#!/usr/bin/env bash
set -euo pipefail

# Only build-generated model metadata is read here. Deployment secrets are
# supplied directly through the RunPod Serverless endpoint environment.
if [[ -n "${HEAR_SOUND_CLEANUP_BUNDLE:-}" ]]; then
  release_env="${HEAR_MODEL_ROOT:-/models}/sound-cleanup-release.env"
  [[ -f "$release_env" ]] || { echo 'Sound Cleanup release metadata missing' >&2; exit 1; }
  source /app/scripts/load-env.sh "$release_env"
fi
exec python -m hear.entrypoints.serverless "$@"
