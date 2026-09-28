#!/usr/bin/env bash
set -euo pipefail
target=runpod-stack
tag=hear-ai:runpod-local
publish=--load
dry_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --push) publish=--push; shift ;;
    --dry-run) dry_run=true; shift ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
case "$target" in runpod-stack|pipeline-pod|pipeline-serverless|transcription-pod|transcription-serverless|reconstruction-pod|reconstruction-serverless|magic-clean-natural-pod|magic-clean-natural-serverless) ;; *) echo 'Invalid image target' >&2; exit 2 ;; esac
context="${RUNFILES_DIR:-$0.runfiles}/_main/hear-runtime-context.tar"
if [[ ! -f "$context" ]]; then context="${BUILD_WORKSPACE_DIRECTORY:?Run with bazel run}/bazel-bin/hear-runtime-context.tar"; fi
[[ -f "$context" ]] || { echo 'Build context missing' >&2; exit 1; }
if [[ "$dry_run" == true ]]; then
  printf 'Context: %s\nTarget: %s\nTag: %s\nOutput: %s\n' "$context" "$target" "$tag" "$publish"
  exit 0
fi
command -v docker >/dev/null || { echo 'The image-builder host needs Docker Buildx; the inference Pod does not.' >&2; exit 1; }
docker buildx version >/dev/null
exec docker buildx build --platform linux/amd64 --target "$target" --tag "$tag" "$publish" - < "$context"
