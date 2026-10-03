#!/usr/bin/env bash
set -euo pipefail

target=runpod-stack
tag=hear-ai:runpod-local
push=false
dry_run=false
fish_license_approved="${HEAR_FISH_LICENSE_APPROVED:-false}"
output_tar="${HEAR_IMAGE_TAR:-/root/hear-ai-build/hear-ai-runpod.tar}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --push) push=true; shift ;;
    --output-tar) output_tar="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

case "$target" in
  runpod-stack|pipeline-pod|pipeline-serverless|pipeline-llm-pod|pipeline-llm-serverless|transcription-pod|transcription-serverless|reconstruction-pod|reconstruction-serverless|magic-clean-natural-pod|magic-clean-natural-serverless) ;;
  *) echo 'Invalid image target' >&2; exit 2 ;;
esac
case "$fish_license_approved" in
  true|false) ;;
  *) echo 'HEAR_FISH_LICENSE_APPROVED must be true or false' >&2; exit 2 ;;
esac

context="${RUNFILES_DIR:-$0.runfiles}/_main/hear-runtime-context.tar"
if [[ ! -f "$context" ]]; then
  context="${BUILD_WORKSPACE_DIRECTORY}/bazel-bin/hear-runtime-context.tar"
fi
[[ -f "$context" ]] || { echo 'Build context missing' >&2; exit 1; }

if [[ "$dry_run" == true ]]; then
  printf 'Context: %s\nTarget: %s\nTag: %s\nTar: %s\nPush: %s\nFish assets approved: %s\n' \
    "$context" "$target" "$tag" "$output_tar" "$push" "$fish_license_approved"
  exit 0
fi

mkdir -p "$(dirname "$output_tar")"
build_revision="$(sha256sum "$context" | awk '{print $1}')"
command -v docker >/dev/null || {
  echo 'Docker with Buildx is required on the image-builder host.' >&2
  exit 1
}
docker buildx version >/dev/null
docker info >/dev/null
if [[ "$push" == true ]]; then
  docker buildx build --network=host --platform linux/amd64 \
    --target "$target" --tag "$tag" --build-arg "HEAR_BUILD_REVISION=$build_revision" --build-arg "HEAR_FISH_LICENSE_APPROVED=$fish_license_approved" --push - < "$context"
else
  docker buildx build --network=host --platform linux/amd64 \
    --target "$target" --tag "$tag" --build-arg "HEAR_BUILD_REVISION=$build_revision" --build-arg "HEAR_FISH_LICENSE_APPROVED=$fish_license_approved" --load - < "$context"
  docker save --output "$output_tar" "$tag"
fi
