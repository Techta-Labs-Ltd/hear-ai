#!/usr/bin/env bash
set -euo pipefail
source_image=hear-ai:production-candidate
tag=ghcr.io/techta-labs-ltd/hear-ai:cutover-7ef75975d2ad
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-image) source_image="${2:?source image required}"; shift 2 ;;
    --tag) tag="${2:?destination image required}"; shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
context="${RUNFILES_DIR:-$0.runfiles}/_main/image-relink-context.tar"
if [[ ! -f "$context" ]]; then
  context="${BUILD_WORKSPACE_DIRECTORY:?}/bazel-bin/image-relink-context.tar"
fi
test -f "$context"
docker info >/dev/null
docker buildx build --platform linux/amd64 --build-arg "HEAR_SOURCE_IMAGE=$source_image" \
  --tag "$tag" --push - < "$context"
