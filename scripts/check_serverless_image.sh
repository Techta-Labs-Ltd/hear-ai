#!/usr/bin/env bash
set -euo pipefail
image="${1:?Serverless image required}"
docker run --rm --entrypoint /opt/venv/bin/python "$image" -c '
import json, os
from pathlib import Path
import runpod
from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole
assert callable(runpod.serverless.register_fitness_check), "RunPod fitness checks missing"
assert os.environ["HEAR_RUNTIME_MODE"] == "production"
assert os.environ["HEAR_SERVERLESS_PRELOAD_MODELS"] == "true"
assert os.environ["HEAR_GPU_IDLE_EVICTION_ENABLED"] == "false"
assert os.environ["HEAR_SERVERLESS_MAX_CONCURRENT_JOBS"] == "1"
assert Path("/app/scripts/run_serverless.sh").is_file()
role = WorkerRole(os.environ["HEAR_WORKER_ROLE"])
if role == WorkerRole.PIPELINE:
    missing = ModelManifest(Path("/app/hear/model_manifest.json")).validate_local(Path("/models"), role)
    assert not missing, missing
else:
    assert role == WorkerRole.MAGIC_CLEAN_NATURAL
    import df.enhance
    for name in ("magic-clean/DeepFilterNet3/config.ini", "magic-clean/DeepFilterNet3/checkpoints/model_120.ckpt.best", "sound-cleanup-release.env", "sound-cleanup-v1-runtime/manifest.json", "sound-cleanup-specialist/runtime/manifest.json"):
        assert (Path("/models") / name).is_file(), name
print(json.dumps({"status": "serverless_image_structure_verified", "role": role.value, "runpod_sdk_loaded": True, "models_packaged": True, "gpu_inference_verified": False}))
'
docker image inspect "$image" --format '{{json .Config.Cmd}}' | python3 -c 'import json,sys; assert json.load(sys.stdin)==["bash","/app/scripts/run_serverless.sh"]'

