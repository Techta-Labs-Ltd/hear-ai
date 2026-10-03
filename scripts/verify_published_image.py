import argparse
import json
import os

import requests

p = argparse.ArgumentParser()
p.add_argument("--write", required=True)
p.add_argument("--compare")
a = p.parse_args()
registry = "techta-labs-ltd/hear-ai"
response = requests.get(
    "https://ghcr.io/token",
    params={"scope": "repository:" + registry + ":pull", "service": "ghcr.io"},
    auth=(os.environ["GHCR_USERNAME"], os.environ["GHCR_TOKEN"]),
    timeout=30,
)
response.raise_for_status()
headers = {
    "Authorization": "Bearer " + response.json()["token"],
    "Accept": "application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json",
}
response = requests.get(
    "https://ghcr.io/v2/" + registry + "/manifests/cutover-7ef75975d2ad",
    headers=headers,
    timeout=30,
)
response.raise_for_status()
digest = response.headers["Docker-Content-Digest"]
manifest = response.json()
if "manifests" in manifest:
    child = next(
        x for x in manifest["manifests"] if x.get("platform", {}).get("architecture") == "amd64"
    )
    response = requests.get(
        "https://ghcr.io/v2/" + registry + "/manifests/" + child["digest"],
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    manifest = response.json()
response = requests.get(
    "https://ghcr.io/v2/" + registry + "/blobs/" + manifest["config"]["digest"],
    headers=headers,
    timeout=30,
)
response.raise_for_status()
config = response.json()
assert (
    config["config"]["Labels"]["org.opencontainers.image.source"]
    == "https://github.com/Techta-Labs-Ltd/hear-ai"
)
response = requests.get(
    "https://api.github.com/orgs/Techta-Labs-Ltd/packages/container/hear-ai",
    headers={
        "Authorization": "Bearer " + os.environ["GHCR_TOKEN"],
        "Accept": "application/vnd.github+json",
    },
    timeout=30,
)
response.raise_for_status()
package = response.json()
assert (package.get("repository") or {}).get("full_name") == "Techta-Labs-Ltd/hear-ai"
data = {
    "image": "ghcr.io/" + registry + "@" + digest,
    "configuration_digest": manifest["config"]["digest"],
    "repository": "Techta-Labs-Ltd/hear-ai",
    "rootfs": config["rootfs"],
    "runtime_configuration": config["config"],
}
if a.compare:
    with open(a.compare) as source:
        before = json.load(source)
    assert before["rootfs"] == data["rootfs"], "Runtime filesystem changed"
    old_runtime = dict(before["runtime_configuration"])
    new_runtime = dict(data["runtime_configuration"])
    old_labels = old_runtime.pop("Labels", {}) or {}
    new_labels = new_runtime.pop("Labels", {}) or {}
    assert old_runtime == new_runtime, "Runtime configuration changed"
    allowed = {"org.opencontainers.image.source", "org.opencontainers.image.description"}
    assert {k: v for k, v in old_labels.items() if k not in allowed} == {
        k: v for k, v in new_labels.items() if k not in allowed
    }, "Unexpected label changes"
with open(a.write, "w") as output:
    json.dump(data, output, indent=2)
    output.write("\n")
print(
    json.dumps(
        {
            "status": "hear_ai_repository_and_image_verified",
            "repository": data["repository"],
            "image": data["image"],
            "configuration_digest": data["configuration_digest"],
            "runtime_filesystem_unchanged": bool(a.compare),
            "runtime_configuration_unchanged": bool(a.compare),
        }
    )
)
