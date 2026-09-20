from __future__ import annotations

import argparse
import json
import ray

from hear.tools.model_provisioning import provision_models_on_ray


class ModelDownloadCommand:
    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--root", default="/models")
        parser.add_argument("--ray-address", default="local")
        args = parser.parse_args()
        address = None if args.ray_address == "local" else args.ray_address
        ray.init(address=address, ignore_reinit_error=False)
        try:
            results = ray.get(provision_models_on_ray.remote(args.root))
        finally:
            ray.shutdown()
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(ModelDownloadCommand.main())
