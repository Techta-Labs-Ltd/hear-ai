"""Private source-inspection subprocess entrypoint; no model or service loading."""

import json
import sys
import threading
from dataclasses import asdict
from pathlib import Path

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode, SourceIdentity
from hear.services.magic_clean.inspection import SourceInspector


class InspectionWorker:
    @staticmethod
    def main() -> None:
        try:
            source, identity, workspace, deadline, scratch, input_bytes, frames = sys.argv[1:]
            guard = ResourceGuard(
                ResourceBudget(int(scratch), int(input_bytes), int(frames)),
                Path(workspace),
                float(deadline),
                threading.Event(),
            )
            result = SourceInspector._inspect_local(
                Path(source), SourceIdentity.model_validate_json(identity), guard
            )
            response = asdict(result)
        except CleanExecutionError as exc:
            response = {"error": exc.code.value}
        except (OSError, RuntimeError, ValueError):
            # Decoder exceptions can include private paths or malformed file data.
            response = {"error": ErrorCode.INVALID_AUDIO.value}
        sys.stderr.write(json.dumps(response, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    InspectionWorker.main()
