import argparse
import json
import sys
from pathlib import Path

from hear.audio.workspace import AudioWorkspace
from hear.config import settings


class TempCleanupCommand:
    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser(description="Hear AI temp file cleanup")
        parser.add_argument(
            "--mode",
            choices=("startup", "periodic", "purge"),
            default="periodic",
            help="startup and periodic sweep expired workspaces; purge deletes the entire scratch root",
        )
        parser.add_argument("--yes", action="store_true", help="Required for purge mode")
        args = parser.parse_args()
        if args.mode == "purge":
            if not args.yes:
                print("Refusing purge without --yes", file=sys.stderr)
                return 2
            summary = AudioWorkspace.purge(Path(settings.HEAR_TEMP_DIR))
            print(json.dumps(summary))
            return 0
        summary = AudioWorkspace.sweep(
            Path(settings.HEAR_TEMP_DIR), settings.AUDIO_MAX_AGE_SECONDS
        )
        print(json.dumps(summary))
        return 0


if __name__ == "__main__":
    raise SystemExit(TempCleanupCommand.main())
