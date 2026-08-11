import argparse
import json
import sys
from hear.core.hear_temp import purge_all_temp, sweep_tracked_temp_files

def main() -> int:
    parser = argparse.ArgumentParser(description="Hear AI temp file cleanup")
    parser.add_argument(
        "--mode",
        choices=("startup", "periodic", "purge"),
        default="periodic",
        help="startup and periodic both run the tracked sweep; purge wipes jobs/* and ai_temp_files",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required for purge mode",
    )
    args = parser.parse_args()

    if args.mode == "purge":
        if not args.yes:
            print("Refusing purge without --yes", file=sys.stderr)
            return 2
        summary = purge_all_temp()
        print(json.dumps(summary))
        return 0

    summary = sweep_tracked_temp_files()
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
