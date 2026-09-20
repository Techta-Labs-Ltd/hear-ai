import argparse
import shutil
import subprocess
from pathlib import Path


class RuntimeSetup:
    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser(
            description="Install locked dependencies and apply pinned patches"
        )
        parser.add_argument("--check", action="store_true")
        parser.add_argument("--no-dev", action="store_true")
        args = parser.parse_args()
        root = Path(__file__).resolve().parents[1]
        uv = shutil.which("uv")
        if uv is None:
            parser.exit(1, "uv must be installed before runtime setup\n")
        try:
            if not args.check:
                install = [uv, "sync", "--locked"]
                if args.no_dev:
                    install.append("--no-dev")
                subprocess.run(install, cwd=root, check=True)
            command = [uv, "run", "--no-sync", "python", "-m", "hear.tools.dependency_patches"]
            if args.check:
                command.append("--check")
            subprocess.run(command, cwd=root, check=True)
        except subprocess.CalledProcessError as exc:
            return exc.returncode
        return 0


if __name__ == "__main__":
    raise SystemExit(RuntimeSetup.main())
