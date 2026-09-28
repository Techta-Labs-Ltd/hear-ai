import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROLE_GROUPS = {
    "pipeline": "pipeline",
    "transcription": "transcription",
    "reconstruction": "reconstruction",
    "magic_clean_natural": "magic-clean-natural",
}


class RuntimeSetup:
    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser(
            description="Install a locked Hear AI role environment and apply its dependency patches"
        )
        parser.add_argument("--role", choices=ROLE_GROUPS, required=True)
        parser.add_argument("--provider", choices=("pod", "serverless"), default="pod")
        parser.add_argument("--feature", choices=("qwen_llm",), action="append", default=[])
        parser.add_argument("--check", action="store_true")
        parser.add_argument("--no-dev", action="store_true")
        args = parser.parse_args()
        if args.feature and args.role != "pipeline":
            parser.error("qwen_llm is available only for the pipeline role")

        root = Path(__file__).resolve().parents[1]
        project = root / "deploy" / "runtime"
        environment = dict(os.environ)
        suffix = args.role if args.provider == "pod" else f"{args.role}-serverless"
        environment.setdefault("UV_PROJECT_ENVIRONMENT", f"/opt/hear-ai-v11/venvs/{suffix}")
        uv = shutil.which("uv")
        if uv is None:
            parser.exit(1, "uv must be installed before runtime setup\n")

        try:
            if not args.check:
                install = [
                    uv,
                    "sync",
                    "--project",
                    str(project),
                    "--locked",
                    "--no-default-groups",
                    "--group",
                    ROLE_GROUPS[args.role],
                    "--group",
                    args.provider,
                ]
                if not args.no_dev:
                    install.extend(("--group", "dev"))
                for feature in args.feature:
                    if feature == "qwen_llm":
                        install.extend(("--group", "pipeline-llm"))
                subprocess.run(install, cwd=root, check=True, env=environment)

            if args.role == "reconstruction":
                fish_root = Path(environment.get("FISH_SPEECH_HOME", "/fish-speech"))
                revision = subprocess.run(
                    ["git", "-C", str(fish_root), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
                if revision != "fc4e1e24ff3b8d7d28fdd66e6789f23acb63c5bb":
                    raise RuntimeError("fish_source_revision_mismatch")
                python = str(Path(environment["UV_PROJECT_ENVIRONMENT"]) / "bin" / "python")
                if not args.check:
                    subprocess.run(
                        [
                            uv,
                            "pip",
                            "install",
                            "--python",
                            python,
                            "--no-deps",
                            "-e",
                            str(fish_root),
                        ],
                        cwd=root,
                        check=True,
                        env=environment,
                    )
                subprocess.run(
                    [python, "-c", "from fish_speech.inference_engine import TTSInferenceEngine"],
                    cwd=root,
                    check=True,
                    env=environment,
                )

            if args.role in {"pipeline", "transcription"}:
                command = [
                    uv,
                    "run",
                    "--project",
                    str(project),
                    "--no-sync",
                    "python",
                    "-m",
                    "hear.tools.dependency_patches",
                ]
                if args.check:
                    command.append("--check")
                subprocess.run(command, cwd=root, check=True, env=environment)
        except subprocess.CalledProcessError as exc:
            return exc.returncode
        return 0


if __name__ == "__main__":
    raise SystemExit(RuntimeSetup.main())
