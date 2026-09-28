"""Run a complete recording through the production DeepFilterNet preset cleaner."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hear.contracts.cleaning import CleaningProfiles, MagicCleanProfile
from hear.runtime.cleaner.deepfilter_available import DeepFilterNetCleaner
from hear.runtime.cleaner.resource_guard import ResourceBudget
from hear.services.sound_cleanup.analysis import SoundAnalyser
from hear.services.sound_cleanup.assets import SoundCleanupAssets
from hear.services.sound_cleanup.separator import EventSeparator
from hear.services.sound_cleanup.service import SoundCleanupService


class LocalMagicCleanCli:
    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("source", type=Path)
        parser.add_argument(
            "--profile", choices=[p.value for p in MagicCleanProfile], default="studio_voice"
        )
        parser.add_argument("--model-root", type=Path, default=Path("/models"))
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
        parser.add_argument("--auto-level", action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument("--remove-clicks", action="store_true")
        parser.add_argument("--reduce-stationary-noise", action="store_true")
        parser.add_argument("--trim-silence", action="store_true")
        parser.add_argument("--sound-cleanup-options", type=Path)
        parser.add_argument("--sound-cleanup-bundle", type=Path)
        parser.add_argument("--sound-cleanup-bundle-sha256")
        parser.add_argument("--separator-bundle", type=Path)
        parser.add_argument("--separator-sha256")
        parser.add_argument("--timeout-seconds", type=int, default=1800)
        args = parser.parse_args()
        source = args.source.resolve(strict=True)
        if not source.is_file() or args.timeout_seconds <= 0:
            parser.error("source must be a regular file and timeout must be positive")
        options = {
            "profile": args.profile,
            "remove_clicks": args.remove_clicks,
            "trim_silence": args.trim_silence,
        }
        if args.auto_level is not None:
            options["auto_level"] = args.auto_level
        options["reduce_stationary_noise"] = args.reduce_stationary_noise
        sound_service = None
        if args.sound_cleanup_options or args.reduce_stationary_noise:
            if args.sound_cleanup_options and args.sound_cleanup_options.stat().st_size > 65536:
                parser.error("sound-cleanup option file is too large")
            if args.sound_cleanup_options:
                options["sound_cleanup"] = json.loads(args.sound_cleanup_options.read_text())
            if not args.sound_cleanup_bundle or not args.sound_cleanup_bundle_sha256:
                parser.error("a pinned sound-cleanup bundle is required")
            separator = (
                EventSeparator(args.separator_bundle, args.separator_sha256 or "", args.device)
                if args.separator_bundle
                else None
            )
            sound_service = SoundCleanupService(
                SoundAnalyser(
                    SoundCleanupAssets.load(
                        args.sound_cleanup_bundle, args.sound_cleanup_bundle_sha256
                    ),
                    device=args.device,
                ),
                separator=separator,
            )
        options = CleaningProfiles.validate(options)
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        names = ("cleaned_master.flac", "delivery_audio.mp3", "validation.json")
        if any((output / name).exists() for name in names):
            parser.error("output already exists; choose a new output directory")
        cleaner = DeepFilterNetCleaner(
            Path(__file__).resolve().parents[1] / "deploy/cleaner/deepfilter3.ini",
            args.model_root.resolve() / "magic-clean/DeepFilterNet3",
            ResourceBudget(20_000_000_000, 2_000_000_000, 48000 * 7200),
            device=args.device,
            sound_cleanup_service=sound_service,
        )
        try:
            if not cleaner.is_ready():
                raise RuntimeError("pinned DeepFilterNet assets/runtime are not ready")
            with tempfile.TemporaryDirectory(prefix=".cleaning-", dir=output) as raw:
                workspace = Path(raw)
                report = cleaner.clean(
                    source,
                    workspace / names[0],
                    workspace,
                    options,
                    datetime.now(UTC) + timedelta(seconds=args.timeout_seconds),
                    args.timeout_seconds,
                )
                (workspace / names[2]).write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
                for name in names:
                    # Exclusive creation also prevents a concurrent CLI invocation
                    # from overwriting a candidate created since the initial check.
                    with (workspace / name).open("rb") as src, (output / name).open("xb") as dst:
                        shutil.copyfileobj(src, dst)
        finally:
            cleaner.close()
        print(
            json.dumps(
                {"output_dir": str(output), "profile": args.profile, "files": list(names)}, indent=2
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(LocalMagicCleanCli.main())
