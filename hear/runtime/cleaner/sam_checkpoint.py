import hashlib
import importlib
import pickle
from pathlib import Path

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class SamCheckpointLoader:
    @staticmethod
    def load(
        path: Path,
        *,
        sha256: str,
        core,
        codec,
        optional_keys: frozenset[str],
        guard: ResourceGuard,
    ) -> tuple[int, int]:
        """Verify every checkpoint key before strict CPU assignment.

        optional_keys must come from the trusted pinned checkpoint manifest, never
        be derived from the file being admitted. Only vision_encoder keys may be
        omitted. The factory must discard constructed modules if loading fails.
        """
        guard.check()
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(char not in "0123456789abcdef" for char in sha256)
            or not isinstance(optional_keys, frozenset)
            or not optional_keys
            or any(
                not isinstance(key, str) or not key.startswith("vision_encoder.")
                for key in optional_keys
            )
        ):
            raise ValueError("SAM requires an exact pinned checkpoint/optional-key manifest")
        try:
            if path.is_symlink() or not path.is_file():
                raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM checkpoint missing")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    guard.check()
                    digest.update(chunk)
            if digest.hexdigest() != sha256:
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "SAM checkpoint hash mismatch"
                )
            torch = importlib.import_module("torch")
            core_state, codec_state = core.state_dict(), codec.state_dict()
            expected = dict(core_state)
            expected.update({"audio_codec." + key: value for key, value in codec_state.items()})
            if (
                not core_state
                or not codec_state
                or len(expected) != len(core_state) + len(codec_state)
                or set(expected) & optional_keys
                or any(
                    value.device.type not in ("cpu", "meta") or value.dtype != torch.float32
                    for value in expected.values()
                )
            ):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "invalid SAM destination modules"
                )
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if not isinstance(state, dict) or set(state) != set(expected) | optional_keys:
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "SAM checkpoint key mismatch"
                )
            for key, value in state.items():
                guard.check()
                if (
                    not isinstance(value, torch.Tensor)
                    or value.device.type != "cpu"
                    or value.dtype != torch.float32
                ):
                    raise CleanExecutionError(
                        ErrorCode.ENGINE_UNAVAILABLE, "invalid SAM checkpoint tensor"
                    )
                if key in expected and value.shape != expected[key].shape:
                    raise CleanExecutionError(
                        ErrorCode.ENGINE_UNAVAILABLE, "SAM checkpoint shape mismatch"
                    )
            guard.check()
            core.load_state_dict({key: state[key] for key in core_state}, strict=True, assign=True)
            codec.load_state_dict(
                {key: state["audio_codec." + key] for key in codec_state}, strict=True, assign=True
            )
            core.eval()
            codec.eval()
            guard.check()
            return len(core_state), len(codec_state)
        except CleanExecutionError:
            raise
        except MemoryError:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "SAM checkpoint allocation failed"
            ) from None
        except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError):
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "SAM checkpoint loading failed"
            ) from None
