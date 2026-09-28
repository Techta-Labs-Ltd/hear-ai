"""Verify authorised source bytes before expensive model execution."""

import hashlib
from pathlib import Path

from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class SourceIntegrity:
    @staticmethod
    def verify(path: Path, expected: str | None) -> None:
        if expected is None:
            return
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise CleanExecutionError(
                ErrorCode.SOURCE_MISMATCH, "downloaded_source_digest_mismatch"
            )
