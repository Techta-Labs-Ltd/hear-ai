"""Compatibility imports for cancellation-safe execution helpers."""

from hear.core.blocking import (
    run_awaitable_to_completion,
    run_blocking_to_completion,
)

__all__ = ["run_awaitable_to_completion", "run_blocking_to_completion"]
