"""Shared residency handling for attempt-local disk-backed tensors."""

import mmap

from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class MappedResidency:
    @staticmethod
    def require_supported():
        if not hasattr(mmap.mmap, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "bounded tensor residency requires madvise"
            )

    @staticmethod
    def evict(guard, *maps):
        # Flush before eviction; this is not a page-cache/cgroup memory limit.
        seen = set()
        for mapping in maps:
            if id(mapping) in seen:
                continue
            seen.add(id(mapping))
            guard.check()
            if mapping.mode != "r":
                mapping.flush()
            mapping._mmap.madvise(mmap.MADV_DONTNEED)
            guard.check()
