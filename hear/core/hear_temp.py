import os
import shutil
import tempfile
import time
from hashlib import sha256

from hear.config import settings


def hear_temp_directory() -> str:
    path = settings.HEAR_TEMP_DIR
    os.makedirs(path, exist_ok=True)
    return path


def hear_temp_managed_directory() -> str:
    path = os.path.join(hear_temp_directory(), "managed")
    os.makedirs(path, exist_ok=True)
    return path


def hear_temp_job_dir(job_id: str, run_id: str) -> str:
    path = os.path.join(
        hear_temp_managed_directory(),
        "jobs",
        _safe_component(job_id),
        _safe_component(run_id),
    )
    os.makedirs(path, exist_ok=True)
    return path


def hear_temp_standalone_dir(purpose: str = "audio") -> str:
    safe_purpose = "".join(c for c in purpose if c.isalnum() or c in "_-") or "audio"
    root = os.path.join(hear_temp_managed_directory(), "standalone")
    os.makedirs(root, exist_ok=True)
    return tempfile.mkdtemp(prefix="item_", suffix=f"_{safe_purpose}", dir=root)


def drop_temp_standalone(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        if os.path.isfile(path):
            os.unlink(path)
            parent = os.path.dirname(path)
            standalone_root = os.path.join(hear_temp_managed_directory(), "standalone")
            if os.path.dirname(parent) == standalone_root:
                try:
                    os.rmdir(parent)
                except OSError:
                    pass
        elif os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def cleanup_job_temp(db, job_id: str, run_id: str | None = None) -> None:
    try:
        if run_id:
            dir_path = os.path.join(
                hear_temp_managed_directory(),
                "jobs",
                _safe_component(job_id),
                _safe_component(run_id),
            )
            shutil.rmtree(dir_path, ignore_errors=True)
            try:
                os.rmdir(os.path.dirname(dir_path))
            except OSError:
                pass
        else:
            base = os.path.join(
                hear_temp_managed_directory(),
                "jobs",
                _safe_component(job_id),
            )
            shutil.rmtree(base, ignore_errors=True)
    except OSError:
        pass


def sweep_tracked_temp_files() -> dict:
    root = hear_temp_managed_directory()
    cutoff = time.time() - settings.AUDIO_MAX_AGE_SECONDS
    removed = 0
    bytes_freed = 0
    for category in ("jobs", "standalone"):
        category_path = os.path.join(root, category)
        if not os.path.isdir(category_path):
            continue
        for entry in os.scandir(category_path):
            try:
                if entry.stat(follow_symlinks=False).st_mtime >= cutoff:
                    continue
                size = _path_size(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path, ignore_errors=True)
                else:
                    os.unlink(entry.path)
                removed += 1
                bytes_freed += size
            except OSError:
                continue
    temp_root = hear_temp_directory()
    for entry in os.scandir(temp_root):
        try:
            if not entry.is_file(follow_symlinks=False):
                continue
            if entry.stat(follow_symlinks=False).st_mtime >= cutoff:
                continue
            size = _path_size(entry.path)
            os.unlink(entry.path)
            removed += 1
            bytes_freed += size
        except OSError:
            continue
    return {"by_job": 0, "by_age": removed, "orphan_fs": removed, "bytes_freed": bytes_freed}


def purge_all_temp() -> dict:
    root = hear_temp_managed_directory()
    removed = 0
    bytes_freed = 0
    for entry in os.scandir(root):
        try:
            bytes_freed += _path_size(entry.path)
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path, ignore_errors=True)
            else:
                os.unlink(entry.path)
            removed += 1
        except OSError:
            continue
    return {"removed": removed, "bytes_freed": bytes_freed}


def _path_size(path: str) -> int:
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for directory, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(directory, name))
            except OSError:
                pass
    return total


def _safe_component(value: str) -> str:
    raw = str(value)
    readable = "".join(c if c.isalnum() or c in "_.-" else "_" for c in raw)[:64]
    return f"{readable or 'item'}-{sha256(raw.encode()).hexdigest()[:12]}"
