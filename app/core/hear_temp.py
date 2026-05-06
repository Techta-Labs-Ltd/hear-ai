import os
import time
import tempfile

from app.config import settings


def hear_temp_directory() -> str:
    custom = (settings.HEAR_TMP_DIR or "").strip()
    if custom:
        path = os.path.abspath(os.path.expanduser(custom))
    else:
        path = os.path.join(tempfile.gettempdir(), "hear-ai")
    os.makedirs(path, exist_ok=True)
    return path


def sweep_orphan_hear_temp_files() -> int:
    root = hear_temp_directory()
    if not os.path.isdir(root):
        return 0
    max_age = max(3600, int(settings.HEAR_TEMP_RETENTION_SECONDS))
    cutoff = time.time() - max_age
    removed = 0
    for name in os.listdir(root):
        path = os.path.join(root, name)
        try:
            if not os.path.isfile(path):
                continue
            if os.path.getmtime(path) > cutoff:
                continue
            os.unlink(path)
            removed += 1
        except OSError:
            pass
    return removed
