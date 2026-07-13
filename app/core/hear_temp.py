import os
import shutil
import tempfile


def hear_temp_directory() -> str:
    path = os.path.join(tempfile.gettempdir(), "hear-ai")
    os.makedirs(path, exist_ok=True)
    return path


def hear_temp_job_dir(job_id: str, run_id: str) -> str:
    path = os.path.join(hear_temp_directory(), job_id, run_id)
    os.makedirs(path, exist_ok=True)
    return path


def register_temp_standalone(path: str, **kw) -> None:
    pass


def drop_temp_standalone(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        if os.path.isfile(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def cleanup_job_temp(db, job_id: str, run_id: str | None = None) -> None:
    try:
        if run_id:
            dir_path = os.path.join(hear_temp_directory(), job_id, run_id)
            shutil.rmtree(dir_path, ignore_errors=True)
        else:
            base = os.path.join(hear_temp_directory(), job_id)
            shutil.rmtree(base, ignore_errors=True)
    except OSError:
        pass


def sweep_tracked_temp_files() -> dict:
    return {"by_job": 0, "by_age": 0, "orphan_fs": 0, "bytes_freed": 0}
