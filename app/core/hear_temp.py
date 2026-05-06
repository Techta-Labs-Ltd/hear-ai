import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import text

from app.config import settings
from app.models.database import AiJob, AiTempFile, SessionLocal, engine


def hear_temp_directory() -> str:
    custom = (settings.HEAR_TMP_DIR or "").strip()
    if custom:
        path = os.path.abspath(os.path.expanduser(custom))
    else:
        path = os.path.join(tempfile.gettempdir(), "hear-ai")
    os.makedirs(path, exist_ok=True)
    return path


def hear_temp_jobs_root() -> str:
    path = os.path.join(hear_temp_directory(), "jobs")
    os.makedirs(path, exist_ok=True)
    return path


def hear_temp_job_dir(job_id: str, run_id: Optional[str]) -> str:
    safe_job = (job_id or "no-job").replace(os.sep, "_")
    safe_run = (run_id or "no-run").replace(os.sep, "_")
    path = os.path.join(hear_temp_jobs_root(), safe_job, safe_run)
    os.makedirs(path, exist_ok=True)
    return path


def register_temp(
    db,
    path: str,
    *,
    purpose: str,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    track_id: Optional[str] = None,
) -> None:
    if not path:
        return
    try:
        size = os.path.getsize(path) if os.path.exists(path) else None
    except OSError:
        size = None
    try:
        row = AiTempFile(
            job_id=job_id,
            run_id=run_id,
            track_id=track_id,
            purpose=purpose,
            path=path,
            size_bytes=size,
        )
        db.add(row)
        db.flush()
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        print(f"[TEMP] register_temp failed for {path}: {exc}")


def register_temp_standalone(
    path: str,
    *,
    purpose: str,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    track_id: Optional[str] = None,
) -> None:
    db = SessionLocal()
    try:
        register_temp(
            db,
            path,
            purpose=purpose,
            job_id=job_id,
            run_id=run_id,
            track_id=track_id,
        )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _unlink_quiet(path: str) -> bool:
    if not path:
        return False
    try:
        if os.path.isfile(path):
            os.unlink(path)
            return True
    except OSError:
        return False
    return False


def drop_temp(db, path: str) -> None:
    _unlink_quiet(path)
    try:
        db.query(AiTempFile).filter(AiTempFile.path == path).delete(
            synchronize_session=False
        )
        db.flush()
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        print(f"[TEMP] drop_temp failed for {path}: {exc}")


def drop_temp_standalone(path: str) -> None:
    _unlink_quiet(path)
    db = SessionLocal()
    try:
        db.query(AiTempFile).filter(AiTempFile.path == path).delete(
            synchronize_session=False
        )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def cleanup_job_temp(db, job_id: Optional[str], run_id: Optional[str]) -> int:
    if not job_id:
        return 0
    removed = 0
    job_dir = os.path.join(
        hear_temp_jobs_root(),
        job_id.replace(os.sep, "_"),
        (run_id or "no-run").replace(os.sep, "_"),
    )
    if os.path.isdir(job_dir):
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
        except OSError:
            pass
    try:
        rows = (
            db.query(AiTempFile.path)
            .filter(AiTempFile.job_id == job_id)
            .filter(AiTempFile.run_id == run_id)
            .all()
        )
        for r in rows:
            if _unlink_quiet(r.path):
                removed += 1
        db.query(AiTempFile).filter(
            AiTempFile.job_id == job_id, AiTempFile.run_id == run_id
        ).delete(synchronize_session=False)
        db.flush()
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        print(f"[TEMP] cleanup_job_temp failed for {job_id}/{run_id}: {exc}")
    return removed


@contextmanager
def tracked_tempfile(
    suffix: str,
    *,
    purpose: str,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    track_id: Optional[str] = None,
    db=None,
):
    if job_id and run_id:
        directory = hear_temp_job_dir(job_id, run_id)
    else:
        directory = hear_temp_directory()
    fd, path = tempfile.mkstemp(suffix=suffix, dir=directory)
    os.close(fd)
    if db is not None:
        register_temp(
            db, path, purpose=purpose, job_id=job_id, run_id=run_id, track_id=track_id
        )
    else:
        register_temp_standalone(
            path, purpose=purpose, job_id=job_id, run_id=run_id, track_id=track_id
        )
    try:
        yield path
    finally:
        pass


def sweep_tracked_temp_files(retention_seconds: Optional[int] = None) -> dict:
    retention = (
        max(0, int(retention_seconds))
        if retention_seconds is not None
        else max(3600, int(settings.HEAR_TEMP_RETENTION_SECONDS))
    )
    summary = {"by_job": 0, "by_age": 0, "orphan_fs": 0, "bytes_freed": 0}
    db = SessionLocal()
    try:
        terminal_rows = (
            db.query(AiTempFile.id, AiTempFile.path)
            .join(
                AiJob,
                (AiJob.id == AiTempFile.job_id) & (AiJob.run_id == AiTempFile.run_id),
            )
            .filter(AiJob.status.in_(["completed", "failed", "cancelled"]))
            .all()
        )
        if terminal_rows:
            ids_to_drop = []
            for r in terminal_rows:
                try:
                    if os.path.isfile(r.path):
                        size = os.path.getsize(r.path)
                        os.unlink(r.path)
                        summary["bytes_freed"] += size
                except OSError:
                    pass
                ids_to_drop.append(r.id)
                summary["by_job"] += 1
            db.query(AiTempFile).filter(AiTempFile.id.in_(ids_to_drop)).delete(
                synchronize_session=False
            )
            db.commit()

        cutoff_sql = text(
            "DELETE FROM ai_temp_files "
            "WHERE created_at < (NOW() - make_interval(secs => :retention)) "
            "RETURNING path"
        )
        result = db.execute(cutoff_sql, {"retention": retention})
        old_paths = [row[0] for row in result.fetchall()]
        for p in old_paths:
            try:
                if os.path.isfile(p):
                    size = os.path.getsize(p)
                    os.unlink(p)
                    summary["bytes_freed"] += size
            except OSError:
                pass
            summary["by_age"] += 1
        db.commit()

        tracked_paths = {
            r.path for r in db.query(AiTempFile.path).all() if r.path
        }
    finally:
        db.close()

    cutoff_ts = time.time() - retention
    root = hear_temp_directory()
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            if full in tracked_paths:
                continue
            try:
                if os.path.getmtime(full) > cutoff_ts:
                    continue
                size = os.path.getsize(full)
                os.unlink(full)
                summary["orphan_fs"] += 1
                summary["bytes_freed"] += size
            except OSError:
                continue

    return summary


def purge_all_temp() -> dict:
    summary = {"jobs_dir_removed": False, "rows_truncated": 0, "bytes_freed": 0}
    jobs_root = hear_temp_jobs_root()
    if os.path.isdir(jobs_root):
        for dirpath, _dirnames, filenames in os.walk(jobs_root):
            for name in filenames:
                full = os.path.join(dirpath, name)
                try:
                    summary["bytes_freed"] += os.path.getsize(full)
                except OSError:
                    pass
        shutil.rmtree(jobs_root, ignore_errors=True)
        summary["jobs_dir_removed"] = True
        os.makedirs(jobs_root, exist_ok=True)
    with engine.begin() as conn:
        result = conn.execute(text("DELETE FROM ai_temp_files"))
        summary["rows_truncated"] = result.rowcount or 0
    return summary


def sweep_orphan_hear_temp_files() -> int:
    summary = sweep_tracked_temp_files()
    return summary["by_job"] + summary["by_age"] + summary["orphan_fs"]
