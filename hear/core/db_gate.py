import asyncio

from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

db_write_lock = asyncio.Lock()

TRANSIENT_PGCODES = {"40001", "40P01"}


def is_transient_db_error(exc: BaseException) -> bool:
    if isinstance(exc, InterfaceError):
        return True
    if isinstance(exc, (OperationalError, DBAPIError)):
        orig = getattr(exc, "orig", None)
        pgcode = getattr(orig, "pgcode", None) or getattr(orig, "sqlstate", None)
        if pgcode in TRANSIENT_PGCODES:
            return True
        message = str(exc).lower()
        if "server closed the connection" in message:
            return True
        if "ssl connection has been closed unexpectedly" in message:
            return True
    return False


async def commit_with_retry(db, retries: int = 12) -> None:
    loop = asyncio.get_running_loop()

    def _commit() -> None:
        db.commit()

    def _rollback() -> None:
        try:
            db.rollback()
        except Exception:
            pass

    for attempt in range(retries):
        try:
            async with db_write_lock:
                await loop.run_in_executor(None, _commit)
            return
        except Exception as exc:
            await loop.run_in_executor(None, _rollback)
            if is_transient_db_error(exc) and attempt < retries - 1:
                await asyncio.sleep(0.04 * (2 ** min(attempt, 10)))
                continue
            raise
