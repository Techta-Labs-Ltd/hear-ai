import asyncio

from sqlalchemy.exc import OperationalError

db_write_lock = asyncio.Lock()


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
        except OperationalError as exc:
            await loop.run_in_executor(None, _rollback)
            if "database is locked" in str(exc).lower() and attempt < retries - 1:
                await asyncio.sleep(0.04 * (2 ** min(attempt, 10)))
                continue
            raise
