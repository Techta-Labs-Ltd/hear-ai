import asyncio

from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

from hear.core.blocking import AsyncCompletion

db_write_lock = asyncio.Lock()
TRANSIENT_PGCODES = {"40001", "40P01"}


class DatabaseCommitter:
    @staticmethod
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

    @staticmethod
    async def commit_with_retry(db, retries: int = 12) -> None:
        """Commit the staged transaction once, rolling it back on any failure.

        ``retries`` remains accepted for compatibility with existing callers, but
        commit failures are deliberately not retried here. A rollback discards the
        session's staged mutations, so retrying only ``commit()`` can falsely report
        success after committing an empty transaction. Retrying safely requires the
        caller to replay the complete unit of work in a fresh transaction.
        """
        del retries

        def _commit() -> None:
            db.commit()

        def _rollback() -> None:
            try:
                db.rollback()
            except Exception:
                pass

        try:
            async with db_write_lock:
                await AsyncCompletion.run_blocking_to_completion(_commit)
        except BaseException:
            await AsyncCompletion.run_blocking_to_completion(_rollback)
            raise
