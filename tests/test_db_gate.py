import asyncio
import threading

import pytest
from sqlalchemy.exc import OperationalError

from hear.core.db_gate import DatabaseCommitter


class _SerializationFailure(Exception):
    pgcode = "40001"


class _Session:
    def __init__(self, *, fail_first_commit: bool = False) -> None:
        self.pending = ["mutation"]
        self.persisted: list[str] = []
        self.fail_first_commit = fail_first_commit
        self.commit_calls = 0
        self.rollback_calls = 0

    def commit(self) -> None:
        self.commit_calls += 1
        if self.fail_first_commit and self.commit_calls == 1:
            raise OperationalError("COMMIT", {}, _SerializationFailure("serialization failure"))
        self.persisted.extend(self.pending)
        self.pending.clear()

    def rollback(self) -> None:
        self.rollback_calls += 1
        self.pending.clear()


def test_commit_with_retry_commits_staged_mutations() -> None:
    session = _Session()
    asyncio.run(DatabaseCommitter.commit_with_retry(session))
    assert session.commit_calls == 1
    assert session.rollback_calls == 0
    assert session.persisted == ["mutation"]


def test_commit_with_retry_never_commits_empty_transaction_after_transient_failure() -> None:
    session = _Session(fail_first_commit=True)
    with pytest.raises(OperationalError, match="serialization failure"):
        asyncio.run(DatabaseCommitter.commit_with_retry(session, retries=3))
    assert session.commit_calls == 1
    assert session.rollback_calls == 1
    assert session.pending == []
    assert session.persisted == []


async def _exercise_commit_cancellation_waits_for_dbapi_worker() -> None:
    started = threading.Event()
    release = threading.Event()
    session = _Session()
    original_commit = session.commit

    def delayed_commit() -> None:
        started.set()
        assert release.wait(timeout=5)
        original_commit()

    session.commit = delayed_commit
    task = asyncio.create_task(DatabaseCommitter.commit_with_retry(session))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert session.commit_calls == 1
    assert session.persisted == ["mutation"]


def test_commit_cancellation_does_not_race_the_dbapi_worker() -> None:
    asyncio.run(_exercise_commit_cancellation_waits_for_dbapi_worker())
