from hear.services.jobs.scheduler import FairJobScheduler, PendingJob


def job(name: str, user: str, job_type: str = "pipeline") -> PendingJob:
    return PendingJob(name, f"run-{name}", user, job_type)


def test_round_robin_prevents_one_user_from_filling_all_slots():
    scheduler = FairJobScheduler(max_active=3, max_active_per_user=1)
    a1, a2 = job("a1", "alice"), job("a2", "alice")
    b1, c1 = job("b1", "bob"), job("c1", "carol")

    for item in (a1, a2, b1, c1):
        assert scheduler.enqueue(item)

    assert scheduler.pop_next() == a1
    assert scheduler.pop_next() == b1
    assert scheduler.pop_next() == c1
    assert scheduler.pop_next() is None

    scheduler.complete(a1)
    assert scheduler.pop_next() == a2


def test_job_type_limit_skips_blocked_user_without_blocking_others():
    scheduler = FairJobScheduler(
        max_active=3,
        max_active_per_user=2,
        type_limits={"reconstruct": 1},
    )
    a1 = job("a1", "alice", "reconstruct")
    b1 = job("b1", "bob", "reconstruct")
    c1 = job("c1", "carol", "audio_tag")
    for item in (a1, b1, c1):
        scheduler.enqueue(item)

    assert scheduler.pop_next() == a1
    assert scheduler.pop_next() == c1
    assert scheduler.pop_next() is None

    scheduler.complete(a1)
    assert scheduler.pop_next() == b1


def test_two_magic_clean_slots_are_shared_across_users_round_robin():
    scheduler = FairJobScheduler(
        max_active=3,
        max_active_per_user=1,
        type_limits={"magic_clean": 2},
    )
    alice_first = job("a1", "alice", "magic_clean")
    alice_second = job("a2", "alice", "magic_clean")
    bob = job("b1", "bob", "magic_clean")
    for item in (alice_first, alice_second, bob):
        scheduler.enqueue(item)

    assert scheduler.pop_next() == alice_first
    assert scheduler.pop_next() == bob
    assert scheduler.pop_next() is None

    scheduler.complete(alice_first)
    assert scheduler.pop_next() == alice_second


def test_cancel_removes_queued_job_and_duplicate_is_idempotent():
    scheduler = FairJobScheduler(max_active=2, max_active_per_user=1)
    queued = job("a1", "alice")

    assert scheduler.enqueue(queued)
    assert not scheduler.enqueue(queued)
    assert scheduler.remove("a1")
    assert not scheduler.remove("a1")
    assert scheduler.queued_count == 0
    assert scheduler.pop_next() is None


def test_stats_report_users_and_active_job_types():
    scheduler = FairJobScheduler(max_active=2, max_active_per_user=1)
    a1, b1 = job("a1", "alice", "pipeline"), job("b1", "bob", "audio_tag")
    scheduler.enqueue(a1)
    scheduler.enqueue(b1)
    assert scheduler.pop_next() == a1

    assert scheduler.stats() == {
        "queued": 1,
        "active": 1,
        "active_users": 1,
        "queued_users": 1,
        "active_by_type": {"pipeline": 1},
    }
