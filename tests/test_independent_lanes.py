from hear.services.jobs.scheduler import FairJobScheduler, PendingJob


def test_magic_clean_and_reconstruction_use_independent_single_job_lanes() -> None:
    scheduler = FairJobScheduler(
        max_active=3,
        max_active_per_user=3,
        type_limits={"magic_clean": 1, "reconstruction": 1},
    )
    magic_clean = PendingJob("magic", "run-1", "user", "magic_clean")
    reconstruct = PendingJob("reconstruct", "run-2", "user", "reconstruct", "reconstruction")
    edit_transcript = PendingJob("edit", "run-3", "user", "edit_transcript", "reconstruction")

    assert scheduler.enqueue(magic_clean)
    assert scheduler.enqueue(reconstruct)
    assert scheduler.enqueue(edit_transcript)
    assert scheduler.pop_next() == magic_clean
    assert scheduler.pop_next() == reconstruct
    assert scheduler.pop_next() is None

    scheduler.complete(reconstruct)
    assert scheduler.pop_next() == edit_transcript
