import pytest

from hear.runtime.cleaner.metrics import StageTimings


def test_repeated_loading_spans_accumulate_without_inference(monkeypatch):
    clock = iter((0, 2_000_000_000, 3_000_000_000, 6_000_000_000, 7_000_000_000, 8_000_000_000))
    monkeypatch.setattr("hear.runtime.cleaner.metrics.time.monotonic_ns", lambda: next(clock))
    timings = StageTimings()
    with timings.measure("loading"):
        assert timings.snapshot() == {}
    with timings.measure("inference"):
        pass
    with timings.measure("loading"):
        pass
    assert timings.snapshot() == {"loading": 3, "inference": 3}
    snapshot = timings.snapshot()
    snapshot["loading"] = 99
    assert timings.snapshot()["loading"] == 3


def test_failed_span_retains_duration_and_original_exception(monkeypatch):
    clock = iter((1_000_000_000, 2_500_000_000))
    monkeypatch.setattr("hear.runtime.cleaner.metrics.time.monotonic_ns", lambda: next(clock))
    timings = StageTimings()
    failure = RuntimeError("injected stage failure")
    with pytest.raises(RuntimeError) as caught:
        with timings.measure("cleanup"):
            raise failure
    assert caught.value is failure
    assert timings.snapshot() == {"cleanup": 1.5}
    timings.reset()
    assert timings.snapshot() == {}


def test_unknown_and_overlapping_stages_are_rejected():
    timings = StageTimings()
    with pytest.raises(ValueError):
        with timings.measure("private-audio-path"):
            pytest.fail("invalid stage entered")
    with timings.measure("loading"):
        with pytest.raises(ValueError):
            with timings.measure("inference"):
                pytest.fail("overlapping stage entered")
        with pytest.raises(RuntimeError):
            timings.reset()
    assert set(timings.snapshot()) == {"loading"}


def test_instances_do_not_share_attempt_data():
    first, second = StageTimings(), StageTimings()
    with first.measure("inspection"):
        pass
    assert second.snapshot() == {}
