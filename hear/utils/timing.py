import time


def elapsed_seconds(started: float) -> float:
    return round(time.perf_counter() - started, 3)
