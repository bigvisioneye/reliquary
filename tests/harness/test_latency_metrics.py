from harness.latency_metrics import percentile, suggest_parallel_workers


def test_percentile_p90() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert percentile(values, 0.5) == 5.5
    assert percentile(values, 0.9) == 9.1


def test_suggest_parallel_workers() -> None:
    # 45s window, 15s cycle p90 -> 3 cycles -> ceil(8/3) = 3 workers
    assert suggest_parallel_workers(15.0, window_seconds=45.0) == 3


def test_suggest_parallel_workers_fast_cycle() -> None:
    assert suggest_parallel_workers(5.0, window_seconds=45.0) == 1
