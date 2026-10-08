import pytest

from igsr.compute_profiler import ComputeProfiler


def test_initialisation_sets_all_metrics_to_zero():
    """All registered metrics start at 0.0."""
    metrics = {"wall_clock_optim", "num_steps"}
    profiler = ComputeProfiler(metrics)
    assert profiler.as_dict() == {"wall_clock_optim": 0.0, "num_steps": 0.0}


def test_accumulate_adds_values_correctly():
    """Successive calls should sum."""
    profiler = ComputeProfiler({"m"})
    profiler.accumulate("m", 1)  # implicit float conversion
    profiler.accumulate("m", 0.5)
    assert profiler.as_dict()["m"] == pytest.approx(1.5)


def test_accumulate_raises_for_unknown_metric():
    """Using a metric that was not declared should error."""
    profiler = ComputeProfiler({"x"})
    with pytest.raises(KeyError):
        profiler.accumulate("y", 42)


def test_as_dict_returns_a_copy():
    """
    External mutation of the returned dict must not affect the
    internal state of the profiler.
    """
    profiler = ComputeProfiler({"x"})
    ext = profiler.as_dict()
    ext["x"] = 99
    assert profiler.as_dict()["x"] == 0.0
