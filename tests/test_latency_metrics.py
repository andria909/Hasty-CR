import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "latency_metrics", Path(__file__).resolve().parents[1] / "scripts/latency_metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def test_timing_boundaries_and_decision_groups():
    for decision, kind in [(None, "hold"),
                           (SimpleNamespace(card="cannon", tag="rl"), "play")]:
        sample = metrics.decision_sample(
            (10, 10.2, 10.3, 10.31, 10.315),
            match=2, tick=100, elapsed_s=42, decision=decision)
        assert sample["kind"] == kind
        assert sample["total_ms"] == pytest.approx(315)
        assert sample["after_capture_ms"] == pytest.approx(115)
        assert sum(sample[key] for key in
                   ("capture_ms", "perception_ms", "checks_ms", "policy_ms")) == pytest.approx(315)
        assert sample["card"] == (None if decision is None else "cannon")


def test_clock_regression_is_rejected():
    with pytest.raises(ValueError, match="monotonic"):
        metrics.decision_sample((0, 2, 1, 3, 4), match=1, tick=1,
                                elapsed_s=0, decision=None)


def test_blocked_frame_is_not_counted_as_a_hold_decision():
    sample = metrics.decision_sample(
        (0, .1, .2, .21, .21), match=1, tick=0, elapsed_s=0,
        decision=None, status="guard_blocked")
    assert sample["kind"] is None
    assert sample["status"] == "guard_blocked"
    assert sample["policy_ms"] == 0


def test_placement_includes_both_commands_and_inter_tap_wait():
    sample = metrics.placement_sample(
        (0, .2, .3, .31, .315), (.316, .35, .43, .47),
        match=1, tick=3, elapsed_s=2,
        decision=SimpleNamespace(card="hog_rider", tag="rl"))
    assert sample["status"] == "action_sent"
    assert sample["capture_start_to_input_ms"] == pytest.approx(470)
    assert sample["capture_ready_to_input_ms"] == pytest.approx(270)
    assert sample["between_taps_ms"] == pytest.approx(80)
    assert sample["decision_to_input_ms"] == pytest.approx(155)
    assert sample["total_ms"] == pytest.approx(315)


def test_placement_cannot_precede_decision():
    with pytest.raises(ValueError, match="monotonically"):
        metrics.placement_sample(
            (0, .2, .3, .31, .315), (.3, .35, .43, .47),
            match=1, tick=3, elapsed_s=2, decision=None)


def test_summary_does_not_double_count_decision_and_input_rows():
    summary_spec = importlib.util.spec_from_file_location(
        "summarize_latency", Path(__file__).resolve().parents[1] / "scripts/summarize_latency.py")
    summary = importlib.util.module_from_spec(summary_spec)
    summary_spec.loader.exec_module(summary)
    kwargs = dict(match=1, tick=3, elapsed_s=2,
                  decision=SimpleNamespace(card="hog_rider", tag="rl"))
    rows = [metrics.decision_sample((0, .2, .3, .31, .315), **kwargs),
            metrics.placement_sample((0, .2, .3, .31, .315),
                                     (.316, .35, .43, .47), **kwargs)]
    result = summary.summarize(rows)
    assert result["actions"]["capture_start_to_input_ms"]["n"] == 1
    assert result["decisions"]["total_ms"]["n"] == 1
    assert result["actions"]["capture_start_to_input_ms"]["mean"] == pytest.approx(470)
    stats = summary.describe([100, 200, 300])
    assert stats["mean"] == 200
    assert stats["std_population"] == pytest.approx(81.649658)
    assert stats["p95"] == 290
