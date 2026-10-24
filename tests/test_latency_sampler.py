import json

import pytest

from sim.latency import LatencySampler


def test_joint_samples_are_seeded_and_keep_measured_relationships(tmp_path):
    profile = tmp_path / "latency.json"
    profile.write_text(json.dumps({"traces": [{"samples": [
        {"request_to_input_ms": 610, "ready_to_input_ms": 250, "capture_ms": 360},
        {"request_to_input_ms": 700, "ready_to_input_ms": 280, "capture_ms": 420},
    ]}]}))
    left, right = (LatencySampler(profile, seed=7) for _ in range(2))
    values = [left.sample() for _ in range(20)]
    assert values == [right.sample() for _ in range(20)]
    assert all(v.request_to_input_ms == v.ready_to_input_ms + v.capture_ms for v in values)
    assert all(v.request_to_input_ms > 500 for v in values)
    # The first eight draws are one contiguous circular block, not IID noise.
    assert all(values[i] != values[i+1] for i in range(7))


def test_invalid_block_size_rejected():
    with pytest.raises(ValueError, match="positive"):
        LatencySampler(block_size=0)
