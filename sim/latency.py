"""Seeded joint latency sampling, independent of the environment's time step.

This does not change ClashEnv. Its present min(delay, cadence) would clip these
samples at 500 ms; integration needs a pending-action queue or a longer cadence.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import random

DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "data/latency/mumu_m5_20261021.json"


@dataclass(frozen=True)
class LatencySample:
    request_to_input_ms: float
    ready_to_input_ms: float
    capture_ms: float


class LatencySampler:
    """Circular block bootstrap: joint samples, with short-range correlation.

    Select a trace proportional to its length, then a uniform starting index.
    Every marginal sample thus has the empirical distribution. Blocks wrap
    within one trace; transitions between blocks are synthetic.
    """

    def __init__(self, profile=DEFAULT_PROFILE, *, seed=0, block_size=8):
        if block_size < 1:
            raise ValueError("block_size must be positive")
        payload = json.loads(Path(profile).read_text())
        self.traces = [trace["samples"] for trace in payload["traces"]]
        if not self.traces or any(not trace for trace in self.traces):
            raise ValueError("latency profile must contain nonempty traces")
        self.rng = random.Random(seed)
        self.block_size = block_size
        self.remaining = 0

    def sample(self):
        if not self.remaining:
            self.trace = self.rng.choices(self.traces, weights=list(map(len, self.traces)))[0]
            self.index = self.rng.randrange(len(self.trace))
            self.remaining = self.block_size
        item = self.trace[self.index]
        self.index = (self.index + 1) % len(self.trace)
        self.remaining -= 1
        return LatencySample(**item)
