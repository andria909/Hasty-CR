"""Measure policy capacity and batch-one decision latency.

The live runner already records end-to-end capture/perception/policy timings.
This isolates the neural part so a larger architecture can be accepted or
rejected before spending a training run on it.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[min(len(values) - 1, int(len(values) * fraction))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark HastyCR policies")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--architecture", choices=("compact", "wide"),
                        default="compact")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()

    import torch
    from sim.env import ACTIONS, NUM_PLANES, NUM_SCALARS
    from sim.train_ppo import (build_network, checkpoint_architecture,
                               load_policy_state)

    blob = None
    architecture = args.architecture
    if args.checkpoint:
        blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        architecture = checkpoint_architecture(blob)
    network = build_network(
        NUM_PLANES, NUM_SCALARS, ACTIONS, architecture).to(args.device).eval()
    if blob is not None:
        load_policy_state(network, blob["state_dict"])

    generator = np.random.default_rng(20261012)
    planes = torch.from_numpy(generator.normal(
        size=(1, NUM_PLANES, 32, 18)).astype(np.float32)).to(args.device)
    scalars = torch.from_numpy(generator.normal(
        size=(1, NUM_SCALARS)).astype(np.float32)).to(args.device)

    def sync() -> None:
        if args.device == "mps":
            torch.mps.synchronize()

    with torch.inference_mode():
        for _ in range(30):
            network(planes, scalars)
        sync()
        samples = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            network(planes, scalars)
            sync()
            samples.append((time.perf_counter() - started) * 1000.0)

    parameters = sum(value.numel() for value in network.parameters())
    print(f"architecture={network.architecture}")
    print(f"planes={NUM_PLANES} scalars={NUM_SCALARS} actions={ACTIONS}")
    print(f"parameters={parameters} ({parameters / 1e6:.2f}M)")
    print(f"device={args.device} iterations={args.iterations}")
    print(f"latency_ms mean={statistics.mean(samples):.3f} "
          f"p50={statistics.median(samples):.3f} "
          f"p95={percentile(samples, 0.95):.3f} max={max(samples):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
