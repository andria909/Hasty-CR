"""Build a conservative policy from small checkpoint task vectors.

For a baseline ``B`` and candidates ``C_i`` this writes

    B + sum(weight_i * (C_i - B))

This is different from averaging complete policies: every weight says how far
to move from the known-good baseline toward one independently useful run.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def task_vector_soup(base: dict, candidates: list[dict],
                     weights: list[float]) -> dict:
    """Return a blended state dict after strict compatibility checks."""
    if len(candidates) != len(weights):
        raise ValueError("candidates and weights must have equal length")
    if not candidates:
        raise ValueError("at least one candidate is required")

    result = {}
    base_keys = set(base)
    for index, candidate in enumerate(candidates):
        if set(candidate) != base_keys:
            missing = sorted(base_keys - set(candidate))
            extra = sorted(set(candidate) - base_keys)
            raise ValueError(
                f"candidate {index} has incompatible keys: "
                f"missing={missing}, extra={extra}")

    for name, origin in base.items():
        for index, candidate in enumerate(candidates):
            other = candidate[name]
            if other.shape != origin.shape or other.dtype != origin.dtype:
                raise ValueError(
                    f"candidate {index} tensor {name!r} is incompatible")
        if origin.is_floating_point():
            blended = origin.clone()
            for candidate, weight in zip(candidates, weights):
                blended.add_(candidate[name] - origin, alpha=weight)
            result[name] = blended
        else:
            if any(not candidate[name].equal(origin)
                   for candidate in candidates):
                raise ValueError(f"non-floating tensor {name!r} differs")
            result[name] = origin.clone()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="blend checkpoint task vectors")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=float, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate) != len(args.weight):
        parser.error("each --candidate needs one --weight")
    if any(abs(weight) > 1.0 for weight in args.weight):
        parser.error("task-vector weights must be between -1 and 1")
    for path in [args.baseline, *args.candidate]:
        if not path.is_file():
            parser.error(f"checkpoint does not exist: {path}")

    import torch

    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    candidates = [torch.load(path, map_location="cpu", weights_only=False)
                  for path in args.candidate]
    blob = dict(baseline)
    blob["state_dict"] = task_vector_soup(
        baseline["state_dict"], [item["state_dict"] for item in candidates],
        args.weight)
    blob["task_vector_soup"] = {
        "baseline": str(args.baseline),
        "candidates": [str(path) for path in args.candidate],
        "weights": args.weight,
    }
    # Optimiser moments no longer correspond to the blended weights.
    blob.pop("optimiser", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
