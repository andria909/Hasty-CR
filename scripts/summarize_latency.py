"""Summarize per-frame live timings; placement rows are never mixed with holds."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def describe(values):
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=float)
    return {
        "n": len(values), "mean": float(a.mean()),
        "std_population": float(a.std()), "min": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)), "max": float(a.max()),
    }


def summarize(rows):
    actions = [r for r in rows if r["status"] == "action_sent"]
    decisions = [r for r in rows if r["status"] == "decision"]
    keys = ("capture_ms", "perception_ms", "checks_ms", "policy_ms",
            "after_capture_ms", "total_ms", "select_card_ms", "between_taps_ms",
            "placement_command_ms", "decision_to_input_ms",
            "capture_start_to_input_ms", "capture_ready_to_input_ms")

    def group(samples):
        return {key: describe([r[key] for r in samples]) for key in keys
                if samples and key in samples[0]}

    return {
        "units": "milliseconds",
        "placement_boundary": "host ADB input tap process returned; not visual troop spawn",
        "std_definition": "population std, ddof=0; percentiles use numpy linear interpolation",
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "first_timestamp": rows[0]["timestamp"] if rows else None,
        "last_timestamp": rows[-1]["timestamp"] if rows else None,
        "actions": group(actions),
        "by_match": {str(m): group([r for r in actions if r["match"] == m])
                     for m in sorted({r["match"] for r in actions})},
        "decisions": group(decisions),
        "hold_decisions": group([r for r in decisions if r["kind"] == "hold"]),
        "by_card": {c: group([r for r in actions if r["card"] == c])
                    for c in sorted({r["card"] for r in actions})},
        "slowest_actions": sorted(actions, key=lambda r: r["capture_start_to_input_ms"],
                                  reverse=True)[:10],
    }


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actions = [r for r in rows if r["status"] == "action_sent"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for key, label, color in [
        ("capture_start_to_input_ms", "Capture request to ADB tap completion", "#2463b3"),
        ("capture_ready_to_input_ms", "Image available to ADB tap completion", "#db8b24"),
    ]:
        axes[0].hist([r[key] for r in actions], bins=25, alpha=.65,
                     label=label, color=color)
    axes[0].set(xlabel="Latency (ms)", ylabel="Card placements",
                title=f"Live MuMu timings: {len(actions)} placements")
    axes[0].legend(fontsize=8)
    for match in sorted({r["match"] for r in actions}):
        selected = [r for r in actions if r["match"] == match]
        axes[1].plot([r["elapsed_s"] for r in selected],
                     [r["capture_start_to_input_ms"] for r in selected],
                     ".-", alpha=.7, label=f"Match {match}")
    axes[1].set(xlabel="Seconds since profiler detected match",
                ylabel="Capture request to ADB tap completion (ms)",
                title="Placement latency over each match")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    result = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot(rows, args.plot)
    for match, values in result["by_match"].items():
        print("match", match, values["capture_start_to_input_ms"])
    print("all", result["actions"].get("capture_start_to_input_ms"))


if __name__ == "__main__":
    main()
