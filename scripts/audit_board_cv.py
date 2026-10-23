"""Paired offline comparison of original and corrected live board perception."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from live_perception import BoardUnitDetector, box_iou


def serialise(result):
    return [{"name": d.unit.name, "side": side,
             "bbox_small": list(d.position.bbox), "confidence": float(d.position.conf),
             "tile": [d.position.tile_x, d.position.tile_y]}
            for side, detections in zip(("ally", "enemy"), result) for d in detections]


def score(predicted, expected):
    """One-to-one class + side + IoU>=.5 matching; wrong labels count FP and FN."""
    remaining = set(range(len(expected)))
    matched = 0
    for prediction in sorted(predicted, key=lambda p: p["confidence"], reverse=True):
        candidates = [(box_iou(prediction["bbox_small"], expected[i]["bbox_small"]), i)
                      for i in remaining if all(prediction[k] == expected[i][k]
                                                for k in ("name", "side"))]
        if candidates and max(candidates)[0] >= .5:
            remaining.remove(max(candidates)[1])
            matched += 1
    return {"tp": matched, "fp": len(predicted)-matched, "fn": len(remaining)}


def main():
    import cr_bot as live
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("reports/live_game_clock_20261021.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    live.logger.remove()
    original = live.Detector(live.DECK).unit_detector
    fixed = BoardUnitDetector(original)
    timings = {"original": [], "fixed": []}
    frames = []
    pairs = json.loads(args.pairs.read_text())["rows"]
    # Warm up both paths; alternate order to reduce thermal/order bias.
    warmup = Image.open(pairs[0]["before_frame"]).convert("RGB").resize((368, 652))
    original.run(warmup)
    fixed.run(warmup)
    for pair in pairs:
        for key in ("before_frame", "after_frame"):
            path = pair[key]
            image = Image.open(path).convert("RGB").resize((368, 652))
            row = {"path": path}
            order = [("original", original), ("fixed", fixed)]
            if len(frames) % 2:
                order.reverse()
            for name, detector in order:
                start = time.perf_counter()
                result = detector.run(image)
                elapsed = (time.perf_counter()-start)*1000
                timings[name].append(elapsed)
                row[name] = serialise(result)
                row[name+"_ms"] = elapsed
            frames.append(row)
        if len(frames) % 20 == 0:
            print("frames", len(frames), flush=True)
    metrics = {}
    labels = json.loads(Path("data/validation/board_regression_20261021/labels_v2.json").read_text())
    by_path = {row["path"]: row for row in frames}
    for name in timings:
        total = {"tp": 0, "fp": 0, "fn": 0}
        for case in labels["frames"]:
            result = score(by_path[case["source_frame"]][name], case["units"])
            for key, value in result.items():
                total[key] += value
        total["precision"] = total["tp"] / max(1, total["tp"]+total["fp"])
        total["recall"] = total["tp"] / max(1, total["tp"]+total["fn"])
        metrics[name] = total
    summary = {"frames": len(frames), "labelled_frames": len(labels["frames"]),
               "metrics": metrics, "label_provenance": labels["provenance"],
               "timings_ms": {k: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                                  "p95": float(np.percentile(v, 95))} for k, v in timings.items()}}
    with (args.output / "predictions.json").open("x") as stream:
        json.dump(frames, stream, indent=2)
    with (args.output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
