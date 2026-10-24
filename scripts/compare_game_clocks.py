"""Read the visible whole-second battle clock in saved before/after pairs on macOS.

This measures displayed clock changes, not the exact instant of troop spawn.
Run after live collection so OCR does not compete with the bot for CPU.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from PIL import Image


def read_clock(path, scratch):
    import Vision
    from Foundation import NSURL

    with Image.open(path) as image:
        crop = image.crop((int(image.width * .78), 0,
                           image.width, int(image.height * .09)))
        crop.resize((600, 440)).save(scratch)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)
    request.setRecognitionLanguages_(["en-US"])
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
        NSURL.fileURLWithPath_(str(scratch)), {})
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        return {"seconds": None, "error": str(error)}
    candidates = []
    for observation in request.results():
        text = observation.topCandidates_(1)[0]
        match = re.fullmatch(r"\s*([0-5]):([0-5]\d)\s*", str(text.string()))
        if match and text.confidence() >= .7:
            candidates.append({"text": str(text.string()),
                               "confidence": float(text.confidence()),
                               "seconds": int(match[1]) * 60 + int(match[2])})
    return max(candidates, key=lambda item: item["confidence"]) if candidates else {"seconds": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = [json.loads(line) for line in args.input.read_text().splitlines()
             if line.strip() and json.loads(line).get("status") == "frame_pair"]
    # Keep the scratch file: no cleanup of existing user files is performed.
    descriptor, name = tempfile.mkstemp(prefix="hasty-clock-", suffix=".png")
    os.close(descriptor)
    scratch = Path(name)
    rows = []
    for pair in pairs:
        before = read_clock(pair["before_frame"], scratch)
        after = read_clock(pair["after_frame"], scratch)
        delta = None
        if before["seconds"] is not None and after["seconds"] is not None:
            difference = before["seconds"] - after["seconds"]
            if 0 <= difference <= 5:
                delta = difference
        rows.append(dict(pair, before_clock=before, after_clock=after,
                         displayed_game_delta_s=delta))
    deltas = [r["displayed_game_delta_s"] for r in rows if r["displayed_game_delta_s"] is not None]
    result = {
        "definition": "visible countdown difference between decision image and post-input control image",
        "resolution_s": 1,
        "warning": "post-input image does not independently prove troop spawn; OCR failures and clock resets excluded",
        "pairs": len(rows), "valid_pairs": len(deltas),
        "excluded_pairs": len(rows) - len(deltas),
        "delta_counts": dict(Counter(deltas)),
        "mean_displayed_delta_s": float(np.mean(deltas)) if deltas else None,
        "std_displayed_delta_s": float(np.std(deltas)) if deltas else None,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
