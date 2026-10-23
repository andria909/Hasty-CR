"""Offline comparison of KataCR's published pair against saved BuildABot outputs."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from brain.katacr import KataCRDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=Path("tmp/katacr_eval_20261021"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--confidence", type=float, default=.4)
    args = parser.parse_args()
    import torch
    import ultralytics
    torch.set_num_threads(4)
    weights = [args.weights_dir / f"detector{i}_v0.7.13.pt" for i in (1,2)]
    detector = KataCRDetector(weights, device=args.device, confidence=args.confidence)
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path("reports/board_cv_fix_20261021_final/predictions.json").read_text())
    labels = json.loads(Path("data/validation/board_regression_20261021/labels_v2.json").read_text())
    selected = {row["source_frame"] for row in labels["frames"]}
    # Warm-up is excluded from steady-state timing, but reported separately.
    detector.detect(Image.open(baseline[0]["path"]))
    warmup_ms = detector.last_ms
    frames = []
    for index, original in enumerate(baseline):
        with Image.open(original["path"]) as source:
            image = source.convert("RGB")
        predictions = detector.detect(image)
        units = [{**row, "bbox_small": [v*(368/image.width if k%2==0 else 652/image.height)
                                          for k,v in enumerate(row["bbox"])]}
                 for row in predictions]
        frames.append({"path": original["path"], "units": units, "ms": detector.last_ms})
        if original["path"] in selected:
            draw = ImageDraw.Draw(image)
            for row in predictions:
                color = "cyan" if row["side"] == "ally" else "red"
                draw.rectangle(row["bbox"], outline=color, width=3)
                draw.text((row["bbox"][0], max(0,row["bbox"][1]-16)),
                          f'{row["name"]} {row["side"]} {row["confidence"]:.2f}',
                          fill=color, stroke_width=1, stroke_fill="black")
            image.save(args.output / (Path(original["path"]).stem+".jpg"), quality=92)
        if (index+1)%20 == 0:
            print("frames", index+1, flush=True)
    # Import after detector timing to keep the unrelated live stack out of it.
    from audit_board_cv import score
    by_path = {row["path"]: row for row in frames}
    old_by_path = {row["path"]: row for row in baseline}
    metrics = {}
    for name in ("original", "fixed", "katacr"):
        total = {"tp":0, "fp":0, "fn":0}
        cases = []
        for case in labels["frames"]:
            path = case["source_frame"]
            predicted = by_path[path]["units"] if name=="katacr" else old_by_path[path][name]
            result = score(predicted, case["units"])
            cases.append({"id":case["id"], **result})
            for key, value in result.items():
                total[key] += value
        total["precision"] = total["tp"]/max(1,total["tp"]+total["fp"])
        total["recall"] = total["tp"]/max(1,total["tp"]+total["fn"])
        metrics[name] = {**total, "cases":cases}
    timings = [row["ms"] for row in frames]
    summary = {"frames":len(frames), "metrics":metrics,
               "timings_ms":{"warmup":warmup_ms,"mean":float(np.mean(timings)),
                             "median":float(np.median(timings)),"p95":float(np.percentile(timings,95))},
               "device":args.device,"platform":platform.platform(),"torch":torch.__version__,
               "ultralytics":ultralytics.__version__, "confidence":args.confidence,
               "crop":detector.crop,"network_input":[576,896],
               "weights":[{"path":str(p),"bytes":p.stat().st_size,
                           "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in weights],
               "annotation_correction":labels["corrections"],
               "limitations":["Only four fully labelled frames/nine units, one provisional annotator.",
                              "Correlated diagnostic frames; not independent generalisation metrics.",
                              "Full-resolution source used for KataCR; BuildABot uses its normal 368x652 input.",
                              "Baseline timings from a previous paired run; not concurrent hardware timing.",
                              "No new battles, training or production model replacement."]}
    for name, data in (("predictions.json",frames),("summary.json",summary)):
        with (args.output/name).open("x") as stream:
            json.dump(data,stream,indent=2)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
