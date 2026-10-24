"""Evaluate KataCR against externally annotated arena images, by exact class.

Synthetic data and a source 'train' split are diagnostics, not a held-out
claim. Team accuracy cannot be measured when annotations have no team field.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain.katacr import KataCRDetector, unit_name
from live_perception import box_iou


def match_objects(predictions, targets, threshold=.5):
    remaining = set(range(len(targets)))
    matches = []
    unmatched = []
    for prediction in sorted(predictions, key=lambda p:p["confidence"], reverse=True):
        candidates = [(box_iou(prediction["bbox"], targets[i]["bbox"]), i)
                      for i in remaining if prediction["raw_name"] == targets[i]["raw_name"]]
        if candidates and max(candidates)[0] >= threshold:
            overlap, index = max(candidates)
            remaining.remove(index)
            matches.append((prediction, targets[index], overlap))
        else:
            unmatched.append(prediction)
    return matches, unmatched, [targets[i] for i in sorted(remaining)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--weights-dir",type=Path,default=Path("tmp/katacr_eval_20261021"))
    parser.add_argument("--device",default="mps")
    args=parser.parse_args()
    torch.set_num_threads(4)
    dataset=json.loads(args.labels.read_text())
    names=dataset["names"]
    detector=KataCRDetector([args.weights_dir/f"detector{i}_v0.7.13.pt" for i in (1,2)],
                            device=args.device,crop=(0.,0.,1.,1.))
    supported=set(n for model in detector.models for n in model.names.values())
    args.output.mkdir(parents=True,exist_ok=True)
    counts=defaultdict(Counter)
    results=[]
    for index,case in enumerate(dataset["frames"]):
        image=Image.open(case["path"]).convert("RGB")
        predictions=detector.detect(image)
        all_targets=[{"raw_name":names[c],"bbox":b} for c,b in
                     zip(case["objects"]["category"],case["objects"]["bbox"])]
        targets=[row for row in all_targets if unit_name(row["raw_name"]) is not None]
        matches, false_positives, missed=match_objects(predictions,targets)
        for target in all_targets:
            counts[target["raw_name"]]["annotated"]+=1
        for _,target,_ in matches:
            counts[target["raw_name"]]["tp"]+=1
        for row in false_positives:
            counts[row["raw_name"]]["fp"]+=1
        for row in missed:
            counts[row["raw_name"]]["fn"]+=1
        results.append({"path":case["path"],"source_id":case["id"],"predictions":predictions,
                        "tp":len(matches),"fp":len(false_positives),"fn":len(missed),
                        "missed":missed,"false_positives":false_positives,"ms":detector.last_ms})
        if index<6:
            draw=ImageDraw.Draw(image)
            for row in targets:
                draw.rectangle(row["bbox"],outline="lime",width=1)
            for row in predictions:
                draw.rectangle(row["bbox"],outline="red",width=2)
                draw.text(row["bbox"][:2],row["raw_name"],fill="red",stroke_width=1,stroke_fill="black")
            image.save(args.output/f"example_{index:02d}.jpg",quality=90)
        if (index+1)%10==0:
            print("frames",index+1,flush=True)
    inventory=[]
    for name in sorted(set(names)|set(counts)):
        c=counts[name]
        included=unit_name(name) is not None
        inventory.append({"class":name,"scope":"unit" if included else "excluded_ui_or_spell",
                          "model_support":name in supported,"annotated":c["annotated"],
                          "tp":c["tp"],"fp":c["fp"],"fn":c["fn"],
                          "precision":c["tp"]/max(1,c["tp"]+c["fp"]),
                          "recall":c["tp"]/max(1,c["tp"]+c["fn"])})
    def aggregate(rows):
        total={k:sum(row[k] for row in rows) for k in ("tp","fp","fn")}
        return {**total,"precision":total["tp"]/max(1,total["tp"]+total["fp"]),
                "recall":total["tp"]/max(1,total["tp"]+total["fn"])}
    summary={"frames":len(results),"annotated_objects":sum(r["annotated"] for r in inventory),
             "classes_present":sum(r["annotated"]>0 for r in inventory),
             "unit_classes_present":sum(r["annotated"]>0 and r["scope"]=="unit" for r in inventory),
             "unit_classes_present_supported":sum(r["annotated"]>0 and r["scope"]=="unit" and r["model_support"] for r in inventory),
             "all_unit_classes":aggregate([r for r in inventory if r["scope"]=="unit"]),
             "supported_unit_classes":aggregate([r for r in inventory if r["scope"]=="unit" and r["model_support"]]),
             "iou":.5,"confidence":detector.confidence,
             "source":{"dataset":dataset["dataset"],"revision":dataset["revision"],
                       "config":dataset["config"],"split":dataset["split"]},
             "limitations":["Synthetic pasted units; geometry/annotation differences affect IoU.",
                            "Source train split used as external diagnostic, not independent held-out test.",
                            "Only three or more examples targeted per observed class, not full class diversity.",
                            "No team ground truth: side accuracy NOT measured.",
                            "Modern heroes/evolutions absent from 2024 weights are counted as unsupported.",
                            "Current unit scope uses project unit_name exclusions; raw inventory retained.",
                            "Simultaneous emulator screenshot capture; timings are not a paired speed benchmark."]}
    for name,data in (("summary.json",summary),("predictions.json",results),("class_inventory.json",inventory)):
        with (args.output/name).open("x") as stream:
            json.dump(data,stream,indent=2)
    with (args.output/"class_inventory.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(inventory[0]));writer.writeheader();writer.writerows(inventory)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
