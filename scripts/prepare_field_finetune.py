"""Prepare a reproducible field dataset with evolution/hero families merged.

Whole background-video groups of the external diagnostic set are excluded
from training. Synthetic cut-outs may still be shared: this is not a guarantee
of independent source actors. Fresh TV Royale frames are never training data.
"""
import argparse
import hashlib
import io
import json
import random
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from brain.katacr import unit_name


def family_name(raw):
    if unit_name(raw) is None:
        return None
    name=raw.removesuffix("-evolution").removesuffix("-hero")
    # A Skeleton Army is a card; a bounding box labels an individual skeleton.
    if name=="skeleton-army":
        return "skeleton"
    if name.startswith("phoenix-") and name in {"phoenix-big","phoenix-small"}:
        return "phoenix"
    return name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata",type=Path,default=Path("tmp/detector_validation_20261021/training_metadata.json"))
    parser.add_argument("--out",type=Path,default=Path("tmp/field_finetune_20261021/dataset"))
    parser.add_argument("--val-size",type=int,default=192)
    args=parser.parse_args()
    source=json.loads(args.metadata.read_text())
    raw_names=source["names"]
    names=sorted({family_name(n) for n in raw_names if family_name(n) is not None})
    lookup={name:i for i,name in enumerate(names)}
    heldout=set(source["holdout_videos"])
    rows=source["rows"]
    train=[r for r in rows if r["row"]["backdrop_video"] not in heldout]
    # Exclude the fixed 54-frame diagnostic set from model-selection validation.
    diagnostics=json.loads(Path("data/validation/external_synthetic_20261021/labels.json").read_text())
    diagnostic_ids={r["row_index"] for r in diagnostics["frames"]}
    pool=[r for r in rows if r["row"]["backdrop_video"] in heldout and r["row_idx"] not in diagnostic_ids]
    random.Random(20261021).shuffle(pool)
    val=[]; seen=Counter()
    while pool and len(val)<args.val_size:
        def gain(row):
            return sum(max(0,8-seen[family_name(raw_names[c])]) for c in set(row["row"]["objects"]["category"])
                       if family_name(raw_names[c]) is not None)
        row=max(pool,key=gain)
        val.append(row);pool.remove(row)
        seen.update(family_name(raw_names[c]) for c in row["row"]["objects"]["category"]
                    if family_name(raw_names[c]) is not None)
    stats={}; records=[]
    for split,selected in (("train",train),("val",val)):
        (args.out/"images"/split).mkdir(parents=True,exist_ok=True)
        (args.out/"labels"/split).mkdir(parents=True,exist_ok=True)
        counts=Counter()
        def download(item):
            index,row=item
            p=args.out/"images"/split/f"row_{row['row_idx']:06d}.jpg"
            if not p.exists():
                error=None
                for _ in range(3):
                    try:
                        binary=urllib.request.urlopen(row["row"]["image"]["src"],timeout=40).read()
                        Image.open(io.BytesIO(binary)).verify()
                        p.write_bytes(binary);error=None;break
                    except Exception as exc:
                        error=exc
                if error is not None:
                    raise error
            with Image.open(p) as image:
                width,height=image.size
            labels=[]; frame_counts=Counter()
            for cls,box in zip(row["row"]["objects"]["category"],row["row"]["objects"]["bbox"]):
                family=family_name(raw_names[cls])
                if family is None:
                    continue
                x1,y1,x2,y2=box
                x1=max(0,min(width,x1));x2=max(0,min(width,x2))
                y1=max(0,min(height,y1));y2=max(0,min(height,y2))
                if x2<=x1 or y2<=y1:
                    continue
                labels.append(f"{lookup[family]} {(x1+x2)/2/width:.7f} {(y1+y2)/2/height:.7f} {(x2-x1)/width:.7f} {(y2-y1)/height:.7f}")
                frame_counts[family]+=1
            label=args.out/"labels"/split/(p.stem+".txt")
            label.write_text("\n".join(labels)+("\n" if labels else ""))
            return {"path":str(p),"row_index":row["row_idx"],"backdrop_video":row["row"]["backdrop_video"],
                    "sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"split":split},frame_counts
        with ThreadPoolExecutor(max_workers=8) as executor:
            for index,(record,count) in enumerate(executor.map(download,enumerate(selected))):
                counts.update(count);records.append(record)
                if (index+1)%100==0:
                    print(split,index+1,"/",len(selected),flush=True)
        stats[split]={"images":len(selected),"counts":dict(sorted(counts.items())),
                      "missing":[n for n in names if counts[n]==0]}
    yaml=[f"path: {args.out.resolve()}","train: images/train","val: images/val","names:"]
    yaml += [f"  {i}: {n}" for i,n in enumerate(names)]
    (args.out/"data.yaml").write_text("\n".join(yaml)+"\n")
    manifest={"source_dataset":"SudhishNeel/clash-royale-detection","revision":"964b5346ec62f6c9cf813e48910d1743a6a37a1c",
              "source_split":"train","config":"synthetic_arena","names":names,
              "raw_to_family":{n:family_name(n) for n in raw_names},"stats":stats,"frames":records,
              "limitations":["Publisher synthetic data, no team labels.","Background-video groups disjoint; cut-out identities may overlap.",
                             "No TV Royale screenshots used in training.","Names with zero training instances are NOT learned just by adding a class."]}
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"classes":len(names),"stats":stats},indent=2))


if __name__=="__main__":
    main()
