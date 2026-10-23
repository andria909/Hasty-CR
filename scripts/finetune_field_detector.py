"""Fine-tune one standard YOLOv8 model from a published KataCR backbone.

This is a class/box detector. Team is not supervised by the external dataset;
the resulting checkpoint must not be claimed to replace team perception.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn

from brain.katacr import load_checkpoint
from prepare_field_finetune import family_name


def adapt_class_head(model, names):
    """Preserve all features/regression; remap trained class logits by family."""
    head=model.model[-1]
    original_names=dict(model.names)
    groups={name:[i for i,raw in original_names.items() if family_name(raw)==name] for name in names}
    for scale,branch in enumerate(head.cv3):
        old=branch[-1]
        replacement=nn.Conv2d(old.in_channels,len(names),old.kernel_size,old.stride,old.padding,
                              dilation=old.dilation,groups=old.groups,bias=True)
        with torch.no_grad():
            nn.init.normal_(replacement.weight,std=.01)
            replacement.bias.fill_(math.log(5/len(names)/(640/float(head.stride[scale]))**2))
            for index,name in enumerate(names):
                sources=groups[name]
                if sources:
                    replacement.weight[index].copy_(old.weight[sources].mean(0))
                    replacement.bias[index].copy_(old.bias[sources].mean(0))
        branch[-1]=replacement
    head.nc=len(names)
    head.no=head.nc+head.reg_max*4
    head.shape=None
    model.names=dict(enumerate(names))
    model.yaml=copy.deepcopy(model.yaml)
    model.yaml["nc"]=len(names)
    model.criterion=None
    return {name:[original_names[i] for i in group] for name,group in groups.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=Path("tmp/field_finetune_20261021/dataset"))
    parser.add_argument("--source",type=Path,default=Path("tmp/katacr_eval_20261021/detector2_v0.7.13.pt"))
    parser.add_argument("--project",type=Path,default=Path("tmp/field_finetune_20261021/runs"))
    parser.add_argument("--name",default="base_families_v1")
    parser.add_argument("--epochs",type=int,default=8)
    parser.add_argument("--batch",type=int,default=4)
    parser.add_argument("--imgsz",type=int,default=640)
    parser.add_argument("--freeze",type=int,default=10)
    parser.add_argument("--device",default="mps")
    args=parser.parse_args()
    args.project=args.project.resolve()
    from ultralytics import YOLO
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    manifest=json.loads((args.dataset/"manifest.json").read_text())
    run=args.project/args.name
    if run.exists():
        raise SystemExit(f"Refusing to overwrite existing experiment {run}; choose another --name")
    init=args.dataset.parent/(args.name+"_init.pt")
    if init.exists():
        raise SystemExit(f"Refusing to overwrite {init}")
    model=load_checkpoint(args.source)
    remapped=adapt_class_head(model,manifest["names"])
    # Standard Ultralytics architecture, not the custom KataCR training loss.
    torch.save({"model":model.half(),"ema":None,"epoch":-1,
                "train_args":{"task":"detect","imgsz":args.imgsz,"single_cls":False}},init)
    provenance={"source":str(args.source),"initialization":str(init),"class_sources":remapped,
                "classes":manifest["names"],"missing_training_examples":manifest["stats"]["train"]["missing"],
                "note":"New classes without a source logit are randomly initialized. No side channel is trained.",
                "training_only":True,"production_promotion":False}
    args.project.mkdir(parents=True,exist_ok=True)
    (args.project/(args.name+"_initialization.json")).write_text(json.dumps(provenance,indent=2)+"\n")
    detector=YOLO(str(init))
    detector.train(data=str((args.dataset/"data.yaml").resolve()),epochs=args.epochs,batch=args.batch,
                   imgsz=args.imgsz,device=args.device,workers=0,project=str(args.project),name=args.name,
                   exist_ok=False,freeze=args.freeze,optimizer="AdamW",lr0=.001,lrf=.1,cos_lr=True,
                   warmup_epochs=1.,patience=4,seed=20261021,deterministic=True,
                   cache=False,amp=False,rect=True,mosaic=0.,close_mosaic=0,mixup=0.,copy_paste=0.,
                   degrees=0.,shear=0.,perspective=0.,translate=.03,scale=.1,flipud=0.,fliplr=.5,
                   hsv_h=.01,hsv_s=.3,hsv_v=.2,plots=False,save=True,save_period=1,val=True)
    print("FINISHED",run,flush=True)


if __name__=="__main__":
    main()
