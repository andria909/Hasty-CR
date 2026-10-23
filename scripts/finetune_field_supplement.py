"""Adapt the completed first-stage checkpoint on rare-object augmentation."""
import argparse
from pathlib import Path

import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--name',default='rare_families_v2')
    args=parser.parse_args()
    from ultralytics import YOLO
    root=Path('tmp/field_finetune_20261021').resolve()
    run=root/'runs'/args.name
    if run.exists():raise SystemExit('Refusing to overwrite experiment')
    torch.set_num_threads(4)
    model=YOLO(str(args.source))
    model.train(data=str(root/'dataset_supplement_v2/data.yaml'),epochs=3,batch=4,imgsz=640,
                device='mps',workers=0,project=str(root/'runs'),name=args.name,exist_ok=False,
                freeze=20,optimizer='AdamW',lr0=.0004,lrf=.2,cos_lr=True,warmup_epochs=.5,
                patience=3,seed=20261022,deterministic=True,cache=False,amp=False,rect=True,
                mosaic=0.,mixup=0.,close_mosaic=0,copy_paste=0.,translate=.03,scale=.1,
                flipud=0.,fliplr=.5,hsv_h=.01,hsv_s=.2,hsv_v=.15,plots=False,
                save=True,save_period=1,val=True)
    print('FINISHED',run,flush=True)


if __name__=='__main__':main()
