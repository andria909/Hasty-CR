"""Compare base-family box detection on synthetic diagnostics and fresh TV.

TV labels are provisional single-annotator labels. No team metric is reported:
the candidate was trained without team supervision. Never launches a match.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from brain.katacr import KataCRDetector
from eval_detector_dataset import match_objects
from prepare_field_finetune import family_name


def normalized(rows):
    result=[]
    for row in rows:
        name=family_name(row['raw_name'].replace('_','-'))
        if name is not None:
            result.append({**row,'raw_name':name})
    return result


def score(predictions, targets):
    matched,fp,fn=match_objects(normalized(predictions),normalized(targets))
    return {'tp':len(matched),'fp':len(fp),'fn':len(fn),
            'false_positives':fp,'missed':fn}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--imgsz',type=int,default=640,
                        help='Candidate inference size; default matches fine-tuning')
    args=parser.parse_args()
    from ultralytics import YOLO
    torch.set_num_threads(3)
    args.output.mkdir(parents=True,exist_ok=False)
    model=YOLO(str(args.checkpoint))
    synthetic=json.loads(Path('data/validation/external_synthetic_20261021/labels.json').read_text())
    previous=json.loads(Path('reports/katacr_broad_20261021_v2/predictions.json').read_text())
    cached={r['path']:r['predictions'] for r in previous}
    tv=json.loads(Path('data/validation/tvroyale_20261021/labels.json').read_text())
    pair=KataCRDetector([Path(f'tmp/katacr_eval_20261021/detector{i}_v0.7.13.pt') for i in (1,2)],
                        device=args.device,crop=tuple(tv['crop_for_detector']))
    results=[]
    for group,frames in [('synthetic',synthetic['frames']),('tv',tv['frames'])]:
        for case in frames:
            image=Image.open(case['path']).convert('RGB')
            if group=='synthetic':
                bounds=(0,0,*image.size)
                targets=[{'raw_name':synthetic['names'][c],'bbox':b} for c,b in
                         zip(case['objects']['category'],case['objects']['bbox'])]
                baseline=cached[case['path']]
            else:
                bounds=tuple(round(v*image.size[i%2]) for i,v in enumerate(tv['crop_for_detector']))
                targets=[{'raw_name':o['name'],'bbox':o['bbox']} for o in case['objects']]
                baseline=pair.detect(image)
            start=time.perf_counter()
            output=model.predict(image.crop(bounds),imgsz=args.imgsz,conf=.4,iou=.6,
                                 device=args.device,verbose=False,agnostic_nms=True)[0]
            predictions=[]
            for box,confidence,cls in zip(output.boxes.xyxy.tolist(),output.boxes.conf.tolist(),output.boxes.cls.tolist()):
                predictions.append({'raw_name':model.names[int(cls)],'confidence':confidence,
                    'bbox':[box[0]+bounds[0],box[1]+bounds[1],box[2]+bounds[0],box[3]+bounds[1]]})
            results.append({'group':group,'path':case['path'],'candidate':score(predictions,targets),
                            'baseline':score(baseline,targets),'predictions':predictions,
                            'baseline_predictions':baseline,'ms':(time.perf_counter()-start)*1000})
            print(group,len(results),flush=True)
    summary={'checkpoint':str(args.checkpoint),'iou':.5,'confidence':.4,'team_evaluated':False,
             'candidate_imgsz':args.imgsz,'device':args.device,
             'limitations':['Synthetic cutout identities can overlap with training.',
                            'TV labels provisional and sparse: single annotator, four correlated frames.',
                            'Timing includes candidate inference only and is not a paired speed benchmark.']}
    for group in ('synthetic','tv'):
        summary[group]={}
        for detector in ('candidate','baseline'):
            c=Counter()
            for r in results:
                if r['group']==group:
                    c.update({k:r[detector][k] for k in ('tp','fp','fn')})
            summary[group][detector]={**c,'precision':c['tp']/max(1,c['tp']+c['fp']),
                                     'recall':c['tp']/max(1,c['tp']+c['fn'])}
    for name,data in [('summary.json',summary),('predictions.json',results)]:
        (args.output/name).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
