"""Evaluate the published KataCR pair on the author's complete real-frame list.

No training, emulator actions, or production changes. The author's validation
list is not evidence of an independent unseen-video test split.
"""
import argparse
import ast
import csv
import hashlib
import json
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms
from ultralytics.utils.metrics import ap_per_class

from brain.katacr import decode_predictions, load_checkpoint, unit_name


def match_predictions(predictions, targets):
    """Confidence-ordered, class-exact, one-to-one matches at ten IoUs."""
    correct=np.zeros((len(predictions),10),dtype=bool)
    matched50=np.full(len(predictions),-1,dtype=int)
    if not len(predictions) or not len(targets):return correct,matched50
    p=predictions[:,:4];t=targets[:,:4]
    intersection=np.maximum(0,np.minimum(t[:,None,2:],p[None,:,2:])-np.maximum(t[:,None,:2],p[None,:,:2])).prod(2)
    area_t=np.maximum(0,t[:,2:]-t[:,:2]).prod(1)
    area_p=np.maximum(0,p[:,2:]-p[:,:2]).prod(1)
    overlap=intersection/np.maximum(1e-9,area_t[:,None]+area_p[None,:]-intersection)
    overlap*=targets[:,None,4]==predictions[None,:,5]
    for column,threshold in enumerate(np.linspace(.5,.95,10)):
        used=set()
        for i in range(len(predictions)):
            available=overlap[:,i].copy()
            if used:available[list(used)]=-1
            index=int(available.argmax())
            if available[index]>=threshold:
                correct[i,column]=True;used.add(index)
                if column==0:matched50[i]=index
    return correct,matched50


def extract_dataset(archive,root,listing):
    """Extract only listed images and YOLO labels; retain the source archive."""
    root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        prefix=z.namelist()[0].split('/')[0]+'/images/part2/'
        for relative in listing:
            for suffix in ('.jpg','.txt'):
                path=Path(relative).with_suffix(suffix)
                if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe dataset path')
                dest=root/path;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():
                    with dest.open('xb') as output:output.write(z.read(prefix+path.as_posix()))


def read_case(root,relative):
    path=root/relative
    with Image.open(path) as im:
        im=im.convert('RGB');W,H=im.size
        pixels=np.full((896,576,3),114,dtype=np.uint8)
        pixels[:,4:572]=np.asarray(im.resize((568,896),Image.Resampling.BILINEAR))
    targets=[]
    for line in path.with_suffix('.txt').read_text().splitlines():
        values=list(map(float,line.split()))
        if not values:continue
        cls,x,y,w,h,side=values[:6]
        targets.append([(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H,cls,side])
    targets=np.asarray(targets,dtype=float).reshape(-1,6)
    targets[:,[0,2]]=targets[:,[0,2]].clip(0,W)
    targets[:,[1,3]]=targets[:,[1,3]].clip(0,H)
    return pixels,targets,(W,H)


def summarize(rows,names,classes,output,prefix):
    classes=sorted(classes)
    tp=np.concatenate([r['tp'] for r in rows])
    predictions=np.concatenate([r['predictions'] for r in rows])
    targets=np.concatenate([r['targets'][:,4] for r in rows])
    keep_pred=np.isin(predictions[:,5],classes);keep_gt=np.isin(targets,classes)
    predictions=predictions[keep_pred];tp=tp[keep_pred];targets=targets[keep_gt]
    stats=ap_per_class(tp,predictions[:,4],predictions[:,5],targets,plot=False,names=dict(enumerate(names)))
    ap=stats[5];indices=stats[6]
    operating=predictions[:,4]>=.4
    correct=int(tp[operating,0].sum());fp=int(operating.sum())-correct;fn=len(targets)-correct
    per_class=[]
    for i,cls in enumerate(indices):
        mask=predictions[:,5]==cls;active=mask&operating
        hit=int(tp[active,0].sum());false=int(active.sum())-hit;total=int((targets==cls).sum())
        per_class.append({'class':names[cls],'objects':total,'tp':hit,'fp':false,'fn':total-hit,
                          'precision':hit/max(1,hit+false),'recall':hit/max(1,total),
                          'AP50':float(ap[i,0]),'AP50_95':float(ap[i].mean())})
    with (output/(prefix+'_classes.csv')).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_class[0]));writer.writeheader();writer.writerows(per_class)
    return {'frames':len(rows),'objects':len(targets),'classes_present':len(indices),
            'mAP50':float(ap[:,0].mean()),'mAP50_95':float(ap.mean()),
            'confidence_0.4_IoU_0.5':{'tp':correct,'fp':fp,'fn':fn,
                'precision':correct/max(1,correct+fp),'recall':correct/max(1,correct+fn)}}


def draw_case(root,relative,predictions,targets,names,out):
    im=Image.open(root/relative).convert('RGB');d=ImageDraw.Draw(im)
    for row in targets:
        d.rectangle(row[:4].tolist(),outline='lime',width=1)
    for row in predictions:
        if row[4]<.4:continue
        d.rectangle(row[:4].tolist(),outline='red',width=2)
        d.text(tuple(row[:2]),names[int(row[5])],fill='red',stroke_width=1,stroke_fill='black')
    im.save(out,quality=95)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('tmp/katacr_original_eval_20261022'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--limit',type=int,default=0,help='Only for smoke tests; default evaluates full list')
    parser.add_argument('--device',default='mps')
    args=parser.parse_args();torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    tree=ast.parse(Path('vendor/KataCR-eval-20261021/katacr/constants/label_list.py').read_text())
    names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='unit_list' for t in n.targets))
    listing=[str(Path(line.removeprefix('./'))) for line in (args.source/'yolo_annotation.txt').read_text().splitlines() if line.strip()]
    if len(listing)!=len(set(listing)):raise ValueError('Duplicate evaluation images')
    if args.limit:listing=listing[:args.limit]
    root=args.source/'real_frames'
    extract_dataset(args.source/'original_dataset.zip',root,listing)
    print('Dataset ready',len(listing),flush=True)
    weights=[Path(f'tmp/katacr_eval_20261021/detector{i}_v0.7.13.pt') for i in (1,2)]
    models=[load_checkpoint(p).fuse(verbose=False).to(args.device) for p in weights]
    supported={names.index(n) for m in models for n in m.names.values() if n in names}
    unit_classes={i for i,n in enumerate(names) if unit_name(n) is not None}
    mapping=[{i:names.index(n) for i,n in m.names.items() if n in names} for m in models]
    with torch.inference_mode():
        dummy=torch.zeros(args.batch,3,896,576,device=args.device)
        for _ in range(2):
            for m in models:m(dummy)
        if args.device=='mps':torch.mps.synchronize()
    rows=[];batch_ms=[];frame_errors=[];side_correct=side_total=0;start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool,(args.output/'predictions.jsonl').open('x') as stream:
        for first in range(0,len(listing),args.batch):
            paths=listing[first:first+args.batch]
            cases=list(pool.map(lambda p:read_case(root,p),paths))
            tick=time.perf_counter()
            pixels=torch.from_numpy(np.stack([r[0].transpose(2,0,1) for r in cases]).copy()).float().to(args.device)/255
            with torch.inference_mode():
                raw=[]
                for model in models:
                    pred=model(pixels);raw.append((pred[0] if isinstance(pred,tuple) else pred).detach().float().cpu())
            frame_predictions=[]
            for j,(_,targets,(W,H)) in enumerate(cases):
                parts=[]
                for k,model in enumerate(models):
                    decoded=decode_predictions(raw[k][j:j+1],model.names,confidence=.001,iou=.7).numpy()
                    decoded=decoded[[int(c) in mapping[k] for c in decoded[:,5]]]
                    for row in decoded:row[5]=mapping[k][int(row[5])]
                    parts.append(decoded)
                predictions=np.concatenate(parts) if parts else np.zeros((0,7))
                if len(predictions):
                    indices=nms(torch.from_numpy(predictions[:,:4].copy()),torch.from_numpy(predictions[:,4].copy()),.7).numpy()[:300]
                    predictions=predictions[indices]
                    predictions[:,[0,2]]=(predictions[:,[0,2]]-4)*W/568
                    predictions[:,[1,3]]*=H/896
                    predictions[:,[0,2]]=predictions[:,[0,2]].clip(0,W)
                    predictions[:,[1,3]]=predictions[:,[1,3]].clip(0,H)
                frame_predictions.append(predictions)
            if args.device=='mps':torch.mps.synchronize()
            duration=(time.perf_counter()-tick)*1000;batch_ms.extend([duration/len(cases)]*len(cases))
            for relative,(_,targets,_),predictions in zip(paths,cases,frame_predictions):
                tp,matched=match_predictions(predictions,targets)
                active=predictions[:,4]>=.4;unit_mask=np.isin(predictions[:,5],list(unit_classes));hits=active&unit_mask&(matched>=0)
                side_total+=int(hits.sum())
                side_correct+=int(((predictions[hits,6]>.5)==targets[matched[hits],5]).sum())
                missed=int(np.isin(targets[:,4],list(unit_classes)).sum())-int(tp[active&unit_mask,0].sum())
                false=int((active&unit_mask).sum())-int(tp[active&unit_mask,0].sum())
                frame_errors.append({'path':relative,'unit_fp':false,'unit_fn':missed})
                rows.append({'tp':tp,'predictions':predictions,'targets':targets})
                stream.write(json.dumps({'path':relative,'predictions':predictions.tolist(),'columns':['x1','y1','x2','y2','confidence','class','enemy_probability']})+'\n')
                if len(rows)<=6:draw_case(root,relative,predictions,targets,names,args.output/f'example_{len(rows):02d}.jpg')
            stream.flush()
            if first//args.batch%25==0:
                elapsed=time.perf_counter()-start
                print(json.dumps({'frames':len(rows),'total':len(listing),'elapsed_s':round(elapsed,1),'remaining_s':round(elapsed/len(rows)*(len(listing)-len(rows)),1)}),flush=True)
    summary={'model':'Published KataCR detector1 + detector2 v0.7.13, original weights, no fine-tuning',
             'dataset':'https://github.com/wty-yy/Clash-Royale-Detection-Dataset','dataset_revision':'31b4151fedb1b914e99c3c122c16dca61cb2b905',
             'list':'images/part2/yolo_annotation.txt','list_sha256':hashlib.sha256((args.source/'yolo_annotation.txt').read_bytes()).hexdigest(),
             'full_list':not args.limit,'class_names':names,'settings':{'confidence_for_AP':.001,'confidence_for_PR':.4,'nms_iou':.7,'max_detections':300,'input_width_height':[576,896],'batch':args.batch,'device':args.device,'team_threshold':.5},
             'all_annotated_classes':summarize(rows,names,set(range(len(names))),args.output,'all'),
             'supported_classes':summarize(rows,names,supported,args.output,'supported'),
             'project_unit_scope':summarize(rows,names,unit_classes,args.output,'units'),
             'unit_side_accuracy_on_class_correct_matches':{'correct':side_correct,'total':side_total,'accuracy':side_correct/max(1,side_total)},
             'timing':{'mode':'batched preprocessing + both model forwards + decoding and NMS; excludes disk reads and metrics; amortized per image, not single-frame latency','mean_ms_per_image':float(np.mean(batch_ms)),'p50_ms_per_image':float(np.median(batch_ms)),'p95_ms_per_image':float(np.percentile(batch_ms,95)),'evaluation_wall_s':time.perf_counter()-start},
             'limitations':['Author validation list; source-video/cutout overlap with original training may exist. Not an independent unseen-video test.','Correlated adjacent frames; aggregate object counts are not independent trials.','Original weights may intentionally omit annotated UI/effect classes; all-class and supported-class scores reported separately.','Project unit scope follows existing unit_name exclusions; exact original classes, no evolution merging.','Our modern runtime adapter; validation uses .7 NMS like author validator, no live HUD removal or tracker.']}
    for rank,error in enumerate(sorted(frame_errors,key=lambda r:r['unit_fp']+r['unit_fn'],reverse=True)[:8]):
        index=listing.index(error['path']);row=rows[index]
        draw_case(root,error['path'],row['predictions'],row['targets'],names,args.output/f'worst_{rank:02d}.jpg')
    (args.output/'frame_errors.json').write_text(json.dumps(frame_errors,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='class_names'},indent=2),flush=True)


if __name__=='__main__':main()
