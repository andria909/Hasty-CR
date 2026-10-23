"""Measure serial pair latency separately from batched dataset evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from brain.katacr import KataCRDetector


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=64)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.output.exists():raise SystemExit('Refusing to overwrite timing output')
    root=Path('tmp/katacr_original_eval_20261022')
    paths=(root/'yolo_annotation.txt').read_text().splitlines()
    selected=[paths[i].removeprefix('./') for i in np.linspace(0,len(paths)-1,args.samples,dtype=int)]
    detector=KataCRDetector([Path(f'tmp/katacr_eval_20261021/detector{i}_v0.7.13.pt') for i in (1,2)],
                            device='mps',confidence=.4,crop=(0,0,1,1))
    with Image.open(root/'real_frames'/selected[0]) as im:
        for _ in range(5):detector.detect(im)
    samples=[]
    for relative in selected:
        with Image.open(root/'real_frames'/relative) as im:
            im=im.convert('RGB');detector.detect(im)
        samples.append({'path':relative,'ms':detector.last_ms})
    values=np.array([r['ms'] for r in samples])
    result={'model':'original KataCR v0.7.13 pair','batch':1,'confidence':.4,'nms_iou':.6,
            'device':'mps','samples':len(samples),'warmup':5,'input_width_height':[576,896],
            'mean_ms':float(values.mean()),'std_ms':float(values.std()),
            'p50_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),
            'min_ms':float(values.min()),'max_ms':float(values.max()),
            'includes':'PIL conversion/resize/padding, tensor preparation, both model forwards, CPU decoding and NMS, MPS synchronization',
            'excludes':'Disk image reads, emulator screenshot acquisition, game decisions, ADB actions',
            'limitations':['MuMu and other user apps remain running; timing includes contention, not an isolated hardware benchmark.',
                           'Different NMS confidence/settings from low-threshold mAP evaluation; operational serial timing recorded separately.'],
            'measurements':samples}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='measurements'},indent=2))


if __name__=='__main__':main()
