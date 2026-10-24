"""Append a reviewed patch list as synthetic augmentation to the v2 dataset."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('patches',type=Path)
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--repeats',type=int,default=16)
    args=parser.parse_args()
    root=Path('tmp/field_finetune_20261021/dataset_supplement_v2')
    manifest=json.loads((root/'manifest.json').read_text());names=manifest['names']
    patches=json.loads(args.patches.read_text())
    base=Path('tmp/field_finetune_20261021/dataset');backgrounds=sorted((base/'images/train').glob('*.jpg'))
    rng=random.Random(20261022);counts=Counter();records=[]
    for index,row in enumerate(patches):
        with Image.open(row['path']) as source:patch=source.convert('RGB').crop(row['bbox'])
        for repeat in range(args.repeats):
            stem=f'{args.prefix}_{index:03d}_{repeat:02d}'
            path=root/'images/train'/(stem+'.jpg')
            if path.exists():raise SystemExit('Refusing to overwrite '+str(path))
            background=rng.choice(backgrounds);image=Image.open(background).convert('RGB');W,H=image.size
            labels=[list(map(float,l.split())) for l in (base/'labels/train'/(background.stem+'.txt')).read_text().splitlines()]
            th=rng.randint(40,85);tw=max(8,round(patch.width/patch.height*th))
            for _ in range(200):
                x=rng.randint(20,W-tw-20);y=rng.randint(160,H-th-70)
                overlap=any(min((cx+bw/2)*W,x+tw)>max((cx-bw/2)*W,x) and min((cy+bh/2)*H,y+th)>max((cy-bh/2)*H,y)
                            for _,cx,cy,bw,bh in labels)
                if not overlap:break
            else:continue
            image.paste(patch.resize((tw,th)),(x,y));image.save(path,quality=95)
            labels.append([names.index(row['family']),(x+tw/2)/W,(y+th/2)/H,tw/W,th/H])
            (root/'labels/train'/(stem+'.txt')).write_text('\n'.join(str(int(l[0]))+' '+' '.join(f'{v:.7f}' for v in l[1:]) for l in labels)+'\n')
            counts[row['family']]+=1;records.append({'image':stem,'background':str(background),'patch':row})
    manifest['patches'].extend(patches);manifest['records'].extend(records)
    total=Counter(manifest['augmented_instances']);total.update(counts);manifest['augmented_instances']=dict(total)
    manifest['limitations'].append('Additional training-only TV replay patches are separate from original TV evaluation replays.')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'appended':len(records),'counts':dict(counts)}))


if __name__=='__main__':main()
