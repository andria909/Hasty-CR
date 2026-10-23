"""Add reviewed rare objects without treating enemy-only scenes as fully labelled.

Creates a new dataset; never mutates the initial dataset. Pasted rectangular
patches are synthetic augmentation, not new independent real observations.
"""
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from prepare_field_finetune import family_name


def main():
    base=Path('tmp/field_finetune_20261021/dataset')
    out=Path('tmp/field_finetune_20261021/dataset_supplement_v2')
    if out.exists():
        raise SystemExit('Choose a new output directory; refusing overwrite')
    manifest=json.loads((base/'manifest.json').read_text())
    names=manifest['names'];rng=random.Random(20261022)
    for split in ('train','val'):
        for kind in ('images','labels'):
            (out/kind/split).mkdir(parents=True,exist_ok=True)
    patches=[]
    raw_root=Path('tmp/detector_validation_20261021/torchroyale_supplement/train')
    source_names=yaml.safe_load(Path('tmp/detector_validation_20261021/torchroyale_torchroyale-enemies_data.yaml').read_text())['names']
    for label in sorted((raw_root/'labels').glob('*.txt')):
        source=list((raw_root/'images').glob(label.stem+'.*'))
        if not source:continue
        im=Image.open(source[0]).convert('RGB');w,h=im.size
        for line in label.read_text().splitlines():
            c,x,y,bw,bh=map(float,line.split());raw=source_names[int(c)]
            if raw not in {'boss-bandit','guardian','evo-musketeer','evo-skeleton-barrel'}:continue
            name={'guardian':'royal-guardian','evo-musketeer':'musketeer','evo-skeleton-barrel':'skeleton-barrel'}.get(raw,raw)
            box=(max(0,round((x-bw/2)*w)),max(0,round((y-bh/2)*h)),min(w,round((x+bw/2)*w)),min(h,round((y+bh/2)*h)))
            patches.append((im.crop(box),name,{'source':str(source[0]),'bbox':box,'raw_name':raw}))
    # Manual visible-object boxes, in a 360x360 enlarged view of each crop.
    # Ambiguous/occluded crops (including the only Skeleton Army evo) excluded.
    crop_boxes={11:(39,70,338,321),14:(32,102,327,286),23:(72,69,279,338)}
    crop_rows=json.loads(Path('tmp/detector_validation_20261021/rare_crop_images.json').read_text())
    for index,box in crop_boxes.items():
        row=crop_rows[index];im=Image.open(row['path']).convert('RGB');w,h=im.size
        actual=tuple(round(v*(w if i%2==0 else h)/360) for i,v in enumerate(box))
        patches.append((im.crop(actual),family_name(row['raw_name']),
                        {'source':row['path'],'bbox':actual,'raw_name':row['raw_name'],'annotation':'provisional visual bbox; source user class'}))
    if not patches:raise RuntimeError('No rare object patches available')
    images=sorted((base/'images/train').glob('*.jpg'));rng.shuffle(images)
    # Replay old examples during adaptation to reduce forgetting.
    for path in images[:768]:
        shutil.copy2(path,out/'images/train'/path.name)
        shutil.copy2(base/'labels/train'/(path.stem+'.txt'),out/'labels/train'/(path.stem+'.txt'))
    for path in (base/'images/val').glob('*.jpg'):
        shutil.copy2(path,out/'images/val'/path.name)
        shutil.copy2(base/'labels/val'/(path.stem+'.txt'),out/'labels/val'/(path.stem+'.txt'))
    counts=Counter();records=[]
    for i in range(384):
        path=images[i%len(images)];im=Image.open(path).convert('RGB');W,H=im.size
        labels=[list(map(float,line.split())) for line in (base/'labels/train'/(path.stem+'.txt')).read_text().splitlines()]
        patch,name,provenance=patches[i%len(patches)]
        target_h=rng.randint(40,90);target_w=max(8,round(patch.width/patch.height*target_h))
        patch=patch.resize((target_w,target_h));x=rng.randint(20,W-target_w-20);y=rng.randint(160,H-target_h-70)
        # Reject placements obscuring any labelled source object, rather than
        # leaving visible fragments as unlabelled negatives.
        def overlaps(row):
            _,cx,cy,bw,bh=row
            return min((cx+bw/2)*W,x+target_w)>max((cx-bw/2)*W,x) and min((cy+bh/2)*H,y+target_h)>max((cy-bh/2)*H,y)
        for _ in range(100):
            if not any(overlaps(row) for row in labels):break
            x=rng.randint(20,W-target_w-20);y=rng.randint(160,H-target_h-70)
        else:continue
        im.paste(patch,(x,y));labels.append([names.index(name),(x+target_w/2)/W,(y+target_h/2)/H,target_w/W,target_h/H])
        stem=f'rare_{i:04d}';im.save(out/'images/train'/(stem+'.jpg'),quality=95)
        (out/'labels/train'/(stem+'.txt')).write_text('\n'.join(str(int(row[0]))+' '+' '.join(f'{v:.7f}' for v in row[1:]) for row in labels)+'\n')
        counts[name]+=1;records.append({'image':stem,'background':str(path),'patch':provenance})
    data={'path':str(out.resolve()),'train':'images/train','val':'images/val','nc':len(names),'names':names}
    (out/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
    report={'names':names,'patches':[p[2] for p in patches],'augmented_instances':dict(counts),'records':records,
            'limitations':['Rectangular synthetic paste, not independent new real frames.','Rare source tracks correlated; Boss Bandit only four real examples.',
                           'No team supervision.','TV Royale remains outside training.','Validation lacks Boss Bandit and Guardian: their accuracy remains unmeasured.']}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'patches':len(patches),'counts':dict(counts),'train_images':len(list((out/'images/train').glob('*.jpg')))},indent=2))


if __name__=='__main__':main()
