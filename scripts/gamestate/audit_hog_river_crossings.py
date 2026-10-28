"""Inventory continuous Hog river crossings; discovery data, not a jump fix.

Keep all observed speeds, including deployment, stalls and knockback. A band
of speeds near 3.2 tiles/s is descriptive evidence of an omitted movement
state, not a filter that proves every such interval is a jump.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from scripts.gamestate.audit_targeting_history import clock_frame,identity
from scripts.gamestate.audit_recorded_tick_speeds import quantiles


def crossings(rows):
    active={};previous=None;result=[]
    def finish(key):
        points=active.pop(key)
        if (len(points)>=5 and min(p['y']for p in points)<=15000
                and max(p['y']for p in points)>=17000):
            result.append(dict(world=key[0],identity=key[1],points=points))
    for index,row in enumerate(rows):
        frame=clock_frame(row)
        if (frame is None or previous is not None and (frame['world']!=previous['world']
                or not 0<=frame['tick']-previous['tick']<=5)):
            for key in list(active):finish(key)
        previous=frame
        if frame is None:continue
        present=set()
        for body in frame['bodies'].values():
            if body['name']!='HogRider' or body['hp']<=0 or not 14000<=body['y']<=18000:
                continue
            key=(frame['world'],identity(body));present.add(key)
            points=active.setdefault(key,[])
            point=dict(tick=frame['tick'],x=body['x'],y=body['y'],hp=body['hp'],frame=index)
            if points and point['tick']==points[-1]['tick']:
                if any(point[k]!=points[-1][k]for k in ('x','y','hp')):
                    finish(key)
                    # Reject this contradictory tick as an anchor too.
                    active[key]=[]
                continue
            points.append(point)
        for key in list(active):
            if key not in present:finish(key)
    for key in list(active):finish(key)
    return result


def audit(sources):
    cases=[];pins=[];bins=Counter();fast=[]
    boundaries=(0,1000,2000,2600,3000,3400,float('inf'))
    labels=('stationary','0..1000','1000..2000','2000..2600','2600..3000','3000..3400','above3400')
    for source in sources:
        path=Path(source['path']);raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=source['sha256']:raise ValueError('Source changed: '+str(path))
        pins.append(dict(path=str(path),sha256=digest))
        for case in crossings(json.loads(line)for line in raw.splitlines()):
            intervals=[]
            for a,b in zip(case['points'],case['points'][1:]):
                speed=math.hypot(b['x']-a['x'],b['y']-a['y'])*20/(b['tick']-a['tick'])
                bins[next(label for limit,label in zip(boundaries,labels)if speed<=limit)]+=1
                if 3000<speed<=3400:fast.append(speed)
                intervals.append(dict(start_tick=a['tick'],end_tick=b['tick'],speed_mt_per_sec=speed))
            cases.append(dict(source=str(path),**case,intervals=intervals))
    return dict(schema='native-hog-river-crossings-v1',sources=pins,cases=cases,
        summary=dict(crossings=len(cases),worlds=len({str(c['world'])for c in cases}),
                     interval_speed_bins=dict(bins),conditional_3000_3400_speed=quantiles(fast)),
        limitations=['Complete traversal of y15000..17000 inside y14000..18000 observation band.',
            'Clock/identity gaps split tracks; no missing samples are filled.',
            'Speeds use 20Hz native ticks; reads are not atomic with the game update.',
            'All speeds retained, including deploy/stun/collision/knockback and sampling error.',
            'The conditional speed statistic is not an unbiased estimate or jump label.',
            'Same-match crossings are correlated; no new simulator physics accepted.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sources={}
    for path in args.manifests:
        for item in json.loads(path.read_text())['sources']:
            if item['path']in sources and sources[item['path']]['sha256']!=item['sha256']:
                raise ValueError('Conflicting source hashes')
            sources[item['path']]=item
    result=audit(list(sources.values()))
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x')as f:json.dump(result,f,separators=(',',':'));f.write('\n')
    print(json.dumps(result['summary']))
