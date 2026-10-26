"""Reproducible, read-only native projectile velocity screening."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from scripts.gamestate.audit_recorded_fireball import fit
from scripts.gamestate.audit_recorded_tick_speeds import quantiles

RAW_SPEEDS={'TowerPrincessProjectile':600,'MusketeerProjectile':1000,
    'FireSpiritsProjectile':400,'ArrowsSpell':1100,'KingProjectile':1000,
    'FireballSpell':600,'chr_wizardProjectile':600,'RocketSpell':350,
    'SpearGoblinProjectile':500,'BombSkeletonProjectile':400}


def audit(paths,clock_report):
    tracks=defaultdict(dict);sources=[]
    for index,path in enumerate(map(Path,paths)):
        h=hashlib.sha256();segment=0;last=None;last_tick=None
        with path.open('rb')as stream:
            for raw in stream:
                h.update(raw);r=json.loads(raw);f=r.get('frame',{});tick=f.get('battle_tick')
                if r.get('status')!='ok' or type(tick)is not int:continue
                identity=tuple(sorted(e['address']for e in f.get('entities',[])if e.get('name')=='KingTower'))
                if len(identity)!=2:continue
                if identity!=last or(last_tick is not None and tick<last_tick-20):segment+=1
                last=identity;last_tick=tick
                for e in f.get('entities',[]):
                    name=e.get('name')
                    if name not in RAW_SPEEDS:continue
                    key=(index,segment,name,e['entity_id'],e['address']);v=(tick,e['x'],e['y'])
                    if tick in tracks[key] and tracks[key][tick]!=v:tracks[key][tick]=None
                    elif tick not in tracks[key]:tracks[key][tick]=v
        sources.append(dict(path=str(path),sha256=h.hexdigest()))
    clocks=json.loads(Path(clock_report).read_text())
    if {s['sha256']for s in clocks['sources']}!={s['sha256']for s in sources}:raise ValueError('Clock captures differ')
    if not 19.5<clocks['device_clock_tick_rate_hz']['median']<20.5:raise ValueError('20Hz clock not supported')
    values=defaultdict(list)
    for key,track in tracks.items():
        result=fit([v for _,v in sorted(track.items())if v is not None])
        if result:values[key[2]].append(result['speed_millitiles_per_tick']*20)
    return dict(schema='native-projectile-velocity-screening-v1',sources=sources,
        clock_sha256=hashlib.sha256(Path(clock_report).read_bytes()).hexdigest(),
        projectiles={n:dict(speed_mt_s=quantiles(v),raw_speed=RAW_SPEEDS[n],
            predicted_mt_s=RAW_SPEEDS[n]*20)for n,v in sorted(values.items())},
        limitations=['XY speeds only; impact, target selection and AOE are not validated',
                      'Repeated tracks within matches are correlated'],physics_fully_validated=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--clock-report',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=audit(a.paths,a.clock_report);Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r['projectiles']))
