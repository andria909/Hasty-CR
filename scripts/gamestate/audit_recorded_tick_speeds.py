"""Descriptive native motion audit; no emulator, hooks, battles or physics edits.

Private-clock validation does not bracket the preceding entity reads. Long
constant-motion windows reduce, but cannot eliminate, tick-assignment error.
Buffs/charge/attack targets are absent, so this is not a mechanics acceptance.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def quantiles(values):
    values=sorted(values)
    if not values:return dict(n=0,p10=None,median=None,p90=None)
    def q(p):
        i=p*(len(values)-1);lo=math.floor(i);hi=math.ceil(i)
        return values[lo]+(values[hi]-values[lo])*(i-lo)
    return dict(n=len(values),p10=q(.1),median=q(.5),p90=q(.9))


def motion_windows(samples):
    """Samples: tick,x,y,hp. Nonoverlapping 1-second windows at declared20Hz."""
    result=[];index=0
    while index<len(samples):
        start=samples[index][0];end=index
        while end<len(samples)and samples[end][0]<start+20:end+=1
        if end==len(samples):break
        window=samples[index:end+1];index=end+1
        if not 20<=window[-1][0]-start<=25 or len(window)<6:continue
        if len({s[3]for s in window})!=1:continue
        lengths=[];speeds=[]
        for a,b in zip(window,window[1:]):
            ticks=b[0]-a[0]
            if not 0<ticks<=5:break
            distance=math.hypot(b[1]-a[1],b[2]-a[2]);lengths.append(distance);speeds.append(distance/ticks)
        else:
            mean=statistics.mean(speeds)
            if mean<5 or statistics.pstdev(speeds)/mean>.08:continue
            chord=math.hypot(window[-1][1]-window[0][1],window[-1][2]-window[0][2])
            if chord<=0 or sum(lengths)/chord>1.01:continue
            result.append(chord/(window[-1][0]-start))
    return result


def audit(paths):
    counters=Counter();tracks=defaultdict(dict);names={};clock=[];sources=[];segment=0
    for source_index,path in enumerate(map(Path,paths)):
        digest=hashlib.sha256();last_identity=None;last_tick=None;clock_start=None
        with path.open('rb')as stream:
            for raw in stream:
                digest.update(raw);d=json.loads(raw);counters['rows']+=1
                frame=d.get('frame',{});own=d.get('own_state',{})
                if d.get('status')!='ok':
                    last_identity=None;clock_start=None;continue
                tick=frame.get('battle_tick')
                if own.get('status')!='ok' or type(tick)is not int or own.get('tick')!=tick or not own.get('world'):
                    counters['rejected_clock_frame']+=1;continue
                units=d.get('units',[])
                kings=tuple(sorted(u['address']for u in units if u.get('name')=='KingTower'))
                identity=(own['world'],kings)
                if identity!=last_identity or (last_tick is not None and tick<last_tick):
                    segment+=1;clock_start=None
                last_identity=identity;last_tick=tick;counters['valid_clock_frames']+=1
                device_ns=frame.get('device_end_ns')
                if type(device_ns)is int:
                    if clock_start is None:clock_start=(tick,device_ns)
                    elif tick-clock_start[0]>=20:
                        dt=(device_ns-clock_start[1])/1e9
                        if dt>0 and tick-clock_start[0]<=40:clock.append((tick-clock_start[0])/dt)
                        clock_start=(tick,device_ns)
                for u in units:
                    uid=u.get('entity_id');hp=u.get('hp')
                    if type(uid)is not int or uid//1000000!=5 or not isinstance(hp,(int,float))or hp<=0:continue
                    if u.get('name') in ('KingTower','PrincessTower'):continue
                    key=(source_index,segment,uid,u.get('address'),u.get('name'),u.get('side'))
                    names[key]=u.get('name');sample=(tick,u['x'],u['y'],hp)
                    if tick in tracks[key]and tracks[key][tick]!=sample:
                        counters['conflicting_same_tick_samples']+=1;tracks[key][tick]=None
                    elif tick not in tracks[key]:tracks[key][tick]=sample
        sources.append(dict(path=str(path),sha256=digest.hexdigest(),bytes=path.stat().st_size))
    grouped=defaultdict(list);track_counts=Counter();segments=defaultdict(set)
    for key,by_tick in tracks.items():
        samples=[s for _,s in sorted(by_tick.items())if s is not None]
        windows=motion_windows(samples)
        if windows:
            name=names[key];grouped[name]+=windows;track_counts[name]+=1;segments[name].add(key[1])
    from sim.gamedata import load_gamedata
    from sim.entities import speed_to_mt_per_sec
    normalize=lambda s:''.join(c.lower()for c in s if c.isalnum())
    simulation=defaultdict(set)
    for spec in load_gamedata().values():
        if spec.unit is not None:simulation[normalize(spec.unit.name)].add(speed_to_mt_per_sec(spec.unit.speed_mt_per_sec))
    rows={}
    for name,values in sorted(grouped.items()):
        q=quantiles([v*20 for v in values]);predictions=simulation.get(normalize(name),set())
        expected=next(iter(predictions))if len(predictions)==1 else None
        bins=Counter(round(v*20/50)*50 for v in values)
        rows[name]=dict(speed_millitiles_per_tick=quantiles(values),speed_millitiles_per_second_at_20hz=q,
            constant_motion_tracks=track_counts[name],recording_segments=len(segments[name]),
            modal_50_millitile_per_second_bin=bins.most_common(1)[0][0],
            simulator_unbuffed_speed_millitiles_per_second=expected,
            simulator_speed_candidates=sorted(predictions),
            median_over_simulator=q['median']/expected if expected else None)
    clock_summary=quantiles(clock)
    source_fingerprints={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()for name in
        ('sim/entities.py','sim/gamedata.py','scripts/gamestate/native/arm64_entities.c',
         'scripts/gamestate/audit_recorded_tick_speeds.py')}
    return dict(schema='recorded-native-tick-speed-audit-v1',sources=sources,source_fingerprints=source_fingerprints,counters=dict(counters),
        recording_segments=segment,device_clock_tick_rate_hz=clock_summary,
        declared_tick_rate_hz=20,clock_median_compatible_with_20hz=(19<=clock_summary['median']<=21 if clock else None),
        motion_windows=dict(duration_ticks_min=20,duration_ticks_max=25,min_samples=6,max_gap_ticks=5,
            max_interval_speed_cv=.08,max_path_length_over_chord=1.01,hp_change_allowed=False),
        units=rows,physics_accepted=False,simulator_changed=False,live_games_started=False,
        limitations=['Entity reads precede tick read; no atomic tick bracketing for coordinates',
            'Own clock metadata validated by reader, but world/hand consistency is not targeting evidence',
            'Unknown buffs, charge and attack state; constant motion is not necessarily unbuffed motion',
            'Correlated windows/tracks are not independent match trials; segments may split one match',
            'Coordinates/targets/hitboxes and pathing parity require separate validation'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();result=audit(a.paths);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(clock=result['device_clock_tick_rate_hz'],units={n:v['speed_millitiles_per_second_at_20hz'] for n,v in result['units'].items()})))
