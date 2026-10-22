"""Bounded offline motion/field-coverage audit; no Android or inference calls."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def distribution(values):
    if not values:return None
    values=sorted(values)
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),
                p05=values[max(0,math.ceil(.05*len(values))-1)],
                p95=values[math.ceil(.95*len(values))-1],min=values[0],max=values[-1])


def sample_rows(path,max_rows,max_bytes):
    rows=[];digest=hashlib.sha256();size=0
    with Path(path).open('rb') as stream:
        for line in stream:
            if len(rows)>=max_rows or size+len(line)>max_bytes:break
            digest.update(line);size+=len(line)
            if line.strip():rows.append(json.loads(line))
    return rows,dict(path=str(path),file_bytes=Path(path).stat().st_size,
                     sampled_rows=len(rows),sampled_bytes=size,sampled_prefix_sha256=digest.hexdigest(),
                     full_file_read=size==Path(path).stat().st_size)


def make_tracks(rows,source):
    tracks=defaultdict(list);fields=Counter();active=0;segments=0;last_world=None;last_tick=None
    for index,row in enumerate(rows):
        for unit in row.get('units',[]):fields.update(unit.keys())
        own=row.get('own_state',{});frame=row.get('frame',{})
        if row.get('status')!='ok' or own.get('status')!='ok':continue
        world,tick=own.get('world'),own.get('tick')
        device=frame.get('device_start_ns')
        if not world or not isinstance(tick,int) or not isinstance(device,int):continue
        if world!=last_world or last_tick is None or tick<last_tick:
            segments+=1
        last_world,last_tick=world,tick
        if any(u['name']=='KingTower' and u.get('hp',0)<=0 for u in row.get('units',[])):continue
        active+=1
        bodies=row.get('units',[])
        for unit in bodies:
            if unit.get('hp',0)<=0 or unit['name'] in ('KingTower','PrincessTower'):continue
            key=(source,segments,world,unit['entity_id'],unit.get('address'),unit['name'],unit['side'])
            # One sample per engine tick avoids pretending repeated clock reads
            # are distinct game-time movement measurements.
            if tracks[key] and tick==tracks[key][-1]['tick']:continue
            tracks[key].append(dict(tick=tick,device=device/1e9,body=unit,index=index,bodies=bodies))
    return tracks,fields,dict(active_clock_rows=active,sampled_battle_segments=segments)


def motion_windows(track,profiles):
    """Nonoverlapping, straight, HP-stable, isolated motion candidates."""
    accepted=[];rejected=Counter();start=0
    if len(track)<2:return accepted,dict(short_track=1)
    while start<len(track)-1:
        a=track[start]
        if a['tick']-track[0]['tick']<20:
            start+=1;continue
        end=start+1
        while end<len(track) and track[end]['tick']-a['tick']<12:end+=1
        if end>=len(track):break
        points=track[start:end+1];b=points[-1]
        reason=None
        if b['tick']-a['tick']>16:reason='window_game_time_gap'
        elif any(v['tick']-u['tick']>5 or v['device']-u['device']>.3 for u,v in zip(points,points[1:])):reason='capture_gap'
        elif any(p['body']['hp']!=a['body']['hp'] for p in points):reason='hp_changed'
        path=sum(math.hypot(v['body']['x']-u['body']['x'],v['body']['y']-u['body']['y']) for u,v in zip(points,points[1:]))
        chord=math.hypot(b['body']['x']-a['body']['x'],b['body']['y']-a['body']['y'])
        if reason is None and(path<=0 or chord/path<.985):reason='stationary_or_turning'
        if reason is None:
            steps=[math.hypot(v['body']['x']-u['body']['x'],v['body']['y']-u['body']['y'])/((v['tick']-u['tick'])*.05) for u,v in zip(points,points[1:])]
            if not steps or statistics.mean(steps)<=0 or statistics.pstdev(steps)/statistics.mean(steps)>.25:reason='uneven_motion'
        if reason is None:
            profile=profiles.get(a['body']['name'])
            # Unknown attack range is not filled in from a guessed base card.
            if profile is None:reason='unknown_profile'
            else:
                for p in points:
                    body=p['body']
                    for other in p['bodies']:
                        if other['entity_id']==body['entity_id'] or other.get('hp',0)<=0:continue
                        separation=math.hypot(other['x']-body['x'],other['y']-body['y'])
                        if other['side']==body['side']:
                            minimum=3000 if other['name'] in ('KingTower','PrincessTower') else 2000
                        else:minimum=max(3000,profile.range_mt+1500)
                        if separation<minimum:
                            reason='near_other_body_or_tower';break
                    if reason:break
        if reason:
            rejected[reason]+=1;start+=1;continue
        seconds=(b['tick']-a['tick'])*.05
        device=b['device']-a['device']
        accepted.append(dict(game_tiles_per_second=chord/1000/seconds,
                             device_tiles_per_second=chord/1000/device,
                             game_seconds_per_device_second=seconds/device,
                             start_tick=a['tick'],end_tick=b['tick']))
        start=end+1
    return accepted,dict(rejected)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile-root',type=Path,required=True)
    p.add_argument('--trace',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-rows',type=int,default=6000)
    p.add_argument('--max-bytes',type=int,default=24*1024*1024)
    args=p.parse_args()
    if args.max_rows<=0 or args.max_bytes<=0:raise ValueError('Positive sample limits required')
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.profile_root.resolve()))
    from sim.gamedata import load_characters,to_snake_case
    from sim.defense_assets import load_defense_assets
    from sim.entities import speed_to_mt_per_sec
    from sim import arena
    cards,_,_,_=load_defense_assets()
    table=load_characters(level=11)
    canonical={to_snake_case(spec.name):spec for spec in table.values()}
    canonical.update({to_snake_case(c.unit.name):c.unit for c in cards.values()if c.unit is not None})
    native_profiles={}
    tracks={};all_fields=Counter();coverage=[];observed=Counter();towers={}
    for path in args.trace:
        rows,source=sample_rows(path,args.max_rows,args.max_bytes)
        native_names={u['name']for row in rows for u in row.get('units',[])}
        profiles={name:canonical.get(to_snake_case(name))for name in native_names}
        native_profiles.update(profiles)
        current,fields,cov=make_tracks(rows,str(path))
        tracks.update(current);all_fields.update(fields);source.update(cov);coverage.append(source)
        for row in rows:
            for unit in row.get('units',[]):
                observed[unit['name']]+=1
                if unit['name'] in ('KingTower','PrincessTower'):
                    towers[(unit['side'],unit['name'],unit['x'],unit['y'])]=True
    values=defaultdict(list);body_medians=defaultdict(list);body_counts=Counter();rejects=Counter();examples=[]
    for key,track in tracks.items():
        name=key[-2]
        windows,reasons=motion_windows(track,native_profiles)
        rejects.update(reasons)
        if not windows:continue
        values[name].extend(windows);body_counts[name]+=1
        body_medians[name].append(statistics.median(w['game_tiles_per_second']for w in windows))
        if len(examples)<30:examples.append(dict(source=key[0],segment=key[1],name=name,side=key[-1],entity_id=key[3],address=key[4],windows=windows[:3]))
    speeds={}
    for name,windows in sorted(values.items()):
        profile=native_profiles[name];nominal=speed_to_mt_per_sec(profile.speed_mt_per_sec)/1000
        median=statistics.median(body_medians[name])
        speeds[name]=dict(nominal_current_profile_tiles_per_second=nominal,
                         nonoverlapping_windows=len(windows),body_tracks=body_counts[name],
                         game_time_speed=distribution([w['game_tiles_per_second']for w in windows]),
                         device_time_speed=distribution([w['device_tiles_per_second']for w in windows]),
                         equally_weighted_body_medians=distribution(body_medians[name]),
                         body_median_vs_nominal_percent=(median/nominal-1)*100 if nominal else None,
                         game_clock_rate=distribution([w['game_seconds_per_device_second']for w in windows]),
                         profile_has_charge=profile.charge_range_mt>0,
                         profile_has_river_jump=profile.jump_enabled,
                         profile_limitations=profile.raw.get('_public_profile_limitations',[]))
    target_fields={k:n for k,n in all_fields.items()if 'target' in k.lower()}
    report=dict(schema='bounded-arm64-mechanics-audit-v1',sources=coverage,
                current_profile_root=str(args.profile_root.resolve()),
                profile_source_sha256={f:hashlib.sha256((args.profile_root/f).read_bytes()).hexdigest()for f in ['sim/public_gamedata.py','sim/entities.py','sim/arena.py']},
                speed=speeds,rejected_candidate_windows=dict(rejects),sample_examples=examples,
                observed_names=dict(observed),unmapped_names=sorted(name for name in observed if name not in ('KingTower','PrincessTower')and native_profiles.get(name)is None),
                actual_target_fields=target_fields,targeting=dict(verified_actual_target_uid_labels=0,accuracy_percent=None,status='not_measurable_without_actual_target_UIDs; no HP/geometry proxy labels used'),
                native_tower_positions=[dict(side=k[0],name=k[1],x=k[2],y=k[3])for k in sorted(towers)],
                sim_tower_positions=dict(enemy_king=arena.ENEMY_KING.__dict__,ally_king=arena.ALLY_KING.__dict__,enemy_princess={k:v.__dict__ for k,v in arena.ENEMY_PRINCESS.items()},ally_princess={k:v.__dict__ for k,v in arena.ALLY_PRINCESS.items()}),
                limitations=['Bounded file prefixes; this is not all 25 matches or an independent benchmark.',
                             'Engine clock interpreted as20Hz per reader contract; no independent pixel-clock audit performed here.',
                             'Filters identify free-motion candidates, not proof of absent buffs, charge, attack state or network interpolation.',
                             'Nonoverlapping windows from one body remain dependent; body-track medians are also reported.',
                             'Native level_index and movement/status effects are not calibrated; profiles use explicit level11.',
                             'No actual attack target UID in sampled schema; targeting accuracy is unavailable.',
                             'No device access, new battles, screenshot inference, training or changes to simulator physics.'])
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(sources=coverage,speed={k:{x:v[x]for x in ['nominal_current_profile_tiles_per_second','nonoverlapping_windows','body_tracks','body_median_vs_nominal_percent']}for k,v in speeds.items()},unmapped=report['unmapped_names'],targeting=report['targeting']),indent=2))


if __name__=='__main__':main()
