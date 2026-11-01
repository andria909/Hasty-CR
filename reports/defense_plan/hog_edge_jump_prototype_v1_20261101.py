"""Diagnostic edge-lane jump prototype; never installed in training Battle.

Tests a fixed heading toward the far bridge exit, followed by ordinary movement.
Does not model central river jumps, native target locks, collision or takeoff
animation. Recorded context is still imposed by the conditioned replay.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

from scripts.gamestate import replay_recorded_episode as recorded
from sim.arena import Point
from sim.engine import Battle


class EdgeJumpBattle(Battle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.experimental_jumps={}
        self.experimental_completed=set()

    def _move(self, entity, dt):
        target=self.entities.get(entity.target_uid)
        if (entity.uid not in self.experimental_jumps and entity.uid not in self.experimental_completed
                and entity.name=='hog_rider' and entity.jump_enabled and entity.jump_speed_mt_per_sec>0
                and target is not None and target.alive
                and (target.pos.y-16000)*(entity.pos.y-16000)<0):
            p=entity.pos
            if ((p.x<2000 or p.x>16000) and
                    ((entity.side==-1 and 14000<=p.y<15500) or
                     (entity.side==1 and 16500<p.y<=18000))):
                waypoint=Point(3500 if p.x<9000 else 14500,18000 if entity.side==-1 else 14000)
                dx,dy=waypoint.x-p.x,waypoint.y-p.y
                length=math.hypot(dx,dy)
                self.experimental_jumps[entity.uid]=(dx/length,dy/length)
        if entity.uid in self.experimental_jumps:
            dx,dy=self.experimental_jumps[entity.uid]
            step=entity.jump_speed_mt_per_sec*dt/1000
            entity.pos=Point(entity.pos.x+round(dx*step),entity.pos.y+round(dy*step))
            if ((entity.side==-1 and entity.pos.y>=16500) or
                    (entity.side==1 and entity.pos.y<=15500)):
                del self.experimental_jumps[entity.uid]
                self.experimental_completed.add(entity.uid)
            return
        return super()._move(entity,dt)


def compare(clip):
    baseline=recorded.replay(clip,battle_options={'hog_sight_mode':'side_clip_v1'})
    with patch.object(recorded,'Battle',EdgeJumpBattle):
        candidate=recorded.replay(clip,battle_options={'hog_sight_mode':'side_clip_v1'})
    assert recorded.Battle is Battle
    return dict(name=clip['name'],source=clip['source'],source_sha256=clip['source_sha256'],
        focal_identity=clip['focal_identity'],world=clip['world'],
        baseline=baseline,prototype=candidate)


def eligible_anchor(case):
    points=case['points']
    # Choose the last stationary anchor before the first motion. No future
    # velocities, endpoint, fitted direction, or error enter the prototype.
    for a,b in zip(points,points[1:]):
        if (a['x'],a['y'])==(b['x'],b['y']):continue
        side=case['identity'][-1]
        if ((a['x']<2000 or a['x']>16000) and
                ((side==0 and 14000<=a['y']<15500) or(side==1 and 16500<a['y']<=18000))):
            return a
        return None
    return None


def audit(inventory,fixtures,output):
    inventory,fixtures,output=map(Path,(inventory,fixtures,output))
    if output.exists():raise ValueError('Preserve existing diagnostic output')
    data=json.loads(inventory.read_text());fixture=json.loads(fixtures.read_text())
    expected={s['path']:s['sha256']for s in data['sources']}
    picked=[(i,c,eligible_anchor(c))for i,c in enumerate(data['cases'])]
    picked=[item for item in picked if item[2]is not None]
    for source in sorted({c['source']for _,c,_ in picked}):
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=expected[source]:
            raise ValueError('Native source hash changed')
    results=[];excluded=[]
    for index,case,anchor in picked:
        try:
            clip=recorded.extract(case['source'],anchor['frame'],case['points'][-1]['tick'],
                case['identity'][0],f'edge_crossing_{index}')
        except ValueError as exc:
            excluded.append(dict(case=index,source=case['source'],reason=str(exc)));continue
        results.append(dict(case=index,**compare(clip)))
        print(json.dumps(dict(event='EDGE_REPLAY',complete=len(results),excluded=len(excluded),eligible=len(picked))),flush=True)
    controls=[compare(c['clip'])for c in fixture['full_crossing_controls']]
    deltas=[r['prototype']['mean_position_error_tiles']-r['baseline']['mean_position_error_tiles']for r in results]
    summary=dict(eligible=len(picked),replayed=len(results),excluded=len(excluded),
        source_world_groups=len({(r['source'],json.dumps(r['world']))for r in results}),
        better=sum(d<-.02 for d in deltas),worse=sum(d>.02 for d in deltas),unchanged=sum(abs(d)<=.02 for d in deltas),
        mean_error_tiles={k:sum(r[k]['mean_position_error_tiles']for r in results)/len(results)
            for k in ('baseline','prototype')} if results else {})
    report=dict(schema='experimental-native-hog-edge-jump-v1',summary=summary,results=results,
        excluded=excluded,full_controls=controls,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in (inventory,fixtures)},
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in
            (Path(__file__),Path(recorded.__file__),Path('sim/engine.py'))},
        accepted_physics_change=False,live_games=False,
        limitations=['Discovery-only, correlated episodes, previously inspected data; not a new holdout.',
            'Only outer-edge jumps; central crossings and full battles are not validated.',
            'Takeoff is approximated from public position; native jump/target states are unavailable.',
            'Other bodies follow the recorded context; damage and collision separation are not simulated.',
            'This subclass is used only by this diagnostic and is absent from training physics.'])
    with output.open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('inventory','fixtures','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(audit(a.inventory,a.fixtures,a.output)))
