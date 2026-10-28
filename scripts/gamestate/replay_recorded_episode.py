"""Small, inspectable real-to-sim movement/targeting replay with measured context.

The focal troop evolves freely. Other bodies follow the recorded positions,
HP, appearances and removals at native observation times (zero-order hold).
Their AI is not run. This isolates a trajectory defect without inventing all
unobserved attack phases, levels, elixir or targets of a full match.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from scripts.gamestate.audit_sim_alignment import unit_profiles
from scripts.gamestate.audit_targeting_history import clock_frame, identity
from sim.arena import Point
from sim.engine import Battle
from sim.entities import make_tower, make_unit
from sim.gamedata import to_snake_case


def extract(path, start_frame, end_tick, focal_uid, name):
    raw = Path(path).read_bytes()
    rows = raw.splitlines()
    first = clock_frame(json.loads(rows[start_frame]))
    if first is None or focal_uid not in first['bodies']:
        raise ValueError('Invalid episode anchor')
    focal_identity = identity(first['bodies'][focal_uid])
    frames = []
    previous = first['tick']
    for index in range(start_frame, len(rows)):
        row = json.loads(rows[index])
        frame = clock_frame(row)
        if frame is None:
            raise ValueError(f'Invalid clock at row {index}')
        if frame['world'] != first['world']:
            break
        if frame['tick'] > end_tick:
            break
        if not 0 <= frame['tick']-previous <= 5:
            raise ValueError(f'Gap/rewind at row {index}')
        body = frame['bodies'].get(focal_uid)
        if body is None or identity(body) != focal_identity:
            raise ValueError('Focal body disappeared or its ID was reused')
        if frames and frame['tick'] == previous:
            # Not atomic reads: a changed body at the same tick is not a
            # second independent observation. Reject instead of cherry-picking.
            prior = {b['entity_id']: b for b in frames[-1]['bodies']}
            if any(prior.get(uid) != b for uid, b in frame['bodies'].items()) or len(prior) != len(frame['bodies']):
                raise ValueError(f'Conflicting duplicate tick at row {index}')
            continue
        frames.append(dict(frame_index=index, tick=frame['tick'],
                           bodies=list(frame['bodies'].values())))
        previous = frame['tick']
    if len(frames) < 3 or end_tick-frames[-1]['tick'] > 5:
        raise ValueError('Episode ended before the requested window')
    return dict(schema='native-conditioned-episode-v1', name=name,
        source=str(path), source_sha256=hashlib.sha256(raw).hexdigest(),
        world=first['world'], focal_identity=focal_identity, frames=frames)


def make_body(body, profiles):
    uid, side = body['entity_id'], 1 if body['side'] == 1 else -1
    point = Point(body['x'], body['y'])
    if body['name'] in ('KingTower', 'PrincessTower'):
        result = make_tower(uid, side, point, body['hp'], 0, 1000, 7500,
                            king=body['name'] == 'KingTower')
    else:
        profile = profiles.get(to_snake_case(body['name']))
        if profile is None:
            return None
        result = make_unit(uid, profile, side, point)
    result.hitpoints = body['hp']
    result.max_hitpoints = body['max_hp']
    result.deploy_remaining_ms = 0
    # Damage/level calibration is separate. Attack occurrences remain visible,
    # but must not kill a context body before its observed native removal.
    result.damage = 0
    return result


def replay(clip, *, initial_target_uid=None, sight_override=None):
    profiles = unit_profiles()
    frames = clip['frames']
    uid = clip['focal_identity'][0]
    observed = {f['tick']: f for f in frames}
    focal_body = next(b for b in frames[0]['bodies'] if b['entity_id'] == uid)
    battle = Battle()
    focal = make_body(focal_body, profiles)
    if focal is None:
        raise ValueError('Focal profile is unavailable')
    battle.add(focal)
    focal.target_uid = initial_target_uid
    if sight_override is not None:
        focal.sight_range_mt = sight_override
    rows, target_changes, shots = [], [], []
    ignored = Counter()
    current_frame = frames[0]
    previous_target = None
    previous_projectile_count = 0
    for tick in range(frames[0]['tick'], frames[-1]['tick']+1):
        if tick in observed:
            current_frame = observed[tick]
            for old_uid in list(battle.entities):
                if old_uid != uid:
                    del battle.entities[old_uid]
            for body in current_frame['bodies']:
                if body['entity_id'] == uid:
                    continue
                entity = make_body(body, profiles)
                if entity is None:
                    ignored[body['name']] += 1
                else:
                    battle.entities[entity.uid] = entity
            battle._buildings_stamp = -1
        battle.now_ms = tick*50
        if tick > frames[0]['tick']:
            battle._acquire_target(focal)
            if not battle._attack(focal, 50):
                if focal.windup_remaining_ms > 0 and focal.target_uid is not None:
                    focal.windup_remaining_ms -= 50
                battle._move(focal, 50)
        if focal.target_uid != previous_target:
            target_changes.append(dict(tick=tick, target_uid=focal.target_uid))
            previous_target = focal.target_uid
        if len(battle.in_flight) > previous_projectile_count:
            shots.append(dict(tick=tick, target_uid=focal.target_uid,
                              due_ms=battle.in_flight[-1][0]))
            previous_projectile_count = len(battle.in_flight)
        if tick in observed:
            native = next(b for b in current_frame['bodies'] if b['entity_id'] == uid)
            error = math.hypot(focal.pos.x-native['x'], focal.pos.y-native['y'])
            rows.append(dict(tick=tick, native_xy=[native['x'], native['y']],
                simulated_xy=[focal.pos.x, focal.pos.y], error_tiles=error/1000,
                simulated_target_uid=focal.target_uid))
    return dict(name=clip['name'], source=clip['source'], source_sha256=clip['source_sha256'],
        mean_position_error_tiles=sum(r['error_tiles'] for r in rows)/len(rows),
        max_position_error_tiles=max(r['error_tiles'] for r in rows), rows=rows,
        simulated_target_changes=target_changes, simulated_shots=shots,
        ignored_context_observations=dict(ignored),
        assumptions=dict(context='measured zero-order hold; no context AI or damage',
            focal_deploy_remaining_ms=0, initial_target_uid=initial_target_uid,
            sight_override=sight_override, collision_separation=False,
            knockback_buffs_and_spells=False, native_target_pointer_available=False),
        limitations=['Conditioned focal trajectory replay, not a full independent battle replay.',
            'Focal deployment is over at the anchor; initial target/attack phase are not observed.',
            'Attack counts, tower damage, first native shot and AOE are not validated by XY error.',
            'Ignored context types and unobserved collision/buff state can affect the path.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = replay(json.loads(args.clip.read_text()))
    result['clip_sha256'] = hashlib.sha256(args.clip.read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('name', 'mean_position_error_tiles',
        'max_position_error_tiles', 'simulated_target_changes', 'simulated_shots')}))
