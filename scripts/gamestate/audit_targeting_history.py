"""Check recorded shot trajectories and joint range/lock hypotheses, offline.

Never promotes inferred victims to native target pointers or changes physics.
Only stable-clock, same-world, same-address samples enter trajectory checks.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from scripts.gamestate.audit_projectile_targets import gap, predict_fresh_target
from scripts.gamestate.audit_sim_alignment import unit_profiles
from sim.arena import Point
from sim.engine import Battle
from sim.entities import KING_COLLISION_MT, PRINCESS_COLLISION_MT, make_tower, make_unit
from sim.gamedata import to_snake_case
from sim.towers import princess_tower, king_tower


def identity(body):
    return tuple(body.get(k) for k in ('entity_id', 'address', 'name', 'side'))


def clock_frame(row):
    frame = row.get('frame', {})
    own = row.get('own_state', {})
    tick = frame.get('battle_tick')
    if (row.get('status') != 'ok' or own.get('status') != 'ok'
            or type(tick) is not int or own.get('tick') != tick or not own.get('world')):
        return None
    bodies = frame.get('entities')
    if not isinstance(bodies, list):
        return None
    kings = tuple(sorted(identity(b) for b in bodies if b['name'] == 'KingTower'))
    if not kings:
        return None
    return dict(tick=tick, world=(own['world'], kings),
                bodies={b['entity_id']: b for b in bodies})


def trajectory_geometry(points, source, target):
    """A static-endpoint ray check independent of the victim's HP change."""
    if len(points) < 3:
        return dict(accepted=False, reason='fewer_than_three_distinct_positions')
    first, last = points[0], points[-1]
    dx, dy = last['x'] - first['x'], last['y'] - first['y']
    length = math.hypot(dx, dy)
    if length < 1000:
        return dict(accepted=False, reason='short_trajectory')
    def along(point):
        return ((point['x'] - first['x']) * dx + (point['y'] - first['y']) * dy) / length
    def perpendicular(point):
        return abs((point['x'] - first['x']) * dy - (point['y'] - first['y']) * dx) / length
    residual = max(perpendicular(p) for p in [source, target, *points])
    forward = [along(p) for p in points]
    accepted = (residual <= 80 and gap(first, source) <= 1750
                and gap(last, target) <= 1100 and along(source) <= 80
                and along(target) >= length - 80
                and all(b >= a for a, b in zip(forward, forward[1:])))
    return dict(accepted=accepted, reason='static_ray_consistent' if accepted else 'ray_geometry_failed',
                max_perpendicular_error_mt=residual, observed_travel_mt=length,
                first_distance_to_source_mt=gap(first, source), last_distance_to_target_mt=gap(last, target))


def make_source(source, profiles):
    if source['name'] in ('PrincessTower', 'KingTower'):
        king = source['name'] == 'KingTower'
        spec = (king_tower if king else princess_tower)(11)
        return make_tower(1, 1, Point(source['x'], source['y']), spec.hitpoints,
                          spec.damage, spec.hit_speed_ms, spec.range_mt, king=king)
    return make_unit(1, profiles[to_snake_case(source['name'])], 1, Point(source['x'], source['y']))


def isolated_attack_probe(source, target, profiles, include_source_radius):
    """Actual engine attack at measured positions; no movement or target choice.

    Only a freshly constructed source instance is modified in the hypothesis.
    Target is forced and timers cleared: this tests reach, not live timing/AI.
    """
    battle = Battle(normal_range_mode='legacy_target_edge_v1', target_lock_mode='legacy_periodic_v1')
    shooter = make_source(source, profiles)
    if target['name'] in ('KingTower', 'PrincessTower'):
        victim = make_tower(2, -1, Point(target['x'], target['y']), 100000, 0, 1000, 0,
                            king=target['name'] == 'KingTower')
    else:
        spec = profiles.get(to_snake_case(target['name']))
        if spec is None:
            return None
        victim = make_unit(2, spec, -1, Point(target['x'], target['y']))
    battle.entities = {1: shooter, 2: victim}
    shooter.deploy_remaining_ms = victim.deploy_remaining_ms = 0
    shooter.target_uid = 2
    shooter.attack_cooldown_ms = shooter.windup_remaining_ms = 0
    if include_source_radius:
        shooter.range_mt += shooter.collision_radius_mt
    spent_tick = battle._attack(shooter, 50)
    return dict(attack_tick=spent_tick, homing_projectiles_created=len(battle.in_flight),
                target_forced=True, timers_cleared=True)


def inspect_event(event, frames, profiles):
    start, end = event['launch_frame'], event['impact_frame']
    first = frames[start]
    result = {k: event[k] for k in ('launch_frame', 'impact_frame', 'source_id', 'projectile_id',
                                   'inferred_target_id', 'source_name', 'inferred_target_name',
                                   'fresh_prediction_agrees')}
    result['static_trajectory_supported'] = False
    if first is None:
        return dict(result, rejected='invalid_launch_clock')
    ids = [event[k] for k in ('source_id', 'inferred_target_id', 'projectile_id')]
    launch = [first['bodies'].get(uid) for uid in ids]
    if any(b is None for b in launch):
        return dict(result, rejected='missing_launch_entity')
    source, target, shot = launch
    result.update(source_position=[source['x'], source['y']], target_position=[target['x'], target['y']])
    # Verify the saved launch body really belongs to this trace, not an ID reuse.
    saved = {b['entity_id']: b for b in event['launch_bodies']}
    if any(identity(b) != identity(saved.get(b['entity_id'], {})) for b in (source, target)):
        return dict(result, rejected='evidence_identity_mismatch')
    samples = []
    last_tick = first['tick']
    for frame in frames[start:end]:
        if (frame is None or frame['world'] != first['world']
                or not 0 <= frame['tick'] - last_tick <= 5):
            return dict(result, rejected='clock_or_world_gap')
        last_tick = frame['tick']
        bodies = [frame['bodies'].get(uid) for uid in ids]
        if any(b is None or identity(b) != identity(original) for b, original in zip(bodies, launch)):
            return dict(result, rejected='entity_missing_or_reused')
        samples.append((frame['tick'], *bodies))
    source_motion = max(gap(s, source) for _, s, _, _ in samples)
    target_motion = max(gap(t, target) for _, _, t, _ in samples)
    result.update(source_motion_during_flight_mt=source_motion,
                  target_motion_during_flight_mt=target_motion)
    # Look back 600 ms in game time, preserving every continuity check.
    prior = []
    next_tick = first['tick']
    for frame in reversed(frames[:start]):
        if frame is None or frame['world'] != first['world'] or not 0 <= next_tick-frame['tick'] <= 5:
            break
        body = frame['bodies'].get(ids[0])
        if body is None or identity(body) != identity(source) or gap(body, source) > 10:
            break
        next_tick = frame['tick']
        prior.append(next_tick)
        if first['tick'] - next_tick >= 12:
            break
    result['stationary_prelaunch_ticks'] = first['tick'] - prior[-1] if prior else 0
    points = []
    for _, _, _, p in samples:
        if not points or (p['x'], p['y']) != (points[-1]['x'], points[-1]['y']):
            points.append(p)
    geometry = trajectory_geometry(points, source, target)
    result['trajectory'] = geometry
    result['static_trajectory_supported'] = bool(
        source_motion <= 10 and target_motion <= 10 and geometry['accepted']
        and result['stationary_prelaunch_ticks'] >= 12 and len(set(prior)) >= 3)
    source_profile = make_source(source, profiles)
    target_profile = profiles.get(to_snake_case(target['name']))
    radius = (KING_COLLISION_MT if target['name'] == 'KingTower' else
              PRINCESS_COLLISION_MT if target['name'] == 'PrincessTower' else
              target_profile.collision_radius_mt if target_profile else None)
    if radius is not None:
        distance = gap(source, target)
        reach = source_profile.range_mt + radius
        result.update(distance_mt=distance, current_attack_reach_mt=reach,
                      source_radius_extension_mt=source_profile.collision_radius_mt,
                      excess_over_current_reach_mt=distance-reach,
                      within_source_radius_extended_reach=distance <= reach+source_profile.collision_radius_mt)
    if result['static_trajectory_supported']:
        result['isolated_attack_probes'] = {
            str(int(extension)): isolated_attack_probe(source, target, profiles, extension)
            for extension in (False, True)}
    # A factorial diagnostic: separate radius and retention. An assumed lock
    # is explicitly NOT a measurement of a previous native target pointer.
    result['assumed_lock_hypotheses'] = {}
    for radius_extension in (False, True):
        for rescan in (False, True):
            predicted, _ = predict_fresh_target(event['launch_bodies'], ids[0], profiles,
                prior_target_id=ids[1], force_rescan=rescan, include_source_radius=radius_extension)
            result['assumed_lock_hypotheses'][f'source_radius_{int(radius_extension)}_rescan_{int(rescan)}'] = (
                predicted == ids[1] if source['name'] not in ('PrincessTower', 'KingTower') else None)
    return result


def previous_shot_evidence(event, prior_events, frames):
    """Link only the latest observed shot of this shooter, across valid frames.

    This records repeated inferred victims, not an uninterrupted native lock:
    intermediate unobserved shots, stuns or reacquisition remain possible.
    """
    previous = [p for p in prior_events if p['source_id'] == event['source_id']
                and p['launch_frame'] < event['launch_frame']]
    if not previous:
        return None
    previous = max(previous, key=lambda p: p['launch_frame'])
    if previous['impact_frame'] > event['launch_frame'] or previous['inferred_target_id'] != event['inferred_target_id']:
        return None
    first, last = frames[previous['launch_frame']], frames[event['launch_frame']]
    if first is None or last is None or not 0 < last['tick'] - first['tick'] <= 120:
        return None
    ids = (event['source_id'], event['inferred_target_id'])
    original = [first['bodies'].get(uid) for uid in ids]
    if any(b is None for b in original):
        return None
    tick = first['tick']
    maximum_motion = [0., 0.]
    for frame in frames[previous['launch_frame']:event['launch_frame']+1]:
        if frame is None or frame['world'] != first['world'] or not 0 <= frame['tick']-tick <= 5:
            return None
        tick = frame['tick']
        for i, (uid, old) in enumerate(zip(ids, original)):
            current = frame['bodies'].get(uid)
            if current is None or identity(current) != identity(old) or current['hp'] <= 0:
                return None
            maximum_motion[i] = max(maximum_motion[i], gap(current, old))
    return dict(launch_frame=previous['launch_frame'], impact_frame=previous['impact_frame'],
                projectile_id=previous['projectile_id'], tick_difference=last['tick']-first['tick'],
                source_max_motion_mt=maximum_motion[0], victim_max_motion_mt=maximum_motion[1],
                native_lock_verified=False)


def audit(manifest_path, include_crown_sources=False):
    manifest = json.loads(manifest_path.read_text())
    profiles = unit_profiles()
    sources, rows = [], []
    for source in manifest['sources']:
        path, evidence_path = Path(source['source']), Path(source['evidence_path'])
        raw, evidence_raw = path.read_bytes(), evidence_path.read_bytes()
        if (hashlib.sha256(raw).hexdigest() != source['source_sha256']
                or hashlib.sha256(evidence_raw).hexdigest() != source['evidence_sha256']):
            raise ValueError('Source/evidence hash differs from audited manifest')
        evidence = json.loads(evidence_raw)
        if evidence['source_sha256'] != source['source_sha256']:
            raise ValueError('Evidence points to different recording')
        frames = [clock_frame(json.loads(line)) for line in raw.splitlines()]
        comparable = [e for e in evidence['evidence'] if e.get('fresh_prediction_agrees') is not None
                      or include_crown_sources and e['source_name'] in ('PrincessTower', 'KingTower')]
        for event in comparable:
            rows.append(dict(run=source['name'], **inspect_event(event, frames, profiles),
                previous_observed_same_victim=previous_shot_evidence(event, comparable, frames)))
        sources.append({k: source[k] for k in ('name', 'source', 'source_sha256', 'evidence_path', 'evidence_sha256')})
    supported = [r for r in rows if r['static_trajectory_supported']]
    outside = [r for r in supported if r.get('excess_over_current_reach_mt', 0) > 80]
    return dict(schema='native-targeting-history-v1', sources=sources, rows=rows,
                includes_crown_sources=include_crown_sources,
                comparable_events=len(rows), stationary_trajectory_supported=len(supported),
                stationary_outside_current_reach=len(outside),
                outside_cases=[dict(run=r['run'], launch_frame=r['launch_frame']) for r in outside],
                rejections=dict(Counter(r.get('rejected', 'analysed') for r in rows)),
                labels_verified=False, physics_changed=False, live_games_started=False,
                unit_profiles_sha256=hashlib.sha256(json.dumps(
                    {k: asdict(v) for k, v in profiles.items()}, sort_keys=True).encode()).hexdigest(),
                source_fingerprints={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,
                    ('scripts/gamestate/audit_targeting_history.py', 'scripts/gamestate/audit_projectile_targets.py',
                     'scripts/gamestate/audit_sim_alignment.py', 'sim/engine.py', 'sim/entities.py', 'sim/gamedata.py'))},
                limitations=[
                    'Victim inference is strengthened by trajectory, not replaced with an actual target pointer.',
                    'Coordinates precede clock read; stable endpoints reduce but do not eliminate sampling uncertainty.',
                    'Source-radius extension and a larger target hitbox can be confounded for stationary towers.',
                    'Assumed prior locks are counterfactuals, not measured retention histories.',
                    'No production attack, movement, collision, or target selection changes from this audit.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-crown-sources', action='store_true')
    args = parser.parse_args()
    report = audit(args.manifest, args.include_crown_sources)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'sources', 'source_fingerprints', 'limitations')}))
