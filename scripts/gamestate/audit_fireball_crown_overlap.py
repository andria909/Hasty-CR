"""Compare native Fireball aim/HP intervals with actual spell resolution.

Only stable aim fields supported by a straight flight and an adjacent
disappearance interval qualify. Nearby hostile entities conservatively exclude
damage attribution. This establishes hit/miss, not level-scaled damage parity.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from scripts.gamestate.audit_targeting_history import clock_frame, identity
from scripts.gamestate.audit_projectile_motion_clips import fitted_clip
from sim.arena import Point
from sim.engine import Battle
from sim.entities import make_tower
from sim.spells import apply_spell, cast_spell, load_spells


def resolve_at(target, aim, mode=None):
    battle = Battle()
    tower = battle.add(make_tower(1, 1, Point(target['x'], target['y']),
        10000, 0, 1000, 0, king=target['name'] == 'KingTower'))
    spec = load_spells(11)['fireball']
    if mode is not None:
        spec = replace(spec, princess_hitbox_mode=mode)
    damage = apply_spell(battle, spec, Point(aim['x'], aim['y']), -1)
    return dict(hit=tower.hitpoints < 10000, level11_damage=damage)


def replay_flight_timing(projectile, points, origin, aim, before, after):
    """Check a native birth candidate against motion before using it as phase.

    This exercises the actual delayed spell scheduler. It does not establish
    when Android accepted the placement input, or level-scaled native damage.
    """
    birth = projectile.get('deployment_tick_candidate')
    if type(birth) is not int or not 0 <= birth <= points[0][0]:
        return dict(accepted=False, reason='invalid_birth_candidate')
    phase_errors = [math.hypot(x-origin['x'], y-origin['y'])/600-(tick-birth)
                    for tick,x,y in points]
    if max(map(abs, phase_errors)) > 1:
        return dict(accepted=False, reason='birth_candidate_disagrees_with_flight',
                    phase_errors_ticks=phase_errors)
    battle = Battle()
    battle.now_ms = birth*50
    spec = load_spells(11)['fireball']
    cast_spell(battle, spec, Point(aim['x'],aim['y']), -1,
               origin=Point(origin['x'],origin['y']))
    # No entities: only the real impact scheduler runs, independent of overlap.
    for _ in range(200):
        battle.step(50)
        if battle.spell_cast_log[0]['resolved']:
            break
    else:
        raise AssertionError('Fireball did not resolve within ten seconds')
    tick = battle.now_ms//50
    return dict(accepted=True, birth_tick_candidate=birth,
                phase_error_ticks=[min(phase_errors),max(phase_errors)],
                engine_resolution_tick=tick,
                engine_in_observed_impact_window=before < tick <= after,
                native_impact_window=[before,after])


def inspect_clip(clip, frames):
    first_index, last_index = clip['first_frame'], clip['last_frame']
    first, last = frames[first_index], frames[last_index]
    if first is None or last is None:
        return None, 'invalid_clock'
    projectile = last['bodies'].get(clip['identity'][0])
    if projectile is None or identity(projectile) != tuple(clip['identity']):
        return None, 'projectile_identity'
    aim = projectile.get('target_position_candidate')
    origin = projectile.get('source_position_fields')
    if not aim or not origin:
        return None, 'missing_aim_fields'
    if not (0 <= aim['x'] <= 18000 and 0 <= aim['y'] <= 32000):
        return None, 'aim_outside_arena'
    kings = [b for b in first['bodies'].values()
             if b['name'] == 'KingTower' and b['side'] == projectile['side']]
    if not any((b['x'], b['y']) == (origin['x'], origin['y']) for b in kings):
        return None, 'origin_not_same_side_king'
    previous_tick = first['tick']
    observed_points = []
    for frame in frames[first_index:last_index+1]:
        if (frame is None or frame['world'] != last['world']
                or not 0 <= frame['tick']-previous_tick <= 5):
            return None, 'flight_discontinuity'
        previous_tick = frame['tick']
        body = frame['bodies'].get(projectile['entity_id'])
        if (body is None or identity(body) != identity(projectile)
                or body.get('target_position_candidate') != aim
                or body.get('source_position_fields') != origin):
            return None, 'unstable_aim_or_identity'
        observed_points.append((frame['tick'], body['x'], body['y']))
    fitted = fitted_clip(observed_points)
    if fitted is None:
        return None, 'nonlinear_or_conflicting_flight'
    if [list(p) for p in fitted['points']] != [list(p) for p in clip['points']]:
        return None, 'manifest_points_disagree'
    dx, dy = aim['x']-origin['x'], aim['y']-origin['y']
    length = math.hypot(dx, dy)
    if length == 0:
        return None, 'zero_flight'
    points = clip['points']
    residual = max(abs((x-origin['x'])*dy-(y-origin['y'])*dx)/length for _, x, y in points)
    if residual > 30:
        return None, 'aim_not_on_observed_ray'
    after_index = None
    for index in range(last_index+1, min(len(frames), last_index+8)):
        frame = frames[index]
        if (frame is None or frame['world'] != last['world']
                or not 0 <= frame['tick']-last['tick'] <= 5):
            break
        if projectile['entity_id'] not in frame['bodies']:
            after_index = index
            break
    if after_index is None:
        return None, 'no_contiguous_disappearance'
    after = frames[after_index]
    # The last visible coordinate may already be at the explosion. Anchor HP
    # to the latest sample still >=500mt away, chosen without looking at HP.
    # Retain the same <=5 tick total bound and screen the whole interval.
    before_index = None
    for index in range(last_index, first_index-1, -1):
        frame = frames[index]
        if after['tick']-frame['tick'] > 5:
            break
        body = frame['bodies'][projectile['entity_id']]
        remaining = math.hypot(body['x']-aim['x'], body['y']-aim['y'])
        if remaining >= 500:
            before_index = index
            break
    if before_index is None:
        return None, 'uncertain_impact_bracketing'
    before = frames[before_index]
    if remaining > 600*(after['tick']-before['tick']+1):
        return None, 'aim_too_far_at_disappearance'
    cases = []
    timing = replay_flight_timing(projectile,points,origin,aim,before['tick'],after['tick'])
    for target in before['bodies'].values():
        if (target['name'] not in ('KingTower', 'PrincessTower')
                or target['side'] == projectile['side'] or target['hp'] <= 0):
            continue
        ax, ay = abs(target['x']-aim['x']), abs(target['y']-aim['y'])
        if math.hypot(ax, ay) > 6000:
            continue
        following = after['bodies'].get(target['entity_id'])
        if (following is None or identity(following) != identity(target)
                or (following['x'], following['y']) != (target['x'], target['y'])
                or not 0 <= following['hp'] <= target['hp']):
            continue
        threats = set()
        interval = frames[before_index:after_index+1]
        for frame in interval:
            for body in frame['bodies'].values():
                if (body['entity_id'] != projectile['entity_id']
                        and body['side'] == projectile['side']
                        and math.hypot(body['x']-target['x'], body['y']-target['y']) < 10000):
                    threats.add((body['entity_id'], body['name']))
        cases.append(dict(source=clip['source'], first_frame=first_index,
            last_frame=last_index, after_frame=after_index, world=last['world'],
            preimpact_frame=before_index,
            projectile=projectile, aim=aim, origin=origin, flight_points=points,
            ray_residual_mt=residual, remaining_mt=remaining,
            interval_ticks=[before['tick'], after['tick']],
            flight_timing=timing,
            target_before=target, target_after=following, offset=[ax, ay],
            observed_hp_drop=target['hp']-following['hp'],
            observed_hit=following['hp'] < target['hp'], threats=sorted(threats),
            accepted=not threats,
            interval_context=[dict(tick=f['tick'], bodies=list(f['bodies'].values())) for f in interval],
            legacy=resolve_at(target, aim, 'collision_circle_v1'),
            rectangle=resolve_at(target, aim, 'rectangle_v1'),
            current=resolve_at(target, aim)))
    return cases, None


def audit(manifest):
    manifest = Path(manifest)
    screen = json.loads(manifest.read_text())
    grouped = {}
    for clip in screen['clips']:
        if clip['identity'][2] == 'FireballSpell':
            grouped.setdefault(clip['source'], []).append(clip)
    counts, cases, sources = Counter(), [], []
    for source in screen['sources']:
        path = source['path']
        if path not in grouped:
            continue
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source['sha256']:
            raise ValueError(f'Source changed: {path}')
        sources.append(source)
        frames = [clock_frame(json.loads(line)) for line in raw.splitlines()]
        for clip in grouped[path]:
            counts['fitted_fireball_clips'] += 1
            result, rejected = inspect_clip(clip, frames)
            if rejected:
                counts[rejected] += 1
            else:
                cases.extend(result)
    accepted = [c for c in cases if c['accepted']]
    summary = dict(candidates=len(cases), accepted=len(accepted),
        distinct_geometries=len({(c['target_before']['name'], *c['offset']) for c in accepted}),
        native_hits=sum(c['observed_hit'] for c in accepted),
        disagreements={mode: sum(c[mode]['hit'] != c['observed_hit'] for c in accepted)
                       for mode in ('legacy', 'rectangle', 'current')})
    unique_flights = {(c['source'],c['first_frame'],tuple(c['projectile'].get(k)
                      for k in ('entity_id','address'))):c['flight_timing'] for c in accepted}
    timed = [t for t in unique_flights.values() if t['accepted']]
    summary['flight_timing'] = dict(unique_flights=len(unique_flights),
        birth_supported=len(timed),
        engine_inside_native_window=sum(t['engine_in_observed_impact_window'] for t in timed))
    code = ['scripts/gamestate/audit_fireball_crown_overlap.py',
            'scripts/gamestate/audit_projectile_motion_clips.py',
            'scripts/gamestate/audit_targeting_history.py',
            'scripts/gamestate/audit_recorded_fireball.py',
            'sim/spells.py', 'sim/public_gamedata.py', 'sim/attack_env.py',
            'sim/engine.py', 'sim/entities.py', 'sim/projectile_calibration.py']
    return dict(schema='recorded-fireball-crown-overlap-v1', sources=sources,
        source_fingerprints={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code},
        velocity_manifest=str(manifest), velocity_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        counts=dict(counts), summary=summary, cases=cases,
        limitations=['No damage-source pointer: isolated means no other observed hostile within ten tiles.',
            'Unobserved instantaneous spells and non-atomic reads cannot be categorically excluded.',
            'Stable aim fields, ray and disappearance constrain the endpoint; these are not screen clicks.',
            'Level11 simulated damage is reported only to exercise the engine; native level scaling is unvalidated.',
            'Agreement on these coordinates does not uniquely identify the full hitbox; intermediate circle radii can also fit.',
            'Same-match repeated shots are correlated; troop, King and other-spell geometry are not recalibrated.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--velocity-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.velocity_manifest)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(counts=result['counts'], summary=result['summary'])))
