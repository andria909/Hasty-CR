"""Exploratory target labels from recorded homing shots and HP drops.

These are INFERRED impacts, not verified engine target IDs. Only unambiguous
nearby shooters and isolated damaged victims are retained. No game input/CV.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from sim.arena import Point
from sim.engine import Battle
from sim.entities import make_tower, make_unit
from sim.gamedata import to_snake_case
from scripts.gamestate.audit_sim_alignment import unit_profiles

ROOT = Path(__file__).resolve().parents[2]


def gap(a, b):
    return math.hypot(a['x'] - b['x'], a['y'] - b['y'])


def shooter_types():
    projectile_rows = json.loads((ROOT / 'data/royaleapi/cards_stats_projectile.json').read_text())
    homing = {r['name'] for r in projectile_rows if r.get('homing')}
    allowed = {'ArcherArrow', 'MusketeerProjectile', 'TowerCannonball', 'TowerPrincessProjectile'} & homing
    result = {name: set() for name in allowed}
    for filename in ('cards_stats_characters.json', 'cards_stats_building.json'):
        for row in json.loads((ROOT / 'data/royaleapi' / filename).read_text()):
            if row.get('projectile') in result:
                result[row['projectile']].add(row['name'])
    return result


def unique_shooter(projectile, bodies, names):
    choices = [b for b in bodies.values() if b['side'] == projectile['side']
               and b['name'] in names and b['hp'] > 0 and gap(b, projectile) <= 1750]
    return choices[0] if len(choices) == 1 else None


def isolated_impact(projectile, before, after):
    """Require exactly one nearby enemy before disappearance AND a HP drop."""
    near = [b for b in before.values() if b['side'] != projectile['side']
            and b['hp'] > 0 and gap(b, projectile) <= 1000]
    if len(near) != 1:
        return None
    victim = near[0]
    current = after.get(victim['entity_id'])
    if current is None or current.get('address') != victim.get('address'):
        return None
    if not 0 <= current['hp'] < victim['hp']:
        return None
    return current


def predict_fresh_target(bodies, source_id, profiles, *, prior_target_id=None,
                         force_rescan=True, include_source_radius=False):
    # This diagnostic was defined against the pre-correction engine contract.
    # Keep its historical fresh/lock hypotheses explicit instead of silently
    # reinterpreting already-pinned reports when production defaults change.
    battle = Battle(normal_range_mode='legacy_target_edge_v1', target_lock_mode='legacy_periodic_v1')
    for raw in bodies:
        side = 1 if raw['side'] == 1 else -1
        pos = Point(raw['x'], raw['y'])
        if raw['name'] in ('KingTower', 'PrincessTower'):
            # Sufficient for candidate identity/geometry. Crown towers as
            # sources are excluded below: their asleep/equipped state is absent.
            entity = make_tower(raw['entity_id'], side, pos, raw['max_hp'], 0, 1000, 7000,
                                king=raw['name'] == 'KingTower')
        else:
            key = to_snake_case(raw['name'])
            if key not in profiles:
                return None, 'unmapped_body_in_launch_snapshot'
            entity = make_unit(raw['entity_id'], profiles[key], side, pos)
        entity.hitpoints = raw['hp']
        entity.deploy_remaining_ms = 0
        # Battle.add allocates fresh IDs. Preserve native IDs here so impact
        # evidence and predicted targets refer to the same bodies.
        battle.entities[entity.uid] = entity
    source = battle.get(source_id)
    if source is None:
        return None, 'missing_source_in_launch_snapshot'
    if source.is_tower:
        return None, 'tower_source_state_not_reconstructed'
    if include_source_radius:
        # An explicit audit hypothesis, NOT a production engine change.
        source.range_mt += source.collision_radius_mt
        source.sight_range_mt += source.collision_radius_mt
    source.target_uid = prior_target_id
    # Compare both explicit branches. We do not know the engine's actual scan
    # phase; these are hypotheses rather than a reconstructed live tick.
    battle.now_ms = ((500 - source.uid * 50) % 500) if force_rescan else 1
    battle._acquire_target(source)
    return source.target_uid, ('fresh_acquisition_only_not_replayed_target_lock'
                               if prior_target_id is None else
                               'inferred_prior_lock_forced_rescan' if force_rescan else
                               'inferred_prior_lock_between_rescans')


def audit(path: Path):
    types, profiles = shooter_types(), unit_profiles()
    previous_bodies, previous_shots, active = {}, {}, {}
    previous_time, identity = None, None
    last_launch = {}
    evidence, launched = [], Counter()
    frames = 0
    segment = 0
    for index, line in enumerate(path.open()):
        row = json.loads(line)
        # Later live logs wrap the same read-only native frame in host timing
        # metadata. Keep the raw-frame format supported for older recordings.
        if isinstance(row.get('frame'), dict):
            row = row['frame']
        entities = row.get('entities')
        if not isinstance(entities, list):
            previous_bodies, previous_shots, active = {}, {}, {}
            previous_time = None
            last_launch = {}
            segment += 1
            continue
        timestamp = row['device_start_ns']
        bodies = {e['entity_id']: e for e in entities if e['entity_id'] // 1_000_000 == 5}
        shots = {e['entity_id']: e for e in entities if e['entity_id'] // 1_000_000 == 4}
        key = tuple(sorted((e['entity_id'], e.get('address')) for e in bodies.values() if e['name'] == 'KingTower'))
        if key != identity or (previous_time is not None and not 0 < timestamp - previous_time <= 150_000_000):
            segment += 1
            previous_bodies, previous_shots, active = {}, {}, {}
            previous_time = None
            identity = key
            last_launch = {}
        for uid, shot in shots.items():
            if uid not in previous_shots and shot['name'] in types:
                launched[shot['name']] += 1
                shooter = unique_shooter(shot, bodies, types[shot['name']])
                if shooter is not None:
                    active[uid] = {'source': shooter, 'birth': shot,
                                   'launch_frame': index, 'bodies': list(bodies.values()),
                                   'previous_projectile_id': last_launch.get(shooter['entity_id'])}
                    last_launch[shooter['entity_id']] = uid
        for uid, last in previous_shots.items():
            if uid in shots or uid not in active:
                continue
            event = active.pop(uid)
            if previous_time is None or gap(event['birth'], last) < 300:
                continue
            victim = isolated_impact(last, previous_bodies, bodies)
            if victim is None or victim['entity_id'] not in {b['entity_id'] for b in event['bodies']}:
                continue
            prediction, scope = predict_fresh_target(event['bodies'], event['source']['entity_id'], profiles)
            evidence.append({
                'trace_segment': segment,
                'projectile_id': uid, 'projectile_name': last['name'],
                'source_id': event['source']['entity_id'], 'source_name': event['source']['name'],
                'inferred_target_id': victim['entity_id'], 'inferred_target_name': victim['name'],
                'launch_frame': event['launch_frame'], 'impact_frame': index,
                'previous_projectile_id': event['previous_projectile_id'],
                'last_projectile_position': [last['x'], last['y']],
                'victim_position': [victim['x'], victim['y']],
                'victim_hp_before': previous_bodies[victim['entity_id']]['hp'],
                'victim_hp_after': victim['hp'],
                'sim_fresh_target_id': prediction, 'sim_comparison_scope': scope,
                'fresh_prediction_agrees': prediction == victim['entity_id'] if prediction is not None else None,
                'label_status': 'inferred_homing_impact_and_hp_drop_not_verified_target_uid',
                'launch_bodies': event['bodies'],
            })
        previous_bodies, previous_shots, previous_time = bodies, shots, timestamp
        frames += 1
    comparable = [e for e in evidence if e['fresh_prediction_agrees'] is not None]
    # Native entity IDs restart across battles. Never infer an old target lock
    # from a projectile with the same numeric ID in a different trace segment.
    by_shot = {(e['trace_segment'], e['projectile_id']): e for e in evidence}
    for event in comparable:
        hypothesis, _ = predict_fresh_target(
            event['launch_bodies'], event['source_id'], profiles,
            include_source_radius=True)
        event['sim_source_radius_hypothesis_target_id'] = hypothesis
        event['source_radius_hypothesis_agrees'] = hypothesis == event['inferred_target_id']
        source = next(b for b in event['launch_bodies'] if b['entity_id'] == event['source_id'])
        victim = next(b for b in event['launch_bodies'] if b['entity_id'] == event['inferred_target_id'])
        source_profile = profiles[to_snake_case(source['name'])]
        target_profile = profiles.get(to_snake_case(victim['name']))
        event['launch_distance_millitiles'] = gap(source, victim)
        event['source_radius_millitiles'] = source_profile.collision_radius_mt
        event['target_radius_millitiles'] = (target_profile.collision_radius_mt
                                            if target_profile is not None else None)
        event['source_range_millitiles'] = source_profile.range_mt
        previous = by_shot.get((event['trace_segment'], event['previous_projectile_id']))
        if previous is None or previous['impact_frame'] > event['launch_frame']:
            continue
        previous_target = previous['inferred_target_id']
        prediction, scope = predict_fresh_target(
            event['launch_bodies'], event['source_id'], profiles,
            prior_target_id=previous_target, force_rescan=False)
        event['prior_inferred_target_id'] = previous_target
        event['sim_retained_target_id'] = prediction
        event['sim_retained_scope'] = scope
        event['retained_prediction_agrees'] = prediction == event['inferred_target_id']
    return {'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'frames': frames, 'eligible_projectile_appearances': dict(launched),
            'inferred_isolated_impacts': len(evidence),
            'fresh_acquisition_comparable': len(comparable),
            'fresh_acquisition_agreements': sum(e['fresh_prediction_agrees'] for e in comparable),
            'fresh_disagreements_explained_by_retaining_previous_inferred_target': sum(
                not e['fresh_prediction_agrees'] and e.get('retained_prediction_agrees', False)
                for e in comparable),
            'source_radius_hypothesis_agreements': sum(
                e['source_radius_hypothesis_agrees'] for e in comparable),
            'limitations': [
                'Exploratory inferred labels, not a targeting-accuracy benchmark.',
                'Nearby victim/HP-drop attribution can still be coincidental; manual shot review is required.',
                'Fresh acquisition ignores prior target locks, scan phase, buffs and deployment state.',
                'Source-radius hypothesis is exploratory and does not modify production targeting or attack reach.',
                'Tower-source targeting is not compared because tower activation/equipment is not reconstructed.',
                'One existing replay; melee, splash, retarget triggers and unseen mechanics are not certified.',
            ], 'evidence': evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.trace)
    with args.output.open('x') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'evidence'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
