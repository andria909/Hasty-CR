"""Recheck recorded static rays, then exercise both actual attack contracts.

Targets are forced and timers cleared. This proves reach at measured points,
not live target-pointer accuracy or complete battle parity.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.gamestate.audit_targeting_history import audit, make_source
from scripts.gamestate.audit_sim_alignment import unit_profiles
from sim.arena import Point
from sim.engine import Battle
from sim.entities import make_tower, make_unit
from sim.gamedata import to_snake_case


def probe(row, profiles, mode):
    source = dict(name=row['source_name'], x=row['source_position'][0], y=row['source_position'][1])
    target = dict(name=row['inferred_target_name'], x=row['target_position'][0], y=row['target_position'][1])
    battle = Battle(normal_range_mode=mode, target_lock_mode='legacy_periodic_v1')
    shooter = make_source(source, profiles)
    if target['name'] in ('PrincessTower', 'KingTower'):
        victim = make_tower(2, -1, Point(target['x'], target['y']), 100000, 0, 1000, 0,
                            king=target['name'] == 'KingTower')
    else:
        victim = make_unit(2, profiles[to_snake_case(target['name'])], -1, Point(target['x'], target['y']))
    battle.entities = {1: shooter, 2: victim}
    shooter.target_uid = 2
    shooter.windup_remaining_ms = shooter.attack_cooldown_ms = shooter.deploy_remaining_ms = victim.deploy_remaining_ms = 0
    battle._attack(shooter, 50)
    return len(battle.in_flight)


def verify(manifests):
    profiles = unit_profiles()
    inputs, rows = [], []
    skipped = 0
    for path in map(Path, manifests):
        result = audit(path, include_crown_sources=True)
        inputs.append(dict(manifest=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           sources=result['sources'], unit_profiles_sha256=result['unit_profiles_sha256']))
        for row in result['rows']:
            if not row['static_trajectory_supported']:
                continue
            if 'excess_over_current_reach_mt' not in row:
                skipped += 1
                continue
            evidence = {k: row[k] for k in ('run', 'launch_frame', 'impact_frame', 'source_id', 'projectile_id',
                'inferred_target_id', 'source_name', 'inferred_target_name', 'source_position', 'target_position',
                'distance_mt', 'trajectory', 'stationary_prelaunch_ticks')}
            for mode in ('legacy_target_edge_v1', 'edge_to_edge_v1'):
                evidence[mode] = probe(row, profiles, mode)
            rows.append(evidence)
    failures = [dict(run=r['run'], launch_frame=r['launch_frame'], projectile_id=r['projectile_id'])
                for r in rows if r['edge_to_edge_v1'] != 1]
    return dict(schema='corrected-normal-range-replay-v2', inputs=inputs, rows=rows,
                static_shots=len(rows), unmapped_static_targets_excluded=skipped,
                legacy_projectiles=sum(r['legacy_target_edge_v1'] for r in rows),
                corrected_projectiles=sum(r['edge_to_edge_v1'] for r in rows),
                failed_cases=failures, all_eligible_recorded_shots_reproduced=bool(rows) and not failures,
                target_forced=True, timers_cleared=True, quality_accepted=False,
                new_default_range_mode=Battle().normal_range_mode, new_default_lock_mode=Battle().target_lock_mode,
                real_game_target_pointers_verified=False, full_mechanics_parity_verified=False,
                fingerprints={s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in (
                    'sim/engine.py', 'sim/entities.py', 'sim/gamedata.py', 'sim/public_gamedata.py',
                    'sim/continuous_push.py', 'scripts/gamestate/audit_targeting_history.py',
                    'scripts/gamestate/verify_normal_range.py')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.manifest)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows', 'inputs', 'fingerprints')}))
    if not result['all_eligible_recorded_shots_reproduced']:
        raise SystemExit('Corrected geometry does not reproduce all eligible recorded shots')
