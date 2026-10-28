import copy

from scripts.gamestate.audit_sim_alignment import unit_profiles
from scripts.gamestate.audit_targeting_history import (
    clock_frame, inspect_event, previous_shot_evidence, trajectory_geometry)
from tests.test_projectile_target_audit import body


def fixture():
    source = body(5000007, 'Musketeer', 1, 14500, 13950)
    target = body(5000002, 'PrincessTower', 0, 14500, 6500)
    king = body(5000000, 'KingTower', 1, 9000, 29000)
    nearer = body(5000009, 'Knight', 0, 13000, 13950)
    frames = []
    for tick in range(24):
        units = copy.deepcopy([source, target, king, nearer])
        if 12 <= tick < 20:
            units += [body(4000001, 'MusketeerProjectile', 1, 14500, 13500-(tick-12)*1000)]
        frames.append(clock_frame(dict(status='ok', own_state=dict(status='ok', world='one', tick=tick),
            frame=dict(battle_tick=tick, entities=units))))
    event = dict(launch_frame=12, impact_frame=20, source_id=source['entity_id'],
        inferred_target_id=target['entity_id'], projectile_id=4000001,
        source_name=source['name'], inferred_target_name=target['name'],
        fresh_prediction_agrees=False, launch_bodies=[source, target, king, nearer])
    return event, frames


def test_static_long_shot_has_independent_ray_evidence_without_claiming_target_pointer():
    event, frames = fixture()
    result = inspect_event(event, frames, unit_profiles())
    assert result['static_trajectory_supported']
    assert result['stationary_prelaunch_ticks'] == 12
    assert result['excess_over_current_reach_mt'] == 450
    assert result['within_source_radius_extended_reach']
    assert result['isolated_attack_probes']['0']['homing_projectiles_created'] == 0
    assert result['isolated_attack_probes']['1']['homing_projectiles_created'] == 1
    assert not result['assumed_lock_hypotheses']['source_radius_0_rescan_0']
    assert result['assumed_lock_hypotheses']['source_radius_1_rescan_0']


def test_a_reused_native_id_or_changed_world_breaks_trajectory():
    for mutation, reason in [('identity', 'entity_missing_or_reused'), ('world', 'clock_or_world_gap')]:
        event, frames = fixture()
        frames = copy.deepcopy(frames)
        if mutation == 'identity':
            frames[15]['bodies'][4000001]['address'] = 'different'
        else:
            frames[15]['world'] = ('different', ())
        result = inspect_event(event, frames, unit_profiles())
        assert not result['static_trajectory_supported'] and result['rejected'] == reason


def test_launch_after_movement_cannot_certify_static_range():
    event, frames = fixture()
    frames = copy.deepcopy(frames)
    frames[10]['bodies'][event['source_id']]['y'] += 60
    result = inspect_event(event, frames, unit_profiles())
    assert not result['static_trajectory_supported']


def test_bent_or_backwards_ray_does_not_support_target_label():
    source, target = dict(x=0, y=0), dict(x=0, y=5000)
    assert trajectory_geometry([dict(x=0, y=y) for y in (450, 2500, 4500)], source, target)['accepted']
    assert not trajectory_geometry([dict(x=0, y=y) for y in (4500, 2500, 450)], source, target)['accepted']
    assert not trajectory_geometry([dict(x=0, y=450), dict(x=500, y=2500), dict(x=0, y=4500)], source, target)['accepted']


def test_invalid_clock_and_long_gap_are_not_bridged():
    assert clock_frame(dict(status='ok', own_state=dict(status='ok', tick=1, world='one'),
                            frame=dict(battle_tick=2, entities=[]))) is None
    event, frames = fixture()
    frames[15] = None
    assert inspect_event(event, frames, unit_profiles())['rejected'] == 'clock_or_world_gap'


def test_previous_shot_needs_continuity_and_cannot_skip_intermediate_other_victim():
    event, frames = fixture()
    earlier = dict(event, launch_frame=0, impact_frame=5, projectile_id=4000000)
    result = previous_shot_evidence(event, [earlier], frames)
    assert result['tick_difference'] == 12 and not result['native_lock_verified']
    different = dict(earlier, launch_frame=6, impact_frame=10, inferred_target_id=5000999)
    assert previous_shot_evidence(event, [earlier, different], frames) is None
    frames[8] = None
    assert previous_shot_evidence(event, [earlier], frames) is None
