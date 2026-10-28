"""Range, movement, retention and historical replay share explicit contracts."""
from dataclasses import replace
import pickle

import pytest

from sim.arena import Point
from sim.engine import Battle
from sim.entities import make_unit, make_tower
from scripts.gamestate.audit_sim_alignment import unit_profiles


def pair(name='musketeer', distance=7450, mirror=False, range_mode='edge_to_edge_v1'):
    specs = unit_profiles()
    b = Battle(normal_range_mode=range_mode, target_lock_mode='in_range_v1')
    source = make_unit(0, specs[name], 1, Point(3500, 14000))
    target = make_tower(0, -1, Point(3500, 14000-distance), 100000, 0, 1000, 0)
    if mirror:
        source.pos = Point(18000-source.pos.x, source.pos.y)
        target.pos = Point(18000-target.pos.x, target.pos.y)
    source = b.add(source); target = b.add(target)
    source.deploy_remaining_ms = 0
    source.windup_remaining_ms = 0
    source.target_uid = target.uid
    return b, source, target


@pytest.mark.parametrize('mirror', [False, True])
@pytest.mark.parametrize('name,reach', [('musketeer', 7500), ('archer', 6500), ('cannon', 7100), ('hog_rider', 2400)])
def test_exact_boundary_and_one_millitile_outside(name, reach, mirror):
    for distance, expected in [(reach, True), (reach+1, False)]:
        battle, source, target = pair(name, distance, mirror)
        assert battle._normal_attack_reach(source, target) == reach
        assert battle._attack(source, 50) is expected


@pytest.mark.parametrize('mirror', [False, True])
def test_real_long_shot_stops_movement_and_delivers_damage(mirror):
    battle, source, target = pair(mirror=mirror)
    start, hp = source.pos, target.hitpoints
    for _ in range(30):
        battle.step(50)
        assert source.pos == start
    assert target.hitpoints < hp


def test_approach_stops_at_new_reach_instead_of_old_boundary():
    battle, source, target = pair(distance=8000)
    source.target_uid = None
    for _ in range(50):
        battle.step(50)
    gap = source.pos.y-target.pos.y
    assert 7440 <= gap <= 7500
    assert target.hitpoints < 100000


def test_in_range_lock_survives_scan_but_dead_hidden_or_out_of_range_does_not():
    for trigger in ['none', 'dead', 'hidden', 'out_of_range', 'every_tick', 'legacy']:
        battle, source, target = pair()
        near = battle.add(make_unit(0, unit_profiles()['knight'], -1, Point(3500, 12000)))
        near.deploy_remaining_ms = 0
        battle.now_ms = (500-source.uid*50) % 500
        if trigger == 'dead': target.hitpoints = 0
        if trigger == 'hidden': target.buff_invisible_until_ms = 10000
        if trigger == 'out_of_range': target.pos = Point(3500, 1000)
        if trigger == 'every_tick': source.retarget_each_tick = True
        if trigger == 'legacy': battle.target_lock_mode = 'legacy_periodic_v1'
        battle._acquire_target(source)
        assert source.target_uid == (target.uid if trigger == 'none' else near.uid)


def test_walking_unit_can_be_kited_and_after_attack_flag_still_resets_lock():
    battle, source, target = pair(distance=9000)
    near = battle.add(make_unit(0, unit_profiles()['knight'], -1, Point(3500, 12000)))
    near.deploy_remaining_ms = 0
    battle.now_ms = (500-source.uid*50) % 500
    battle._acquire_target(source)
    assert source.target_uid == near.uid
    source.retarget_after_attack = True
    source.windup_remaining_ms = 0
    assert battle._attack(source, 50)
    assert source.target_uid is None


def test_legacy_state_restores_legacy_rules_and_new_state_keeps_new_rules():
    current = Battle(target_lock_mode='in_range_v1')
    restored = pickle.loads(pickle.dumps(current))
    assert restored.normal_range_mode == 'edge_to_edge_v1'
    assert restored.target_lock_mode == 'in_range_v1'
    old_state = dict(current.__dict__)
    old_state.pop('normal_range_mode'); old_state.pop('target_lock_mode')
    restored.__setstate__(old_state)
    assert restored.normal_range_mode == 'legacy_target_edge_v1'
    assert restored.target_lock_mode == 'legacy_periodic_v1'
    with pytest.raises(ValueError): Battle(normal_range_mode='typo')


def test_retarget_each_tick_is_carried_from_unit_spec():
    spec = replace(unit_profiles()['musketeer'], retarget_each_tick=True)
    assert make_unit(1, spec, 1, Point(0, 0)).retarget_each_tick


def test_training_records_geometry_and_cannot_resume_under_other_rules():
    from sim.continuous_push import ContinuousPushEnv
    env = ContinuousPushEnv(continuous_profile={'source_radius_reach': 1, 'target_lock_in_range': 1})
    _, info = env.reset(seed=123)
    assert info['metrics']['normal_range_mode'] == 'edge_to_edge_v1'
    assert info['metrics']['target_lock_mode'] == 'in_range_v1'
    snapshot = env.snapshot()
    env.restore(snapshot)
    other = ContinuousPushEnv(continuous_profile={'source_radius_reach': 0, 'target_lock_in_range': 1})
    with pytest.raises(ValueError, match='profile differs'): other.restore(snapshot)
    with pytest.raises(ValueError): ContinuousPushEnv(continuous_profile={'source_radius_reach': .5})
