"""Both data paths agree on units; carrying JumpSpeed does not enable jumping."""
import pickle

from sim.arena import Point
from sim.engine import Battle
from sim.entities import make_unit, make_tower
from sim.gamedata import build_unit_spec, load_gamedata


def test_local_and_public_jump_speed_are_converted_once(tmp_path):
    public = load_gamedata(11, root=tmp_path/'absent')['hog_rider'].unit
    local = build_unit_spec('HogRider', dict(Speed=120, JumpEnabled=True, JumpSpeed=160),
                            11, 'Rare', {}, {})
    for spec in (public, local):
        entity = make_unit(1, spec, 1, Point(3500, 12000))
        assert spec.jump_speed_mt_per_sec == entity.jump_speed_mt_per_sec == 3200
        assert entity.speed_mt_per_sec == 2400
        assert entity.jump_enabled


def test_missing_jump_speed_and_old_entity_snapshot_keep_zero_default():
    spec = build_unit_spec('Knight', dict(Speed=60), 11, 'Common', {}, {})
    unit = make_unit(1, spec, 1, Point(3500, 12000))
    assert unit.jump_speed_mt_per_sec == 0
    unit.__dict__.pop('jump_speed_mt_per_sec')
    assert pickle.loads(pickle.dumps(unit)).jump_speed_mt_per_sec == 0


def test_jump_speed_metadata_does_not_change_ordinary_movement():
    spec = load_gamedata(11)['hog_rider'].unit
    battle = Battle()
    hog = battle.add(make_unit(1, spec, 1, Point(14500, 12000)))
    tower = battle.add(make_tower(2, -1, Point(14500, 6500), 10000, 0, 1000, 0))
    hog.target_uid = tower.uid
    hog.deploy_remaining_ms = 0
    battle._move(hog, 50)
    assert hog.pos == Point(14500, 11880)  # normal 2400 mt/s, not jump 3200
