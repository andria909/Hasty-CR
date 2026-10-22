"""Super Archer's area travels; zero direct damage does not mean inert shots."""
from sim.arena import Point
from sim.engine import Battle
from sim.entities import make_unit
from sim.public_gamedata import load_public_gamedata


def fixture():
    cards,_=load_public_gamedata()
    battle=Battle()
    source=make_unit(0,cards['super_archers'].unit,1,Point(9000,21000),0)
    target=make_unit(0,cards['knight'].unit,-1,Point(9000,16000),0)
    ally=make_unit(0,cards['knight'].unit,1,Point(9500,16000),0)
    for e in (source,target,ally):battle.add(e)
    return battle,source,target,ally


def test_public_follow_parent_profile_keeps_exact_fields_and_geometry_limitations():
    battle,source,_,_=fixture()
    assert source.damage==0
    cfg=source.projectile_follow_area
    assert (cfg['damage_per_second'],cfg['radius_mt'],cfg['duration_ms'])==(242,3000,1050)
    assert (cfg['hit_frequency_ms'],cfg['attract_percentage'])==(500,450)
    assert (cfg['projectile_speed_mt_per_sec'],cfg['projectile_range_mt'])==(6666,10000)
    assert cfg['declared_homing_time_ms']==100
    assert cfg['declared_attract_max_angle']==90
    cards,_=load_public_gamedata()
    assert 'follow_parent_area_angular_pull_mask_not_modeled' in cards['super_archers'].unit.raw['_public_profile_limitations']


def test_actual_attack_launches_moving_area_not_fake_instant_impact_damage():
    battle,source,target,ally=fixture()
    before=target.hitpoints
    battle._deal_damage(source,target)
    assert len(battle.areas)==1 and not battle.in_flight
    assert target.hitpoints==before
    for now,damage in ((250,0),(499,0),(500,121),(1000,242)):
        battle.now_ms=now;battle._tick_areas()
        assert target.hitpoints==before-damage
        expected_y=21000-now*6666//1000
        assert abs(battle.areas[0][1].y-expected_y)<=1
        assert ally.hitpoints==ally.max_hitpoints
    # The area keeps flying beyond the initial target, then ends at its
    # declared duration rather than becoming a stationary indefinite wound.
    assert battle.areas[0][1].y<target.pos.y
    battle.now_ms=1050;battle._tick_areas()
    assert not battle.areas and target.hitpoints==before-242


def test_moving_area_pull_uses_current_center_and_excludes_allies():
    battle,source,target,ally=fixture()
    battle._deal_damage(source,target)
    battle.now_ms=500;battle._tick_areas()
    centre=battle.areas[0][1]
    before=target.pos
    ally_before=ally.pos
    battle._tick_area_attraction(50)
    assert abs(target.pos.y-centre.y)<abs(before.y-centre.y)
    assert ally.pos==ally_before
    assert target.pos.y-before.y==225 # public450% of tile/sec over50ms
