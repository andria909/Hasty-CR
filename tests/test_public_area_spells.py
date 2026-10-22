import json

import pytest

from sim import arena
from sim.engine import Battle
from sim.entities import make_tower
from sim.public_area_spells import DATA,load_public_area_spells
from sim.spells import apply_spell


def target(battle,*,tower=False,building=False,flying=False,x=9000):
    entity=make_tower(0,-1,arena.Point(x,16000),100000,0,1000,0)
    entity.is_tower=tower;entity.is_building=building or tower;entity.flying=flying
    entity.collision_radius_mt=0
    return battle.add(entity)


def test_exact_public_area_specs_are_temporal_and_expose_limits():
    result=load_public_area_spells()
    poison=result.spells['poison'];eq=result.spells['earthquake'];tornado=result.spells['tornado']
    assert (poison.damage_per_second,poison.hit_frequency_ms,poison.life_duration_ms)==(91,1000,8000)
    assert poison.tower_damage_per_hit==21 and poison.tower_damage_override==168
    assert (eq.damage_per_second,eq.building_damage_percent,eq.tower_damage_per_hit)==(82,350,49)
    assert not eq.hits_air and eq.hits_ground
    assert (tornado.damage_per_second,tornado.hit_frequency_ms,tornado.life_duration_ms,tornado.attract_percentage)==(169,500,1050,360)
    for key,spec in result.spells.items():
        assert spec.damage==0 and not spec.initial_damage
        assert result.audit[key]['supported'] and result.audit[key]['source_stats_exact']
        assert result.audit[key]['mechanics_verified'] is False
        assert result.audit[key]['source_sha256'] and result.audit[key]['limitations']


@pytest.mark.parametrize('name,duration,troop_damage,tower_damage',[
    ('poison',8000,728,168),('earthquake',3000,246,147),('tornado',1050,168,28)])
def test_real_area_engine_has_no_instant_damage_and_exact_stationary_ticks(name,duration,troop_damage,tower_damage):
    spec=load_public_area_spells().spells[name]
    battle=Battle();troop=target(battle);tower=target(battle,tower=True)
    apply_spell(battle,spec,arena.Point(9000,16000),1)
    assert troop.hitpoints==tower.hitpoints==100000
    assert len(battle.areas)==1
    for now in range(50,duration+51,50):
        battle.now_ms=now;battle._tick_areas()
    assert 100000-troop.hitpoints==troop_damage
    assert 100000-tower.hitpoints==tower_damage
    assert not battle.areas
    assert all(event[0]>0 for event in battle.damage_log)


def test_earthquake_excludes_air_and_applies_raw_building_multiplier():
    battle=Battle();air=target(battle,flying=True);building=target(battle,building=True)
    apply_spell(battle,load_public_area_spells().spells['earthquake'],arena.Point(9000,16000),1)
    battle.now_ms=50;battle._tick_areas()
    assert air.hitpoints==100000
    assert 100000-building.hitpoints==287  # exact existing-engine interpretation: floor(82*350/100)


def test_tornado_pull_moves_troops_but_not_buildings():
    battle=Battle();troop=target(battle,x=10000);building=target(battle,building=True,x=10000)
    apply_spell(battle,load_public_area_spells().spells['tornado'],arena.Point(9000,16000),1)
    battle.now_ms=50;battle._tick_area_attraction(50)
    assert troop.pos.x==9820
    assert building.pos.x==10000


def test_overrides_do_not_extrapolate_and_unequal_ticks_fail_closed(tmp_path):
    result=load_public_area_spells(level=12)
    assert 'poison' not in result.spells and 'earthquake' not in result.spells
    assert 'tornado' in result.spells
    for name in ('cards_stats_spell.json','cards_stats_character_buff.json','rarities.json','combat_rules.json'):
        payload=json.loads((DATA/name).read_text())
        if name=='combat_rules.json':payload['rules']['poison']['tower_damage_override']=169
        (tmp_path/name).write_text(json.dumps(payload))
    result=load_public_area_spells(root=tmp_path)
    assert 'poison' not in result.spells
    assert any('equal exact ticks' in message for message in result.audit['poison']['missing'])
