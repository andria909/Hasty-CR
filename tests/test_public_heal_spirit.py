"""Source-derived Heal Spirit healing, with exact public rarity multipliers."""
import json
from pathlib import Path

import pytest

from sim.arena import Point
from sim.engine import Battle
from sim.entities import make_unit,make_tower
from sim.public_gamedata import load_public_gamedata
from sim.public_stat_scaling import scale_public_stat

DATA=Path(__file__).resolve().parents[1]/'data/royaleapi'


def test_exact_rarity_multiplier_reproduces_all_public_damage_arrays():
    rarities={r['name']:r for r in json.loads((DATA/'rarities.json').read_text())}
    checked=0
    for name in ('cards_stats_characters.json','cards_stats_projectile.json'):
        for row in json.loads((DATA/name).read_text()):
            array=row.get('damage_per_level')
            if not array or not row.get('damage') or row.get('rarity') not in rarities:continue
            rarity=rarities[row['rarity']]
            i=rarity['tournament_level_index']
            assert scale_public_stat(row['damage'],row['rarity'],11,rarities)==array[i]
            checked+=1
    assert checked>=140
    with pytest.raises(ValueError):
        scale_public_stat(189,'Rare',1000,rarities)


def test_public_heal_spirit_is_troop_with_current_hp_and_exact_healing_rate():
    cards,_=load_public_gamedata()
    card=cards['heal_spirit']
    assert card.cost==1 and card.unit is not None
    spirit=card.unit
    assert spirit.hitpoints==215 and spirit.cannot_target_towers
    assert spirit.damage==110
    assert spirit.projectile_heal_per_second==400 # floor(189*212/100), exact public multiplier
    assert spirit.projectile_heal_radius_mt==2500
    assert spirit.projectile_heal_duration_ms==1000
    assert spirit.projectile_heal_tick_ms==250


def test_heal_spirit_impact_heals_friendlies_over_source_duration_not_enemies_or_towers():
    cards,_=load_public_gamedata()
    battle=Battle()
    spirit=make_unit(0,cards['heal_spirit'].unit,1,Point(9000,20000),0)
    enemy=make_unit(0,cards['knight'].unit,-1,Point(9000,20000),0)
    ally=make_unit(0,cards['knight'].unit,1,Point(9500,20000),0)
    far=make_unit(0,cards['knight'].unit,1,Point(15000,20000),0)
    tower=make_tower(0,1,Point(9000,20000),3000,0,1000,7500)
    for e in (spirit,enemy,ally,far,tower):battle.add(e)
    ally.hitpoints-=500;far.hitpoints-=500;tower.hitpoints-=500
    before=ally.hitpoints;enemy_before=enemy.hitpoints;tower_before=tower.hitpoints
    battle._land(spirit,enemy,spirit.damage)
    assert enemy.hitpoints==enemy_before-110
    assert ally.hitpoints==before
    assert len(battle.areas)==1
    for now,healed in ((249,0),(250,100),(500,200),(750,300),(1000,400),(1500,400)):
        battle.now_ms=now;battle._tick_areas()
        assert ally.hitpoints==before+healed
        assert far.hitpoints==far.max_hitpoints-500
        assert tower.hitpoints==tower_before
        assert enemy.hitpoints==enemy_before-110
    assert not battle.areas


def test_heal_spirit_kamikaze_projectile_really_triggers_healing_after_source_dies():
    cards,_=load_public_gamedata()
    battle=Battle()
    spirit=make_unit(0,cards['heal_spirit'].unit,1,Point(9000,21000),0)
    spirit.deploy_remaining_ms=0
    ally=make_unit(0,cards['knight'].unit,1,Point(10000,20000),0)
    ally.hitpoints-=500;ally.speed_mt_per_sec=0;ally.attacks_ground=False
    enemy=make_unit(0,cards['knight'].unit,-1,Point(9000,19000),0)
    enemy.speed_mt_per_sec=0;enemy.damage=0
    for entity in (spirit,ally,enemy):battle.add(entity)
    before=ally.hitpoints
    for _ in range(80):battle.step()
    assert not spirit.alive
    assert ally.hitpoints==before+400
    assert enemy.hitpoints==enemy.max_hitpoints-110
    assert not battle.areas and not battle.in_flight
