"""Damage-bearing projectile links must survive the public fallback."""
from sim.arena import Point
from sim.engine import Battle
from sim.entities import make_unit,make_tower
from sim.public_gamedata import load_public_gamedata


def test_princess_follows_custom_first_projectile_instead_of_decoration():
    cards,_=load_public_gamedata()
    princess=cards['princess'].unit
    assert princess.raw['projectile']=='PrincessProjectileDeco'
    assert princess.raw['_public_damage_projectile']=='PrincessProjectile'
    assert princess.damage==169


def test_princess_long_range_projectile_really_damages_tower_once_per_shot():
    cards,_=load_public_gamedata()
    battle=Battle()
    princess=make_unit(0,cards['princess'].unit,-1,Point(3500,14500),0)
    princess.deploy_remaining_ms=0
    tower=make_tower(0,1,Point(3500,24500),3000,100,800,7500)
    battle.add(princess);battle.add(tower)
    before=tower.hitpoints
    for _ in range(200):
        battle.step()
        if tower.hitpoints<before:break
    assert tower.hitpoints==before-169
    assert princess.hitpoints==princess.max_hitpoints
    assert battle.now_ms<10000


def test_firecracker_follows_spawn_projectile_but_does_not_invent_full_scatter_hit():
    cards,_=load_public_gamedata()
    firecracker=cards['firecracker'].unit
    assert firecracker.raw['projectile']=='FirecrackerProjectile'
    assert firecracker.raw['_public_damage_projectile']=='FirecrackerExplosion'
    assert firecracker.damage==64
    assert firecracker.raw['_public_fragment_count']==5
    assert 'projectile_fragment_count_and_scatter_geometry_not_modeled' in firecracker.raw['_public_profile_limitations']
