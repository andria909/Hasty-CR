"""A zero-HP bomb carrier must resolve its source countdown and disappear."""
from sim.arena import Point
from sim.engine import Battle
from sim.entities import make_unit
from sim.public_gamedata import load_public_gamedata


def test_bomb_tower_bomb_source_fuse_and_scaled_blast_resolve_once():
    cards,units=load_public_gamedata()
    spec=units['BombTowerBomb']
    assert spec.raw['hitpoints']==0
    assert spec.lifetime_ms==spec.raw['deploy_time']==3000
    assert spec.death_damage==222
    assert spec.always_untargetable
    assert spec.raw['_public_profile_limitations']
    battle=Battle()
    bomb=make_unit(0,spec,-1,Point(9000,20000),0)
    victim=make_unit(0,cards['knight'].unit,1,Point(9000,22000),0)
    victim.speed_mt_per_sec=0;victim.damage=0
    battle.add(bomb);battle.add(victim)
    before=victim.hitpoints
    for _ in range(59):battle.step()
    assert bomb.alive and victim.hitpoints==before
    battle.step()
    assert not bomb.alive and victim.hitpoints==before-222
    for _ in range(60):battle.step()
    assert victim.hitpoints==before-222


def test_hold_only_bomb_tower_back_does_not_leave_an_immortal_bomb():
    from sim.defense_env import DefenseEnv
    from sim.defense_scenarios import Scenario,ScheduledPlay
    scenario=Scenario('bomb-tower-back-regression','solo:bomb_tower','simple','train','left',
                      (ScheduledPlay('bomb_tower',position='back'),))
    env=DefenseEnv(scenario=scenario,split='train',max_ms=180000)
    env.reset(seed=42)
    for _ in range(360):
        _,_,terminal,truncated,info=env.step(0)
        if terminal or truncated:break
    assert terminal and not truncated
    assert info['terminal_reason']=='cleared'
    assert info['metrics']['elapsed_ms']<40000
