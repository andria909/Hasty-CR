"""Portable Miner emergence and tower damage, without private client files."""
from sim import arena
from sim.entities import make_unit
from sim.env import DECK_26
from sim.match import Match
from sim.public_gamedata import load_public_gamedata


def miner_match():
    cards,_=load_public_gamedata(level=11)
    deck=['miner']+[name for name in DECK_26 if name!='miner'][:7]
    match=Match(cards=cards,decks=(deck,DECK_26),seed=42,spells={})
    player=match.players[1]
    player.hand=deck[:4]
    player.queue=deck[4:]
    player.elixir=10000
    return cards,match


def test_public_miner_retains_exact_burrow_and_reduced_tower_damage():
    cards,_=load_public_gamedata(level=11)
    spec=cards['miner'].unit
    assert spec.raw['spawn_pathfind_speed']==650
    assert spec.burrow_speed_mt_per_sec==650*1000//60
    assert spec.tower_damage_pct==25
    unit=make_unit(0,spec,1,arena.Point(3500,3000),0)
    assert unit.burrow_speed_mt_per_sec==spec.burrow_speed_mt_per_sec
    assert unit.tower_damage_pct==25
    assert unit.speed_mt_per_sec==1800  # independent walking vs tunnelling speed


def test_miner_can_target_enemy_tower_but_cannot_damage_it_underground():
    cards,match=miner_match()
    tower=match.towers[-1]['left']
    before=tower.hitpoints
    assert match.play_card(1,'miner',tower.pos)
    miner=next(e for e in match.battle.entities.values() if e.name=='miner')
    expected=cards['miner'].unit.deploy_time_ms+arena.distance(arena.ALLY_KING,tower.pos)*1000//miner.burrow_speed_mt_per_sec
    assert miner.deploy_remaining_ms==expected and miner.underground
    while match.elapsed_ms+50 < expected:
        match.step()
        assert tower.hitpoints==before
    for _ in range(100):
        match.step()
        if tower.hitpoints<before:
            break
    assert tower.hitpoints==before-cards['miner'].unit.damage*25//100
    assert not miner.underground
