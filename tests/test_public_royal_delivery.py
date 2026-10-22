"""Royal Delivery is a delayed 3-elixir spell, never an immortal dummy body."""
from sim.arena import Point
from sim.engine import Battle
from sim.entities import make_unit,make_tower
from sim.env import DECK_26
from sim.match import Match
from sim.public_gamedata import load_public_gamedata
from sim.public_scenario_spells import load_public_scenario_spells
from sim.spells import cast_spell


def test_delivery_card_uses_catalogue_cost_and_has_no_stationary_dummy_unit():
    cards,_=load_public_gamedata()
    assert cards['royal_delivery'].cost==3
    assert cards['royal_delivery'].unit is None


def test_delivery_delay_damage_air_ground_excludes_buildings_and_spawns_recruit():
    cards,units=load_public_gamedata()
    battle=Battle()
    battle.unit_lookup=lambda name:units.get(name)
    aim=Point(9000,21000)
    ground=make_unit(0,cards['knight'].unit,-1,aim,0)
    air=make_unit(0,cards['baby_dragon'].unit,-1,aim,0)
    tower=make_tower(0,-1,aim,3000,0,1000,7500)
    for e in (ground,air,tower):battle.add(e)
    hp=[e.hitpoints for e in (ground,air,tower)]
    cast_spell(battle,load_public_scenario_spells().spells['royal_delivery'],aim,1)
    assert len(battle.spell_impacts)==1
    battle.now_ms=2999;battle._resolve_spell_impacts()
    assert [e.hitpoints for e in (ground,air,tower)]==hp
    battle.now_ms=3000;battle._resolve_spell_impacts()
    assert ground.hitpoints==hp[0]-384
    assert air.hitpoints==hp[1]-384
    assert tower.hitpoints==hp[2]
    recruits=[e for e in battle.entities.values()if e.side==1]
    assert len(recruits)==1 and recruits[0].name=='delivery_recruit'
    assert recruits[0].deploy_remaining_ms==250
    assert not battle.spell_impacts


def test_match_rejects_enemy_half_delivery_and_charges_three_elixir_on_own_half():
    cards,_=load_public_gamedata()
    deck=['royal_delivery']+list(DECK_26)[:7]
    spell=load_public_scenario_spells().spells['royal_delivery']
    match=Match(cards=cards,decks=(deck,DECK_26),seed=42,spells={'royal_delivery':spell})
    p=match.players[1];p.hand=deck[:4];p.queue=deck[4:];p.elixir=10000
    assert not match.play_card(1,'royal_delivery',Point(9000,10000))
    assert p.elixir==10000 and not match.battle.spell_impacts
    assert match.play_card(1,'royal_delivery',Point(9000,22000))
    assert p.elixir==7000 and len(match.battle.spell_impacts)==1
