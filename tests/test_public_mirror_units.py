"""Portable Mirror copies an actual higher-level unit, never an empty spell."""
import pytest

from sim.arena import Point
from sim.env import DECK_26
from sim.match import Match
from sim.public_gamedata import load_public_gamedata
from sim.public_scenario_spells import load_public_scenario_spells

DECK=['mirror']+list(DECK_26)[:7]


def fixture(side):
    cards,_=load_public_gamedata()
    for seed in range(200):
        match=Match(cards=cards,decks=(DECK,DECK),seed=seed,spells={})
        player=match.players[side]
        if 'knight' in player.hand and player.queue[0]=='mirror':
            player.elixir=10000
            return cards,match,player
    raise AssertionError('No deterministic legal Knight→Mirror opening found')


@pytest.mark.parametrize('side',(1,-1))
def test_knight_then_mirror_is_level12_cost4_and_cycles_mirror_once(side):
    cards,match,player=fixture(side)
    assert cards['mirror'].mirror_units_only and cards['mirror'].unit is None
    assert 'mirror' not in player.hand and not match.can_play(side,'mirror')
    y=23000 if side==1 else 9000
    assert match.play_card(side,'knight',Point(7000,y))
    assert 'mirror' in player.hand and player.elixir==7000
    player.elixir=3999
    assert not match.can_play(side,'mirror')
    player.elixir=7000
    assert match.play_card(side,'mirror',Point(11000,y))
    assert player.elixir==3000 and player.elixir_spent==7
    knights=sorted([e for e in match.battle.entities.values()if e.name=='knight'],key=lambda e:e.uid)
    higher=load_public_gamedata(level=12)[0]['knight'].unit
    assert len(knights)==2
    assert knights[0].max_hitpoints==cards['knight'].unit.hitpoints
    assert (knights[1].max_hitpoints,knights[1].damage)==(higher.hitpoints,higher.damage)
    assert knights[1].max_hitpoints>knights[0].max_hitpoints
    assert knights[1].elixir_value==4000
    assert player.last_card=='knight'
    assert player.queue.count('mirror')==1 and 'mirror' not in player.hand
    assert not match.play_card(side,'mirror',Point(11000,y))
    assert player.elixir==3000


def test_unsupported_spell_copy_and_missing_higher_profile_fail_without_spending(monkeypatch):
    _,match,player=fixture(1)
    player.hand[0]='mirror'
    player.last_card='fireball'
    before=player.elixir
    assert match.mirrored(1)==(None,0)
    assert not match.play_card(1,'mirror',Point(9000,22000))
    assert player.elixir==before
    player.last_card='knight'
    monkeypatch.setattr('sim.match._levelled_cards',lambda level:{})
    assert match.mirrored(1)==(None,0)
    assert not match.can_play(1,'mirror')


def test_mirror_admission_is_explicit_dynamic_unit_copy_not_fake_spell():
    catalog=load_public_scenario_spells(engine_capabilities={'mirror_units_and_opening_rules'})
    assert 'mirror' not in catalog.spells
    assert catalog.audit['mirror']['supported']
    assert catalog.audit['mirror']['representation']=='dynamic_unit_copy'
    assert catalog.audit['mirror']['limitations']
