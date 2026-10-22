"""Public Collector economy and the real opening-hand exclusion rules."""
from collections import Counter
import random

import pytest

from sim.arena import Point
from sim.env import DECK_26
from sim.match import Match,STARTING_HAND_EXCLUSIONS,starting_card_order
from sim.public_gamedata import load_public_gamedata,load_public_elixir_table
from sim.public_scenario_spells import load_public_scenario_spells

DECK=['elixir_collector','mirror']+list(DECK_26)[:6]


def fixture(side):
    cards,_=load_public_gamedata()
    match=Match(cards=cards,decks=(DECK,DECK),seed=42,spells={})
    p=match.players[side]
    # A later-game fixture, after cycling the Collector into the hand.
    p.queue.remove('elixir_collector')
    replaced=p.hand[0];p.hand[0]='elixir_collector';p.queue.append(replaced)
    p.elixir=10000
    assert match.play_card(side,'elixir_collector',Point(9000,23000 if side==1 else 9000))
    unit=next(e for e in match.battle.entities.values()if e.name=='elixir_collector')
    p.elixir=0
    return match,p,unit


def test_public_collector_profile_is_real_building_not_zero_hp_placeholder():
    cards,_=load_public_gamedata(level=11)
    card=cards['elixir_collector']
    assert card.cost==6 and card.unit.from_building_card
    assert card.unit.hitpoints==1070
    assert card.unit.lifetime_ms==65000 and card.unit.deploy_time_ms==1000
    assert load_public_gamedata(level=12)[0]['elixir_collector'].unit.hitpoints==1176
    assert load_public_elixir_table()['elixir_collector']==dict(generate_ms=9000,amount_milli=1000,death_milli=1000,death_opp_milli=0)
    audit=load_public_scenario_spells(engine_capabilities={'elixir_collector_income_and_starting_hand'}).audit['elixir_collector']
    assert audit['supported'] and audit['representation']=='building'


def test_opening_exclusions_apply_to_both_players_and_preserve_random_queue_positions():
    cards,_=load_public_gamedata()
    positions={name:set()for name in STARTING_HAND_EXCLUSIONS}
    for seed in range(80):
        match=Match(cards=cards,decks=(DECK,DECK),seed=seed,spells={})
        repeat=Match(cards=cards,decks=(DECK,DECK),seed=seed,spells={})
        for side in (1,-1):
            p=match.players[side]
            assert not STARTING_HAND_EXCLUSIONS.intersection(p.hand)
            assert Counter(p.hand+p.queue)==Counter(DECK)
            assert p.hand==repeat.players[side].hand and p.queue==repeat.players[side].queue
            for name in positions:positions[name].add(p.queue.index(name))
    assert all(value=={0,1,2,3}for value in positions.values())
    # Established v25 decks keep exactly the previous seeded shuffle.
    expected=list(DECK_26);random.Random(42).shuffle(expected)
    assert starting_card_order(DECK_26,random.Random(42))==expected


@pytest.mark.parametrize('side',(1,-1))
def test_collector_produces_after_nine_seconds_and_pays_one_on_death_once(side):
    match,p,unit=fixture(side)
    for _ in range(179):match._unit_elixir(50)
    assert p.elixir==0
    match._unit_elixir(50)
    assert p.elixir==1000
    unit.hitpoints=0
    match._unit_elixir(0)
    assert p.elixir==2000
    match._unit_elixir(0)
    assert p.elixir==2000


@pytest.mark.parametrize('side',(1,-1))
def test_full_lifetime_yields_seven_productions_plus_death_payout(side):
    match,p,unit=fixture(side)
    for _ in range(1300):
        match.battle.step(50)
        match._unit_elixir(50)
    assert not unit.alive
    assert p.elixir==8000


def test_same_tick_fresh_collector_death_still_receives_death_payout():
    match,p,unit=fixture(1)
    unit.hitpoints=0
    match._unit_elixir(0)
    assert p.elixir==1000
