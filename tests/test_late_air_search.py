import json
from pathlib import Path
from scripts.attack.audit_air_label_latency import opening
from scripts.attack.search_late_air_defense import rollout
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.runner import resolve_deck


def fixture(bank):
    manifest=json.loads(Path('data/joint_skill_examples_20261027/manifest.json').read_text())
    record=next(r for r in manifest['records']if r.get('skill')=='air_defense' and r['accepted'] and r['seed']==3800009 and r['reaction_ms']==1500 and r['initial_own_elixir']==bank)
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    match,hp=opening(record,cards,spells)
    return match,hp,cards,spells,record['lane']


def test_other_slot_can_be_requested_before_first_card_is_delivered():
    match,hp,cards,spells,lane=fixture(10)
    result=rollout(match,hp,cards,spells,lane,dict(card='cannon',cell=[7,22]),dict(card='musketeer',offset_ms=250,variant=2))
    assert result['valid'] and len(result['commands'])==2
    a,b=result['commands'];assert a['slot']!=b['slot']
    assert b['requested_ms']==a['requested_ms']+250<a['placed_ms']
    assert a['placed_ms']==a['requested_ms']+1350
    assert b['placed_ms']==b['requested_ms']+1350
    assert result['elixir_spent']==7
    assert match.elapsed_ms==2000 and match.players[1].elixir_spent==0


def test_pending_card_reserves_elixir_against_unaffordable_second_card():
    match,hp,cards,spells,lane=fixture(5)
    result=rollout(match,hp,cards,spells,lane,dict(card='musketeer',cell=[3,24]),dict(card='fireball',offset_ms=250,variant=0))
    assert not result['valid']
    assert result['elixir_spent']==4
    assert 'not_legal_or_unreserved_elixir'in result['invalid_reasons']
    assert [r['card']for r in result['commands']]==['musketeer']
