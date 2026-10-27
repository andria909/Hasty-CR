import json
from pathlib import Path
import pytest
from scripts.attack.audit_air_label_latency import opening,outcome
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.runner import resolve_deck


def test_original_air_success_and_quantized_delay_are_reproduced_without_mutation():
    m=json.loads(Path('data/joint_skill_examples_20261027/manifest.json').read_text())
    record=next(r for r in m['records']if r.get('skill')=='air_defense' and r['accepted'])
    target=next(t for t in record['trials']if t['success'])
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    match,hp=opening(record,cards,spells);bank=match.players[1].elixir
    short=outcome(match,hp,target['action'],225)
    assert short['success'] and short['own_damage']==target['own_damage']==0
    assert short['placed_ms']==record['decision_ms']+250
    assert outcome(match,hp,target['action'],1350)['placed_ms']==record['decision_ms']+1350
    assert match.elapsed_ms==record['decision_ms'] and match.players[1].elixir==bank


@pytest.mark.parametrize('delay',[-1,10001,float('nan')])
def test_invalid_delay_rejected(delay):
    with pytest.raises(ValueError,match='delay'):outcome(None,{},0,delay)
