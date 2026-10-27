import json
from pathlib import Path
import torch
from scripts.attack.probe_late_air_policies import episode
from sim.defense_assets import load_defense_assets
from sim.env import ACTIONS,DECK_26
from sim.runner import resolve_deck


class WaitPolicy:
    def __call__(self,planes,scalars):
        logits=torch.zeros((1,ACTIONS));logits[:,0]=100
        return logits,torch.zeros(1)


def test_no_teacher_actions_are_injected_for_wait_policy():
    record=json.loads(Path('reports/late_air_native_search_20261027.json').read_text())['contexts'][0]['opening']
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    row=episode(WaitPolicy(),record,cards,spells)
    assert row['actions']==[] and row['elixir_spent']==0
    assert all(d['action']==0 for d in row['decisions'])
    assert row['total_damage']>0 and not row['clear_without_damage']
    assert row['elapsed_ms']<=60000 and row['rejections']==0
