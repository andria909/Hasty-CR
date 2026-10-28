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


def test_empty_board_does_not_skip_inflight_enemy_fireball():
    from sim.match import Match
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    match=Match(cards=cards,spells=spells,decks=(list(DECK_26),list(DECK_26)),seed=19,
                geometry='native_anchors_candidate_v1')
    player=match.players[-1];rest=[c for c in player.deck if c!='fireball']
    player.hand=['fireball',*rest[:3]];player.queue=rest[3:];player.elixir=10000
    hp={k:t.hitpoints for k,t in match.towers[1].items()}
    assert match.play_card(-1,'fireball',match.towers[1]['king'].pos)
    assert match.towers[1]['king'].hitpoints==hp['king']
    class NoNewDecisions:
        def __call__(self,*args):raise AssertionError('No new cards after the last enemy body')
    row=episode(NoNewDecisions(),{},cards,spells,initial=(match,hp))
    assert row['total_damage']>0 and row['damage']['king']>0
    assert not row['clear_without_damage'] and row['remaining_enemies']==0
    assert row['actions']==row['decisions']==[]
