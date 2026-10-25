import json
from pathlib import Path

import numpy as np
import torch

from sim.fireball_balanced import paired_loss,skill_eligibility,deployed_rehearsal


def protocol():return json.loads(Path('data/fireball/balanced_protocol.json').read_text())


def test_pair_loss_requires_opposite_decisions_not_always_wait_or_always_cast():
    legal=torch.ones((1,3),dtype=torch.bool);fireball=torch.tensor([[False,True,False]]);label=torch.tensor([1])
    def loss(pos,neg):
        a,b,c=paired_loss(torch.tensor([pos]),torch.tensor([neg]),legal,fireball,label)
        return a+b+.5*c
    good=loss([0.,8.,0.],[0.,-8.,0.])
    assert good<loss([0.,-8.,0.],[0.,-8.,0.])
    assert good<loss([0.,8.,0.],[0.,8.,0.])


def test_balanced_pair_gradients_preserve_positive_and_suppress_negative_fireball():
    pos=torch.zeros((1,3),requires_grad=True);neg=torch.zeros((1,3),requires_grad=True)
    a,b,c=paired_loss(pos,neg,torch.ones_like(pos,dtype=torch.bool),torch.tensor([[False,True,False]]),torch.tensor([1]))
    (a+b+.5*c).backward()
    assert pos.grad[0,1]<0 and neg.grad[0,1]>0


def test_collapsed_tower_skill_never_passes_even_if_macro_and_negatives_look_good():
    p=protocol();positive=dict(by_kind={k:dict(successes=32,episodes=32)for k in p['skill_floors']})
    positive['by_kind']['tower_finish']['successes']=0
    assert not skill_eligibility(positive,dict(first_action_fireball_rate=0),1.,dict(first_action_fireball_rate=.3),p)
    positive['by_kind']['tower_finish']['successes']=23
    assert skill_eligibility(positive,dict(first_action_fireball_rate=0),.97,dict(first_action_fireball_rate=.3),p)


def test_deployed_rehearsal_has_actual_public_task_inputs_and_actionable_states():
    from sim.env import ACTIONS
    class WaitPolicy(torch.nn.Module):
        def forward(self,planes,scalars):
            logits=torch.zeros((len(planes),ACTIONS));logits[:,0]=10
            return logits,torch.zeros(len(planes))
    observations,masks,probabilities,contexts=deployed_rehearsal(WaitPolicy(),'cpu',12,protocol())
    assert len(observations)==len(contexts)==12
    assert np.all(masks.sum(-1)>1)
    assert all(o['scalars'][-3:].tolist()==[0.,0.,1.]for o in observations)
    assert probabilities.shape==(12,ACTIONS)
    assert {c['seed']for c in contexts}=={880001}


def test_mechanical_mask_uses_the_match_geometry_for_pocket_cells():
    from sim.match import Match
    from sim.env import DECK_26
    from sim.defense_assets import load_defense_assets
    from sim.runner import resolve_deck
    from sim.eval_defense_duel import mechanical_mask
    cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,DECK_26))
    match=Match(cards=cards,decks=(list(DECK_26),list(DECK_26)),spells=spells,geometry='native_anchors_candidate_v1')
    match.players[1].hand=['hog_rider','knight','cannon','skeletons']
    match.players[1].queue=[c for c in DECK_26 if c not in match.players[1].hand]
    match.towers[-1]['left'].hitpoints=0
    native=mechanical_mask(match,cards,spells,1)
    # This is consistency with the configured pocket hypothesis, not a new
    # claim that its shape has been measured in the real client.
    assert native[1+5*18+3]
    from sim import arena
    match.geometry=arena.LEGACY_GEOMETRY
    assert not mechanical_mask(match,cards,spells,1)[1+5*18+3]
