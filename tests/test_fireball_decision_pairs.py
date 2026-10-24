import numpy as np
import pytest
import torch

from sim.fireball_decision_pairs import decision_pair,selectivity_loss


@pytest.mark.parametrize('lane',['left','right'])
def test_pair_changes_hp_without_legality_or_private_timer_changes(lane):
    pair=decision_pair(860001,lane)
    assert pair['metadata']['positive_reference_impact_verified']
    assert pair['metadata']['negative_hp']>pair['metadata']['fireball_crown_damage']
    assert pair['action_mask'][pair['reference_action']]
    assert pair['fireball_mask'][pair['reference_action']]
    assert not pair['fireball_mask'][0] and pair['action_mask'][0]
    assert np.array_equal(pair['positive']['scalars'][-3:],pair['negative']['scalars'][-3:])
    assert not np.array_equal(pair['positive']['scalars'],pair['negative']['scalars'])
    assert pair['positive']['scalars'].shape==pair['negative']['scalars'].shape==(309,)


def test_negative_loss_does_not_force_wait_instead_of_other_cards():
    # Fireball is column1; WAIT and another useful card are columns0 and2.
    logits=torch.tensor([[0.,8.,1.]],requires_grad=True)
    legal=torch.ones_like(logits,dtype=torch.bool)
    fireball=torch.tensor([[False,True,False]])
    loss=selectivity_loss(logits,legal,fireball);loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0,1]>0
    assert logits.grad[0,0]<0 and logits.grad[0,2]<0
    assert abs(logits.grad[0,2])>abs(logits.grad[0,0])


def test_loss_is_stable_for_extreme_fireball_preference():
    logits=torch.tensor([[0.,1000.,-1000.]],requires_grad=True)
    loss=selectivity_loss(logits,torch.ones_like(logits,dtype=torch.bool),torch.tensor([[False,True,False]]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_pair_generator_rejects_test_split():
    with pytest.raises(ValueError,match='forbid test'):decision_pair(1020000,split='test')
