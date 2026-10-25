import pytest
import torch

from sim.card_balance import card_balance_loss
from sim.env import ACTIONS,NUM_SCALARS,DECK_26


def batch():
    logits=torch.zeros((1,ACTIONS),requires_grad=True)
    scalars=torch.zeros((1,NUM_SCALARS+3));mask=torch.zeros((1,ACTIONS),dtype=torch.bool);mask[0,0]=True
    for slot,card in enumerate(('hog_rider','musketeer','cannon','fireball')):
        scalars[0,7+slot*len(DECK_26)+DECK_26.index(card)]=1
        mask[0,1+slot*576]=True
    return logits,scalars,mask


def test_equal_card_mass_has_zero_penalty_and_absent_cards_are_not_forced():
    logits,scalars,mask=batch()
    assert abs(float(card_balance_loss(logits,scalars,mask).detach()))<1e-7
    mask[0,1+3*576]=False
    assert abs(float(card_balance_loss(logits,scalars,mask).detach()))<1e-7


def test_gradient_reduces_spam_and_increases_underused_available_card():
    logits,scalars,mask=batch()
    with torch.no_grad():logits[0,1]=8
    loss=card_balance_loss(logits,scalars,mask);loss.backward()
    assert 0<float(loss.detach())<=1
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0,1]>0
    assert logits.grad[0,1+576]<0


def test_wait_only_never_demands_an_unaffordable_play():
    logits,scalars,mask=batch();mask[:]=False;mask[0,0]=True
    loss=card_balance_loss(logits,scalars,mask);loss.backward()
    assert float(loss.detach())==0
    assert torch.isfinite(logits.grad).all()


def test_unit_target_does_not_force_spells_to_match_unit_frequency():
    logits,scalars,mask=batch()
    with torch.no_grad():logits[0,1+3*576]=8
    assert float(card_balance_loss(logits,scalars,mask,'all_cards').detach())>.1
    assert abs(float(card_balance_loss(logits,scalars,mask,'units').detach()))<1e-7


def test_missing_own_hand_and_illegal_target_fail_explicitly():
    logits,scalars,mask=batch();scalars[:]=0
    with pytest.raises(ValueError,match='own-hand'):card_balance_loss(logits,scalars,mask)
    with pytest.raises(ValueError,match='Unknown'):card_balance_loss(logits,scalars,mask,'invented')
