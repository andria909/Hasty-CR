import pytest
import torch
from sim.card_exploration import card_mixture_distribution
from sim.env import ACTIONS
from sim.train_ppo import masked_distribution


def setup():
    logits=torch.zeros(1,ACTIONS);mask=torch.zeros_like(logits,dtype=torch.bool)
    mask[:,0]=True;mask[:,1]=True;mask[:,577:1153]=True
    return logits,mask


def test_equal_card_exploration_is_independent_of_spatial_option_count():
    logits,mask=setup();logits[:,0]=200
    distribution=card_mixture_distribution(logits,mask,.3)
    assert distribution.probs[0,1].item()==pytest.approx(.1)
    assert distribution.probs[0,577:1153].sum().item()==pytest.approx(.1)
    assert distribution.probs[0,0].item()==pytest.approx(.8)
    assert (distribution.probs[~mask]==0).all()
    assert distribution.probs.sum().item()==pytest.approx(1.)


def test_rollout_and_ppo_recompute_identical_actual_mixture_logprob():
    logits,mask=setup();torch.manual_seed(12)
    old=card_mixture_distribution(logits,mask,.2);action=old.sample()
    new=card_mixture_distribution(logits,mask,.2)
    assert (new.log_prob(action)-old.log_prob(action)).exp().item()==pytest.approx(1.)
    assert torch.equal(card_mixture_distribution(logits,mask,0.).logits,masked_distribution(logits,mask).logits)


def test_temperature_control_retains_gradient_for_rare_explored_card():
    logits,mask=setup();logits[0,0]=200;logits.requires_grad_()
    mixed=card_mixture_distribution(logits/10,mask,.2)
    loss=-mixed.log_prob(torch.tensor([1])).mean();loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0,1]<0 and logits.grad[0,0]>0
    assert (logits.grad[~mask]==0).all()


def test_wait_only_mask_and_invalid_exploration_share():
    logits,mask=setup();mask[:]=False;mask[0,0]=True
    assert card_mixture_distribution(logits,mask,.2).probs[0,0].item()==1.
    with pytest.raises(ValueError):card_mixture_distribution(logits,mask,1.)
