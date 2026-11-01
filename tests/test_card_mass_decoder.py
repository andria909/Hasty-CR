import torch

from scripts.attack.evaluate_card_mass_decoder import choose


def test_card_mass_choice_preserves_best_legal_cell_and_ignores_illegal_logits():
    logits=torch.full((1,2306),-10.);mask=torch.zeros_like(logits,dtype=torch.bool)
    mask[:,0]=True;mask[:,1:37]=True
    logits[:,0]=1.;logits[:,1:37]=0.;logits[:,8]=.5
    logits[:,900]=10000.
    assert choose(logits,mask,'flat')==0
    assert choose(logits,mask,'card_mass')==8


def test_unavailable_cards_have_no_mass_and_forced_wait_remains_wait():
    logits=torch.zeros((1,2306));mask=torch.zeros_like(logits,dtype=torch.bool);mask[:,0]=True
    assert choose(logits,mask,'card_mass')==0
    mask[:,1200]=True;logits[:,1200]=2.
    assert choose(logits,mask,'card_mass')==1200
