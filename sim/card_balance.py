"""Soft, differentiable diversity of card choices across a PPO minibatch.

Aggregate legal spatial probabilities by card identity, not hand slot. WAIT,
unaffordable cards and unavailable cards are not forced into the target. This
is an auxiliary objective; actual accepted card counts and game quality still
need separate validation. No reward is given merely for playing a card.
"""
import math

import torch

from sim.env import DECK_26,NUM_SLOTS,GRID_W,GRID_H,LANES


def card_balance_loss(logits,scalars,masks,target='all_cards'):
    if target not in ('all_cards','units'):raise ValueError('Unknown card-balance target')
    cells=GRID_W*GRID_H;card_end=1+NUM_SLOTS*cells;hand_start=3+2*len(LANES)
    if logits.ndim!=2 or masks.shape!=logits.shape or scalars.ndim!=2 or len(scalars)!=len(logits):
        raise ValueError('Card-balance batch shapes differ')
    if not bool(masks.any(dim=-1).all()):raise ValueError('Every state requires a legal action')
    hands=scalars[:,hand_start:hand_start+NUM_SLOTS*len(DECK_26)].reshape(-1,NUM_SLOTS,len(DECK_26))
    if not bool(torch.isclose(hands.sum(-1),torch.ones_like(hands.sum(-1))).all()):raise ValueError('Complete own-hand identities required')
    probabilities=torch.softmax(logits.masked_fill(~masks,-torch.inf),dim=-1)
    slot_mass=probabilities[:,1:card_end].reshape(-1,NUM_SLOTS,cells).sum(-1)
    mass=(slot_mass.unsqueeze(-1)*hands).sum((0,1))
    slot_legal=masks[:,1:card_end].reshape(-1,NUM_SLOTS,cells).any(-1)
    available=(slot_legal.unsqueeze(-1)*hands).sum((0,1))>0
    if target=='units':
        available=available&torch.tensor([card not in ('fireball','zap')for card in DECK_26],device=logits.device)
    if not bool(available.any()):return logits[masks].sum()*0
    mass=mass*available
    if float(mass.sum().detach())<=1e-12:return logits[masks].sum()*0
    actual=mass/mass.sum();desired=available.to(actual.dtype)/available.sum()
    middle=(actual+desired)/2
    epsilon=torch.finfo(actual.dtype).tiny
    divergence=.5*((actual*(actual.clamp_min(epsilon).log()-middle.clamp_min(epsilon).log())).sum()+
                   (desired*(desired.clamp_min(epsilon).log()-middle.clamp_min(epsilon).log())).sum())
    return divergence/math.log(2)
