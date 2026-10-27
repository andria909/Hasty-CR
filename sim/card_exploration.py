"""Explicit card-balanced exploration mixed with the learned distribution.

The same mixture must define rollout sampling, PPO log-prob and entropy.
Evaluation remains the explicitly reported greedy learned policy.
"""
import math
import torch
from sim.env import ACTIONS
from sim.train_ppo import masked_distribution


def card_mixture_distribution(logits,masks,epsilon):
    if not math.isfinite(epsilon)or not 0<=epsilon<=.5:raise ValueError('Invalid card exploration share')
    if logits.ndim!=2 or logits.shape!=masks.shape or logits.shape[1]!=ACTIONS or masks.dtype!=torch.bool:
        raise ValueError('Expected complete legal action batches')
    base=masked_distribution(logits,masks)
    if epsilon==0:return base
    if not masks.any(-1).all():raise ValueError('At least one legal action required')
    cards=masks[:,1:2305].reshape(-1,4,576)
    counts=cards.sum(-1)
    groups=(counts>0).sum(-1)+masks[:,0]+masks[:,2305:].sum(-1)
    exploration=torch.zeros_like(logits)
    exploration[:,0]=masks[:,0]/groups
    exploration[:,1:2305]=(cards/counts.clamp_min(1)[...,None]/groups[:,None,None]).reshape(-1,2304)
    exploration[:,2305:]=masks[:,2305:]/groups[:,None]
    return torch.distributions.Categorical(probs=(1-epsilon)*base.probs+epsilon*exploration)
