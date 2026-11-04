"""Opt-in PPO actor reductions; physical-time GAE and critic stay unchanged."""
import torch

from sim.train_defense_ppo import normalize_advantages

MODES = ('all', 'choice')


def validate_modes(reduction, normalization):
    if reduction not in MODES or normalization not in MODES:
        raise ValueError('Policy reduction and advantage normalization must be all or choice')


def choices(mask):
    return mask.sum(dim=-1) > 1


def normalized_advantages(advantages, choice, mode):
    if mode not in MODES or advantages.shape != choice.shape or choice.dtype != torch.bool:
        raise ValueError('Invalid policy advantage normalization or choice mask')
    if mode == 'all':
        return normalize_advantages(advantages)
    selected = advantages[choice]
    # A singleton cannot estimate variance; retain its raw advantage instead of
    # centering it to zero and discarding the only actual choice in the rollout.
    normalized = normalize_advantages(selected) if selected.numel() > 1 else selected
    result = torch.zeros_like(advantages)
    return result.masked_scatter(choice, normalized)


def reduce_actor(terms, choice, mode):
    if mode not in MODES or terms.shape != choice.shape or choice.dtype != torch.bool:
        raise ValueError('Invalid policy reduction or choice mask')
    if mode == 'all':
        return terms.mean()
    # Keep a differentiable exact zero if there is no choice in a minibatch.
    return terms.masked_fill(~choice, 0).sum() / choice.sum().clamp_min(1)
