"""Paired TRAIN controls for cast selectivity; no changes to frozen v1 tasks.

The negative is an empty, full-HP enemy tower, not simply HP one point above
Fireball damage: Fireball+Zap and pre-casting into supported pushes can be valid.
The loss suppresses unhelpful Fireball preference without teaching every other
card to wait. These controls add no privileged feature to the policy input.
"""
import numpy as np

from sim.fireball_curriculum import FireballEnv

CONTRACT='fireball_empty_board_decision_pairs_v1'


def decision_pair(seed,lane='left',split='train'):
    if split not in ('train','validation'):raise ValueError('Controls forbid test-driven training')
    env=FireballEnv(split=split,kind='tower_finish',lane=lane)
    env.reset(seed=seed)
    target=env.match.towers[-1][lane]
    positive_hp=target.hitpoints
    spell_damage=env._spells['fireball'].damage_to(target)
    if not 0<positive_hp<=spell_damage<target.max_hitpoints:
        raise ValueError('Invalid paired crown-damage boundary')
    positive=env._observe()
    support=env._action_mask()
    reference=env.reference_action()
    slot=env.match.players[1].hand.index('fireball')
    fireball_mask=np.zeros_like(support)
    fireball_mask[1+slot*576:1+(slot+1)*576]=support[1+slot*576:1+(slot+1)*576]
    target.hitpoints=target.max_hitpoints
    negative=env._observe()
    if not np.array_equal(support,env._action_mask()):raise ValueError('Paired control changed action legality')
    target.hitpoints=positive_hp
    # Keep the teacher label only if the original engine actually finishes.
    success=False
    for step in range(30):
        _,_,done,truncated,info=env.step(reference if step==0 else 0)
        if done or truncated:
            success=info['metrics']['fireball_success'];break
    if not success:raise ValueError('Positive paired reference failed its actual impact')
    return dict(positive=positive,negative=negative,action_mask=support,fireball_mask=fireball_mask,
                reference_action=reference,metadata=dict(contract=CONTRACT,split=split,seed=seed,lane=lane,
                    positive_hp=positive_hp,negative_hp=target.max_hitpoints,fireball_crown_damage=spell_damage,
                    initial_hand=list(env._initial_hand),initial_elixir=env.scenario.initial_elixir,
                    positive_reference_impact_verified=True))


def selectivity_loss(logits,legal_mask,fireball_mask):
    """Negative-only mass penalty; leaves legal non-Fireball action choice free."""
    import torch
    if (logits.shape!=legal_mask.shape or logits.shape!=fireball_mask.shape or
            torch.any(fireball_mask & ~legal_mask) or torch.any(~legal_mask.any(-1)) or
            torch.any(~(legal_mask & ~fireball_mask).any(-1))):
        raise ValueError('Invalid Fireball selectivity support')
    # logsumexp difference is stable even when the initial policy strongly
    # prefers a Fireball tile. WAIT remains mechanically legal.
    all_legal=torch.logsumexp(logits.masked_fill(~legal_mask,float('-inf')),dim=-1)
    non_fireball=torch.logsumexp(logits.masked_fill(~(legal_mask & ~fireball_mask),float('-inf')),dim=-1)
    return (all_legal-non_fireball).mean()
