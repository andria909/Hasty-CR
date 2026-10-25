"""Paired HP discrimination and public full-match rehearsal for Fireball v3."""
import torch
import torch.nn.functional as F

from sim.fireball_decision_pairs import selectivity_loss


def paired_loss(positive_logits,negative_logits,legal,fireball,reference,margin=2.):
    if torch.any(~fireball.any(-1)) or torch.any(~(legal & ~fireball).any(-1)):
        raise ValueError('Pairs require legal Fireball and non-Fireball choices')
    positive_ce=F.cross_entropy(positive_logits.masked_fill(~legal,-1e9),reference)
    negative_mass=selectivity_loss(negative_logits,legal,fireball)
    def odds(logits):
        return (torch.logsumexp(logits.masked_fill(~fireball,float('-inf')),-1)-
                torch.logsumexp(logits.masked_fill(~(legal & ~fireball),float('-inf')),-1))
    contrast=F.softplus(margin-(odds(positive_logits)-odds(negative_logits))).mean()
    return positive_ce,negative_mass,contrast


def skill_eligibility(positive,negative,agreement,teacher_negative,protocol):
    rates={k:v['successes']/v['episodes']for k,v in positive['by_kind'].items()}
    if set(rates)!=set(protocol['skill_floors']):raise ValueError('Incomplete skill set')
    return (all(rates[k]>=threshold for k,threshold in protocol['skill_floors'].items())and
            agreement>=protocol['rehearsal_action_agreement_floor']and
            negative['first_action_fireball_rate']<=teacher_negative['first_action_fireball_rate']+
                protocol['negative_cast_rate_margin_over_teacher'])


def deployed_rehearsal(teacher,device,count,protocol):
    """Original policy, actual public inputs/masks, training-only match seeds."""
    import numpy as np
    from sim import arena
    from sim.match import Match
    from sim.env import DECK_26
    from sim.defense_assets import load_defense_assets
    from sim.runner import resolve_deck
    from sim.eval_defense_duel import policy_observation,mechanical_mask,JOINT_MODE
    from sim.adapter import grid_to_point
    from sim.selfplay import decode_for
    from sim.train_ppo import masked_distribution
    from sim.train_defense_ppo import tensor_observations
    cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,DECK_26))
    observations=[];masks=[];probabilities=[];contexts=[];match_index=0
    teacher.eval()
    with torch.no_grad():
        while len(observations)<count:
            seed=protocol['rehearsal_match_seed_start']+match_index;match_index+=1
            match=Match(cards=cards,decks=(list(DECK_26),list(DECK_26)),spells=spells,
                        seed=seed,geometry=protocol['geometry'])
            pending={1:None,-1:None};next_decision=0
            while not match.finished and match.elapsed_ms<protocol['match_max_ms']and len(observations)<count:
                if match.elapsed_ms>=next_decision:
                    for side in (1,-1):
                        mask=mechanical_mask(match,cards,spells,side,pending[side]is not None)
                        obs=policy_observation(match,side,JOINT_MODE)
                        logits,_=teacher(*tensor_observations([obs],device))
                        dist=masked_distribution(logits,torch.as_tensor(mask[None],device=device))
                        if mask.sum()>1 and len(observations)<count:
                            observations.append(obs);masks.append(mask);probabilities.append(dist.probs[0].cpu())
                            contexts.append(dict(seed=seed,side=side,elapsed_ms=match.elapsed_ms))
                        decoded=decode_for(int(dist.probs.argmax()))
                        if decoded is not None:
                            slot,x,y=decoded
                            pending[side]=(match.elapsed_ms+protocol['match_delay_ms'],match.players[side].hand[slot],x,y)
                    next_decision+=500
                for side in (1,-1):
                    command=pending[side]
                    if command is not None and command[0]<=match.elapsed_ms:
                        _,card,x,y=command;match.play_card(side,card,grid_to_point(x,y,side));pending[side]=None
                match.step(arena.TICK_MS)
    return observations,np.asarray(masks),torch.stack(probabilities).to(device),contexts
