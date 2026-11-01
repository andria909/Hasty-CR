"""Optional card/WAIT entropy regularization without changing action probabilities."""
import math

import torch


def entropy_components(distribution):
    probabilities=distribution.probs
    if probabilities.ndim!=2 or probabilities.shape[1]<2305:
        raise ValueError('Expected WAIT, four 576-cell slots, and optional abilities')
    groups=torch.cat((probabilities[:,:1],probabilities[:,1:2305].reshape(-1,4,576).sum(-1),
                      probabilities[:,2305:]),dim=-1)
    full=distribution.entropy()
    group=torch.distributions.Categorical(probs=groups).entropy()
    return full,group


def weighted_entropy(distribution,spatial_weight=1.):
    if isinstance(spatial_weight,bool)or not math.isfinite(spatial_weight)or not 0<=spatial_weight<=1:
        raise ValueError('Spatial entropy weight must be finite in [0,1]')
    # Preserve the original objective and gradients exactly for the control.
    if spatial_weight==1.:return distribution.entropy()
    full,group=entropy_components(distribution)
    # H(action)=H(card/WAIT)+E_card[H(cell|card)]. Abilities are singleton groups.
    return spatial_weight*full+(1-spatial_weight)*group
