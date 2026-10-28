"""Permute card slots and their own queued-command features together."""
import numpy as np
from sim.hand_slot_augmentation import permute


def permute_pending(observation,actions,order):
    planes,scalars=np.asarray(observation['planes']),np.asarray(observation['scalars'])
    if planes.shape!=(24,32,18) or scalars.shape!=(313,):
        raise ValueError('Pending24/313 observation required')
    base,mapped=permute(scalars[:309],actions,order)
    order=np.asarray(order)
    changed=dict(observation,planes=np.concatenate((planes[:20],planes[20:][order])),
                 scalars=np.concatenate((base,scalars[309:][order])))
    return changed,mapped
