"""Left/right data augmentation for the public continuous observation."""
import numpy as np
from sim.env import ACTIONS,NUM_SCALARS


def reflect_actions(values):
    if values.shape != (ACTIONS,):
        raise ValueError('Unexpected action shape')
    out=values.copy()
    out[1:2305]=values[1:2305].reshape(4,32,18)[:,:,::-1].reshape(-1)
    return out


def reflect_observation(observation):
    planes=observation['planes'];scalars=observation['scalars']
    if planes.shape != (20,32,18) or scalars.shape != (NUM_SCALARS+3,):
        raise ValueError('Continuous public20/309 contract required')
    scalars=scalars.copy()
    for left,right in ((3,4),(5,6),(NUM_SCALARS-2,NUM_SCALARS-1)):
        scalars[left],scalars[right]=scalars[right],scalars[left]
    return dict(planes=np.ascontiguousarray(planes[:,:,::-1]),scalars=scalars)
