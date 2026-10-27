"""Single-decision hand-slot permutation; preserve card/cell and WAIT semantics."""
import numpy as np
from sim.env import ACTIONS, CARD_ACTIONS


def permute(scalars, actions, permutation):
    permutation = np.asarray(permutation)
    if permutation.shape != (4,) or sorted(permutation.tolist()) != [0,1,2,3]:
        raise ValueError('A permutation of four hand slots is required')
    scalars, actions = np.asarray(scalars), np.asarray(actions)
    if scalars.shape != (309,) or actions.shape != (ACTIONS,):
        raise ValueError('Public309/action contract required')
    result_scalars, result_actions = scalars.copy(), actions.copy()
    result_scalars[7:39] = scalars[7:39].reshape(4,8)[permutation].reshape(-1)
    result_actions[1:CARD_ACTIONS] = actions[1:CARD_ACTIONS].reshape(4,576)[permutation].reshape(-1)
    return result_scalars, result_actions
