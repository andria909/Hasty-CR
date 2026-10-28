"""Shared Cannon-only central defence restriction in own-relative 18x32 cells."""
import numpy as np

CENTRAL_MODE='central_defense_v1'
MODES=('unrestricted',CENTRAL_MODE)
CENTRAL_CELLS=tuple((x,y)for y in range(18,24)for x in range(6,12))


def restrict_cannon(mask,hand,mode):
    if mode not in MODES:raise ValueError('Unknown Cannon placement contract')
    if mode=='unrestricted':return mask
    if len(hand)!=4 or mask.ndim!=1 or len(mask)<2305 or mask.dtype!=np.bool_:
        raise ValueError('Complete four-slot boolean mask required')
    result=mask.copy();allowed=np.zeros((32,18),dtype=bool)
    for x,y in CENTRAL_CELLS:allowed[y,x]=True
    for slot,card in enumerate(hand):
        if card=='cannon':result[1+slot*576:1+(slot+1)*576]&=allowed.reshape(-1)
    return result
