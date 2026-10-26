"""Two public frames plus their age; independent of simulator private state.

Only the current frame remains in the original channels. Additional channels
can be zero-initialized through load_policy_state, preserving old policies.
No running training job imports this experimental adapter.
"""
from copy import deepcopy
import numpy as np

from sim.env import NUM_PLANES

CONTRACT='joint_public_presence_continuous_slots_history2_v1'
HISTORY_PLANES=2*NUM_PLANES+1


class PublicFrameHistory:
    def __init__(self,max_gap_ms=1000):
        if type(max_gap_ms)is not int or max_gap_ms<=0:raise ValueError('Positive history gap required')
        self.max_gap_ms=max_gap_ms;self.reset()

    def reset(self):
        self.world=None;self.current=None;self.previous=None
        self.current_ms=None;self.previous_ms=None

    def observe(self,observation,*,at_ms,world):
        planes=observation['planes'];scalars=observation['scalars']
        if (type(at_ms)is not int or at_ms<0 or planes.ndim!=3 or planes.shape[0]!=NUM_PLANES
                or not np.isfinite(planes).all() or not np.isfinite(scalars).all()):
            raise ValueError('Finite public frame and nonnegative integer clock required')
        if (self.current is None or world!=self.world or at_ms<self.current_ms
                or at_ms-self.current_ms>self.max_gap_ms):
            self.reset();self.world=deepcopy(world)
        elif at_ms>self.current_ms:
            self.previous=self.current.copy();self.previous_ms=self.current_ms
        # Reading a frame twice must not turn the current frame into history.
        self.current=planes.copy();self.current_ms=at_ms
        previous=np.zeros_like(planes)if self.previous is None else self.previous
        age=0. if self.previous is None else (at_ms-self.previous_ms)/self.max_gap_ms
        age_plane=np.full((1,*planes.shape[1:]),age,dtype=planes.dtype)
        return dict(observation,planes=np.concatenate((planes,previous,age_plane)),scalars=scalars.copy())

    def snapshot(self):return deepcopy(self.__dict__)

    def restore(self,state):
        if state.get('max_gap_ms')!=self.max_gap_ms:raise ValueError('History timing contract differs')
        self.__dict__=deepcopy(state)
