"""One-policy joint attack/defense curriculum with an explicit task feature.

These are mixed skill episodes, not a claim of full-match strategic acceptance.
"""
from copy import deepcopy
from dataclasses import dataclass,replace
import math
import random

import numpy as np

from sim import arena
from sim.attack_env import AttackEnv
from sim.defense_env import DefenseEnv

OBSERVATION_CONTRACT='joint_attack_defense_timer_task_v1'


@dataclass(frozen=True)
class StrategyScenario:
    task: str
    scenario: object

    @property
    def scenario_id(self):return self.task+'::'+self.scenario.scenario_id
    @property
    def family_id(self):return self.task+'::'+self.scenario.family_id
    @property
    def split(self):return self.scenario.split
    def to_dict(self):return dict(task=self.task,scenario=self.scenario.to_dict())


class StrategyEnv:
    def __init__(self,seed=0,split='train',action_delay_ms=150,max_ms=240000,
                 geometry='legacy_v1',coordinate_mode='physical_live_cells_v2',
                 initial_hand_mode='shuffle',deadline_scale=3.,visible_fraction=.5):
        if not math.isfinite(deadline_scale)or not 0<deadline_scale<=5:
            raise ValueError('Invalid bounded joint deadline scale')
        if initial_hand_mode!='shuffle':
            raise ValueError('Joint curriculum requires actual shuffled hands for both tasks')
        self.seed=int(seed);self.split=split
        self.geometry=arena.resolve_geometry(geometry)
        self.coordinate_mode=coordinate_mode
        self.initial_hand_mode=initial_hand_mode
        self.deadline_scale=float(deadline_scale)
        self._attack=AttackEnv(seed=seed,split=split,action_delay_ms=action_delay_ms,
            max_ms=max_ms,geometry=geometry,coordinate_mode=coordinate_mode,
            initial_hand_mode=initial_hand_mode,visible_fraction=visible_fraction)
        self._attack.scenarios=[replace(s,deadline_ms_range=tuple(round(n*deadline_scale)
            for n in s.deadline_ms_range))for s in self._attack.scenarios]
        self._defense=DefenseEnv(seed=seed,split=split,action_delay_ms=action_delay_ms,
            max_ms=max_ms,geometry=geometry,coordinate_mode=coordinate_mode)
        self.scenarios=([StrategyScenario('attack',s)for s in self._attack.scenarios]+
                        [StrategyScenario('defense',s)for s in self._defense.scenarios])
        self._active=None;self.scenario=None

    @property
    def match(self):return self._active.match if self._active is not None else None

    def _convert(self,observation,info):
        observation=dict(observation)
        scalars=observation['scalars']
        if self.scenario.task=='defense':
            scalars=np.concatenate((scalars,np.zeros(2,dtype=np.float32)))
        observation['scalars']=np.concatenate((scalars,np.array(
            [float(self.scenario.task=='attack')],dtype=np.float32)))
        info=dict(info);info['metrics']=dict(info['metrics'],task=self.scenario.task,
            strategy_scenario_id=self.scenario.scenario_id,
            observation_contract=OBSERVATION_CONTRACT,deadline_scale=self.deadline_scale)
        return observation,info

    def reset(self,seed=None,scenario_id=None):
        self.seed=self.seed+1 if seed is None else int(seed)
        cases=[s for s in self.scenarios if scenario_id is None or s.scenario_id==scenario_id]
        if not cases:raise ValueError('Joint scenario outside requested split/filter')
        self.scenario=random.Random(self.seed).choice(cases)
        self._active=self._attack if self.scenario.task=='attack'else self._defense
        obs,info=self._active.reset(seed=self.seed,scenario_id=self.scenario.scenario.scenario_id)
        return self._convert(obs,info)

    def step(self,action):
        if self._active is None:raise RuntimeError('Call reset first')
        obs,reward,terminated,truncated,info=self._active.step(action)
        obs,info=self._convert(obs,info)
        return obs,reward,terminated,truncated,info

    def snapshot(self):return deepcopy(self.__dict__)

    def restore(self,state):
        for key in ('geometry','coordinate_mode','initial_hand_mode','deadline_scale'):
            if state[key]!=getattr(self,key):raise ValueError('Joint snapshot contract mismatch: '+key)
        self.__dict__=deepcopy(state)
        return self._convert(self._active._observe(),self._active._info())
