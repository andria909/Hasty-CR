"""Balanced mixed train passes: all defense cases plus resampled attack cases."""
import hashlib
import json
import random


class StrategyScenarioCycle:
    def __init__(self,scenarios,seed=42,initial_hand_mode='shuffle'):
        self.scenarios=tuple(scenarios)
        if not self.scenarios or any(s.split!='train'for s in self.scenarios):
            raise ValueError('Joint sampler accepts train cases only')
        if initial_hand_mode!='shuffle':raise ValueError('Joint sampler requires shuffled hands')
        ids=[s.scenario_id for s in self.scenarios]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate joint scenario IDs')
        self.groups={task:sorted(s.scenario_id for s in self.scenarios if s.task==task)
                     for task in ('attack','defense')}
        if any(not group for group in self.groups.values()):raise ValueError('Both training tasks required')
        self.catalog_sha256=hashlib.sha256(json.dumps([s.to_dict()for s in self.scenarios],
            sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.seed=int(seed);self.cycle=0;self.offset=0;self.initial_hand_mode=initial_hand_mode
        self._prepare()

    def _prepare(self):
        rng=random.Random(f'{self.seed}:joint-cycle:{self.cycle}')
        count=max(map(len,self.groups.values()));self.order=[]
        for group in self.groups.values():
            expanded=[]
            while len(expanded)<count:
                chunk=list(group);rng.shuffle(chunk);expanded+=chunk
            self.order+=expanded[:count]
        rng.shuffle(self.order)

    def next(self):
        if self.offset==len(self.order):
            self.cycle+=1;self.offset=0;self._prepare()
        result=dict(scenario_id=self.order[self.offset],seed=self.seed+self.cycle)
        self.offset+=1
        return result

    def state_dict(self):
        return dict(schema='joint-balanced-scenario-cycle-v1',seed=self.seed,cycle=self.cycle,
            offset=self.offset,catalog_sha256=self.catalog_sha256,initial_hand_mode=self.initial_hand_mode)

    def load_state_dict(self,state):
        if (state.get('schema')!='joint-balanced-scenario-cycle-v1' or
                any(state.get(key)!=getattr(self,key)for key in ('seed','catalog_sha256','initial_hand_mode'))):
            raise ValueError('Joint sampler contract mismatch')
        cycle,offset=state['cycle'],state['offset']
        if (not isinstance(cycle,int)or isinstance(cycle,bool)or cycle<0 or
                not isinstance(offset,int)or isinstance(offset,bool)or not 0<=offset<=len(self.order)):
            raise ValueError('Invalid joint sampler cursor')
        self.cycle=cycle;self.offset=offset;self._prepare()
