"""Shuffled full-game train cycles with an explicit hard-family curriculum."""
import hashlib
import json
import random


class ContinuousScenarioCycle:
    def __init__(self,scenarios,seed=42,initial_hand_mode='shuffle',focus='uniform'):
        self.scenarios=tuple(scenarios)
        if focus not in ('uniform','air_golem'):raise ValueError('Unknown family curriculum')
        if not self.scenarios or any(s.split!='train'for s in self.scenarios):raise ValueError('Training cases only')
        if initial_hand_mode!='shuffle':raise ValueError('Full games require shuffled hands')
        ids=[s.scenario_id for s in self.scenarios]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate scenario IDs')
        self.focus=focus;self.seed=seed;self.cycle=0;self.offset=0;self.initial_hand_mode=initial_hand_mode
        catalog=[s.to_dict()for s in self.scenarios]
        self.catalog_sha256=hashlib.sha256(json.dumps(dict(cases=catalog,focus=focus),sort_keys=True).encode()).hexdigest()
        self._prepare()

    def _prepare(self):
        self.order=[]
        for s in self.scenarios:
            weight=3 if self.focus=='air_golem'and s.family in ('air','golem')else 1
            self.order.extend([s.scenario_id]*weight)
        random.Random(f'{self.seed}:continuous:{self.cycle}:{self.focus}').shuffle(self.order)

    def next(self):
        if self.offset==len(self.order):self.cycle+=1;self.offset=0;self._prepare()
        # Repeated hard cases get fresh layouts/hands, not cloned same-seed states.
        result=dict(scenario_id=self.order[self.offset],seed=self.seed+self.cycle*len(self.order)+self.offset)
        self.offset+=1
        return result

    def state_dict(self):
        return dict(schema='continuous-family-cycle-v1',focus=self.focus,seed=self.seed,cycle=self.cycle,
                    offset=self.offset,catalog_sha256=self.catalog_sha256)

    def load_state_dict(self,state):
        if (state.get('schema')!='continuous-family-cycle-v1'or
                any(state.get(k)!=getattr(self,k)for k in('focus','seed','catalog_sha256'))):
            raise ValueError('Sampler source/curriculum changed')
        if type(state['cycle'])is not int or state['cycle']<0 or type(state['offset'])is not int:
            raise ValueError('Invalid cycle cursor')
        self.cycle=state['cycle'];self._prepare()
        if not 0<=state['offset']<=len(self.order):raise ValueError('Invalid cycle offset')
        self.offset=state['offset']
