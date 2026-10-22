"""Shared paired curriculum: one hand-shuffle seed per complete scenario pass."""
from __future__ import annotations

import random


class ScenarioCycle:
    def __init__(self, scenario_ids, seed=42):
        self.scenario_ids = tuple(scenario_ids)
        if not self.scenario_ids or len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError('Scenario IDs must be nonempty and unique')
        if any(not isinstance(s, str) or not s for s in self.scenario_ids):
            raise ValueError('Scenario IDs must be nonempty strings')
        self.seed = int(seed)
        self.cycle = 0
        self.offset = 0
        self._prepare_order()

    def _prepare_order(self):
        self.order = list(self.scenario_ids)
        random.Random(f'{self.seed}:scenario-cycle:{self.cycle}').shuffle(self.order)

    def next(self):
        if self.offset == len(self.order):
            self.cycle += 1
            self.offset = 0
            self._prepare_order()
        scenario = self.order[self.offset]
        self.offset += 1
        return dict(scenario_id=scenario, seed=self.seed + self.cycle)

    def state_dict(self):
        return dict(schema='scenario-cycle-v1', scenario_ids=list(self.scenario_ids),
                    seed=self.seed, cycle=self.cycle, offset=self.offset)

    def load_state_dict(self, state):
        if (state.get('schema') != 'scenario-cycle-v1'
                or tuple(state['scenario_ids']) != self.scenario_ids
                or state['seed'] != self.seed):
            raise ValueError('Sampler configuration mismatch')
        cycle, offset = state['cycle'], state['offset']
        if (not isinstance(cycle, int) or cycle < 0 or not isinstance(offset, int)
                or not 0 <= offset <= len(self.scenario_ids)):
            raise ValueError('Invalid sampler cursor')
        self.cycle, self.offset = cycle, offset
        self._prepare_order()
