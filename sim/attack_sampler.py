"""Train-only paired attack shuffle cycles; no training or environment mutation."""
import hashlib
import json

from sim.defense_sampler import ScenarioCycle


class AttackScenarioCycle:
    """Shuffle all supplied train cases once, sharing one hand seed per pass.

    No fixed ordering by goal type/difficulty. Hidden/visible siblings receive
    the same shuffle seed and their catalog context shares the private deadline.
    Callers must construct AttackEnv with the recorded initial_hand_mode.
    """
    def __init__(self,scenarios,seed=42,initial_hand_mode='shuffle'):
        scenarios=tuple(scenarios)
        if not scenarios or any(s.split!='train'for s in scenarios):
            raise ValueError('Attack sampler accepts train scenarios only')
        if initial_hand_mode not in ('scenario','shuffle'):
            raise ValueError('Unknown attack initial hand mode')
        self.initial_hand_mode=initial_hand_mode
        self._cycle=ScenarioCycle([s.scenario_id for s in scenarios],seed=seed)
        definitions=[s.to_dict() if hasattr(s,'to_dict') else vars(s)for s in scenarios]
        self.catalog_sha256=hashlib.sha256(json.dumps(definitions,sort_keys=True,
                                                     separators=(',',':')).encode()).hexdigest()

    def next(self):return self._cycle.next()

    def state_dict(self):
        return dict(schema='attack-scenario-cycle-v1',initial_hand_mode=self.initial_hand_mode,
                    catalog_sha256=self.catalog_sha256,sampler=self._cycle.state_dict())

    def load_state_dict(self,state):
        if (state.get('schema')!='attack-scenario-cycle-v1' or
                state.get('initial_hand_mode')!=self.initial_hand_mode or
                state.get('catalog_sha256')!=self.catalog_sha256):
            raise ValueError('Attack sampler checkpoint contract mismatch')
        self._cycle.load_state_dict(state['sampler'])
