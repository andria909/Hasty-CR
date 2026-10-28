"""Optional crown-tower assistance reward; public observations stay unchanged."""
import math

from sim.continuous_push import ContinuousPushEnv
from sim.tower_support import TowerSupport


class TowerSupportContinuousPushEnv(ContinuousPushEnv):
    def __init__(self, *args, tower_support_weight=0., **kwargs):
        if isinstance(tower_support_weight, bool) or not math.isfinite(tower_support_weight) or tower_support_weight < 0:
            raise ValueError('Tower support weight must be finite and nonnegative')
        self.tower_support_weight = float(tower_support_weight)
        self.tower_support = None
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self.tower_support = None
        observation, _ = super().reset(*args, **kwargs)
        self.tower_support = TowerSupport(self.match.battle).attach()
        self._previous = self._metrics()
        return observation, self._info()

    def step(self, action):
        before = self._credited_value()
        observation, reward, terminated, truncated, info = super().step(action)
        bonus = self.tower_support_weight * (self._credited_value() - before)
        info['tower_support_step_reward'] = bonus
        return observation, reward + bonus, terminated, truncated, info

    def _credited_value(self):
        if not self.tower_support:return 0.
        return self.tower_support.total_paid_value if self.profile['tower_damage_all']else self.tower_support.assisted_value

    def _metrics(self):
        metrics = super()._metrics()
        value = self._credited_value()
        metrics.update(tower_support_paid_value=value,
                       tower_support_reward=self.tower_support_weight * value,
                       tower_support_weight=self.tower_support_weight,
                       tower_damage_reward_mode='all_damage'if self.profile['tower_damage_all']else 'assisted',
                       own_tower_damage=self.tower_support.total_damage if self.tower_support else 0.)
        return metrics

    def restore(self, state):
        if state['tower_support_weight'] != self.tower_support_weight:
            raise ValueError('Tower support weight differs')
        super().restore(state)
