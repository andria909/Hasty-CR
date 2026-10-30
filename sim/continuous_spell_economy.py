"""Optional crown-only spell ablation; default training factories unchanged."""
import math

from sim.continuous_spell_waste import SpellWasteContinuousPushEnv
from sim.continuous_tower_support import TowerSupportContinuousPushEnv
from sim.spell_economy import SpellEconomyTracker


class SpellEconomyContinuousPushEnv(SpellWasteContinuousPushEnv):
    def __init__(self, *args, crown_only_weight=0., dormant_king_weight=0., **kwargs):
        for value in (crown_only_weight, dormant_king_weight):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError('Finite nonnegative spell economy coefficient required')
        self.crown_only_weight = float(crown_only_weight)
        self.dormant_king_weight = float(dormant_king_weight)
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self.spell_waste = None
        # Attach exactly one paid play-log observer, replacing the parent one.
        observation, _ = TowerSupportContinuousPushEnv.reset(self, *args, **kwargs)
        self.spell_waste = SpellEconomyTracker(self.match).attach()
        self._previous = self._metrics()
        return observation, self._info()

    def _economy_penalty(self):
        if not self.spell_waste:
            return 0.
        metrics = self.spell_waste.metrics()
        dormant = metrics['dormant_at_request_king_only_elixir']
        other = metrics['nonlethal_crown_only_elixir']-dormant
        # Disjoint bins: no double penalty for a single king-only Fireball.
        return self.crown_only_weight*other+self.dormant_king_weight*dormant

    def step(self, action):
        before = self._economy_penalty()
        obs, reward, terminated, truncated, info = super().step(action)
        penalty = self._economy_penalty()-before
        info['spell_economy_step_penalty'] = penalty
        return obs, reward-penalty, terminated, truncated, info

    def _metrics(self):
        metrics = super()._metrics()
        metrics.update(crown_only_weight=self.crown_only_weight,
                       dormant_king_weight=self.dormant_king_weight,
                       spell_economy_penalty=self._economy_penalty())
        return metrics

    def restore(self, state):
        if (state['crown_only_weight'] != self.crown_only_weight or
                state['dormant_king_weight'] != self.dormant_king_weight):
            raise ValueError('Spell economy coefficient differs')
        super().restore(state)
