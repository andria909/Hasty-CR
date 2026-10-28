"""Optional paid empty-spell penalty, compatible with tower assistance."""
import math
from sim.continuous_tower_support import TowerSupportContinuousPushEnv
from sim.spell_waste import SpellWasteTracker


class SpellWasteContinuousPushEnv(TowerSupportContinuousPushEnv):
    def __init__(self,*args,spell_waste_weight=0.,**kwargs):
        if isinstance(spell_waste_weight,bool) or not math.isfinite(spell_waste_weight) or spell_waste_weight<0:
            raise ValueError('Finite nonnegative empty-spell coefficient required')
        self.spell_waste_weight=float(spell_waste_weight);self.spell_waste=None
        super().__init__(*args,**kwargs)
        # This variant replaces, rather than adds to, the old Fireball-only miss penalty.
        self.profile['missed_fireball_penalty']=0.

    def reset(self,*args,**kwargs):
        self.spell_waste=None
        observation,_=super().reset(*args,**kwargs)
        self.spell_waste=SpellWasteTracker(self.match).attach()
        self._previous=self._metrics()
        return observation,self._info()

    def step(self,action):
        before=self.spell_waste.metrics()['empty_spell_elixir'] if self.spell_waste else 0.
        observation,reward,terminated,truncated,info=super().step(action)
        wasted=self.spell_waste.metrics()['empty_spell_elixir']-before
        penalty=self.spell_waste_weight*wasted
        info['empty_spell_step_penalty']=penalty
        return observation,reward-penalty,terminated,truncated,info

    def _metrics(self):
        metrics=super()._metrics()
        if self.spell_waste:
            metrics.update(self.spell_waste.metrics())
        else:
            metrics.update(empty_spell_elixir=0.,empty_spell_counts={},resolved_spell_counts={},
                           empty_spell_elixir_by_card={},unresolved_spell_counts={})
        metrics.update(spell_waste_weight=self.spell_waste_weight,
                       empty_spell_penalty=self.spell_waste_weight*metrics['empty_spell_elixir'])
        return metrics

    def restore(self,state):
        if state['spell_waste_weight']!=self.spell_waste_weight:
            raise ValueError('Empty-spell coefficient differs')
        super().restore(state)
