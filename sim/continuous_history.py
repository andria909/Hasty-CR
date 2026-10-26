"""Opt-in public-frame history for full-game PPO and its validation."""
from sim.continuous_push import ContinuousPushEnv
from sim.public_frame_history import PublicFrameHistory,CONTRACT


class HistoryContinuousPushEnv(ContinuousPushEnv):
    def __init__(self,*args,**kwargs):
        self.public_history=PublicFrameHistory()
        super().__init__(*args,**kwargs)

    def reset(self,*args,**kwargs):
        self.public_history.reset()
        return super().reset(*args,**kwargs)

    def _observe(self):
        return self.public_history.observe(super()._observe(),at_ms=self.match.elapsed_ms,world=self.match.seed)

    def _metrics(self):
        return dict(super()._metrics(),observation_contract=CONTRACT)
