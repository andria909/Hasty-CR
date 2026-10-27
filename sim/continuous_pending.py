"""Opt-in queue-aware observations; combat and actions stay in the base env."""
from sim.continuous_push import ContinuousPushEnv
from sim.pending_inputs import append_pending,CONTRACT


class PendingContinuousPushEnv(ContinuousPushEnv):
    def _observe(self):
        return append_pending(super()._observe(),self.pending,self.match.elapsed_ms,reflected=self.reflected)

    def _metrics(self):
        return dict(super()._metrics(),observation_contract=CONTRACT)
