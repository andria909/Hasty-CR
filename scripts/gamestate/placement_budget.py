"""Conservative spendable balance while independent slots await confirmation."""


class PlacementBudget:
    def __init__(self):
        self.value=None
        self.tick=None

    def observe(self,tick,native):
        if self.tick is None:self.value=native
        elif tick<self.tick:raise ValueError('Budget clock moved backwards')
        else:
            # Single-elixir income is a conservative lower bound even when
            # double/triple income is active. Never exceed fresh native mana.
            income=(tick-self.tick)*.05/2.8
            self.value=min(native,10.,self.value+income)
        self.tick=tick
        return self.value

    def spend(self,cost):
        if self.value is None or cost<0 or cost>self.value+1e-6:raise ValueError('Unreserved or unaffordable command')
        self.value=max(0.,self.value-cost)

    def reconcile(self,native,*,all_confirmed):
        if all_confirmed:self.value=native
