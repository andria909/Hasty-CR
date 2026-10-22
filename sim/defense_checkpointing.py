"""Checkpoint cadence without changing optimizer or rollout boundaries."""


def checkpoint_due(completed,every=1,*,final=False,budget_stop=False):
    if not isinstance(every,int) or isinstance(every,bool) or every<1:
        raise ValueError('checkpoint-every must be a positive integer')
    if completed<0:
        raise ValueError('completed updates cannot be negative')
    return bool(final or budget_stop or (completed>0 and completed%every==0))
