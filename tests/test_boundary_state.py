from copy import deepcopy
import random
from types import SimpleNamespace

from scripts.attack.boundary_state import equal_boundary


def state():
    x=SimpleNamespace(rng=random.Random(34))
    x.self=x
    return dict(first=x,second=x)


def test_cycles_and_same_state_random_generators_are_compared_by_content():
    a=state();b=deepcopy(a)
    assert equal_boundary(a,b)
    b['first'].rng.random()
    assert not equal_boundary(a,b)


def test_changed_shared_reference_is_not_equal_even_with_equal_fields():
    a=state();b=deepcopy(a)
    b['second']=deepcopy(b['first'])
    assert not equal_boundary(a,b)
