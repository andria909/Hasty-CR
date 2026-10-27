import numpy as np
import pytest
from scripts.attack.sequence_sampling import sequence_probabilities


def test_duplicate_wait_frames_do_not_increase_sequence_or_wait_mass():
    def rows(n):
        actions=[3]+[0]*n+[4,5]
        return [dict(array_index=i,accepted=True,seed=1,context=0 if i<=n else 1,action=a) for i,a in enumerate(actions)]
    for n in (1,100):
        r=rows(n);p=sequence_probabilities(r,range(len(r)),.25)
        assert np.isclose(p[:n+1].sum(),.5)
        assert np.isclose(p[1:n+1].sum(),.125)
        assert np.isclose(p[0],.375)
        assert np.isclose(p[n+1:].sum(),.5)


def test_holdout_never_enters_probability_vector():
    r=[dict(array_index=i,accepted=True,seed=i,context=0,action=1) for i in range(3)]
    assert np.allclose(sequence_probabilities(r,[0,1]),[.5,.5])
    r[2]['action']=0
    assert np.allclose(sequence_probabilities(r,[0,1]),[.5,.5])
    with pytest.raises(ValueError,match='without a deployment'):
        sequence_probabilities(r,[0,1,2])
