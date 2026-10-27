import numpy as np
import pytest
from sim.pending_inputs import append_pending


def base():return dict(planes=np.zeros((20,32,18),np.float32),scalars=np.zeros(309,np.float32))


def test_queued_position_and_time_are_visible_without_mutating_base():
    original=base();original['scalars'][0]=.6
    a=append_pending(original,[(3350,'musketeer',0,1,18)],2250)
    b=append_pending(original,[(3350,'musketeer',0,8,28)],2250)
    assert a['planes'].shape==(24,32,18) and a['scalars'].shape==(313,)
    assert a['planes'][20,18,1]==1 and b['planes'][20,28,8]==1
    assert not np.array_equal(a['planes'],b['planes'])
    assert a['scalars'][309]==pytest.approx(1100/3000)
    assert np.array_equal(a['planes'][:20],original['planes'])
    assert np.array_equal(a['scalars'][:309],original['scalars'])
    assert original['planes'].sum()==0


def test_reflection_and_expired_command_presence():
    x=append_pending(base(),[(1000,'cannon',2,6,21)],1000,reflected=True)
    assert x['planes'][22,21,11]==1 and x['scalars'][311]==0
    assert append_pending(base(),[],0)['planes'][20:].sum()==0


def test_duplicate_slot_is_rejected():
    with pytest.raises(ValueError):
        append_pending(base(),[(1000,'cannon',2,6,21),(1250,'musketeer',2,3,23)],0)


def test_migration_rejects_unidentified_checkpoint():
    from sim.pending_inputs import initialize_from_base
    with pytest.raises(ValueError,match='contract'):
        initialize_from_base(dict(config={},state_dict={}))
