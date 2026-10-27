import numpy as np
import pytest
from sim.train_attack_ppo import make_training_env,observation_contract
from sim.pending_inputs import CONTRACT


def test_pending_marker_tracks_actual_delivery_and_preserves_base_behavior():
    base=make_training_env(curriculum='continuous_v1',action_delay_ms=1350)
    new=make_training_env(curriculum='continuous_v1',action_delay_ms=1350,pending_inputs=True)
    a,_=base.reset(seed=3800009);b,_=new.reset(seed=3800009)
    assert b['planes'].shape==(24,32,18) and b['scalars'].shape==(313,)
    action=int(np.flatnonzero(base._mask()[1:])[0]+1)
    for step in range(8):
        a,ra,da,ta,ia=base.step(action if step==0 else 0)
        b,rb,db,tb,ib=new.step(action if step==0 else 0)
        assert np.array_equal(a['planes'],b['planes'][:20])
        assert np.array_equal(a['scalars'],b['scalars'][:309])
        assert np.array_equal(base._mask(),new._mask())
        assert ra==rb and da==db and ta==tb
        assert b['planes'][20:].sum()==(1 if step<5 else 0)
    assert base.actions==new.actions
    assert observation_contract('joint','continuous_v1',pending_inputs=True)==CONTRACT


def test_pending_history_combination_is_rejected():
    with pytest.raises(ValueError):make_training_env(curriculum='continuous_v1',history_frames=2,pending_inputs=True)
