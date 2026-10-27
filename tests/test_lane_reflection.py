import numpy as np
from sim.env import ACTIONS,NUM_SCALARS
from sim.lane_reflection import reflect_actions,reflect_observation
from sim.continuous_push import ContinuousPushEnv


def test_reflection_swaps_spatial_actions_and_lane_features_and_is_involutive():
    obs=dict(planes=np.arange(20*32*18,dtype=np.float32).reshape(20,32,18),scalars=np.arange(NUM_SCALARS+3,dtype=np.float32))
    reflected=reflect_observation(obs)
    for left,right in ((3,4),(5,6),(NUM_SCALARS-2,NUM_SCALARS-1)):
        assert reflected['scalars'][left]==obs['scalars'][right]
    np.testing.assert_array_equal(reflected['scalars'][-3:],obs['scalars'][-3:])
    for key,value in reflect_observation(reflected).items():np.testing.assert_array_equal(value,obs[key])
    indices=np.arange(ACTIONS)
    np.testing.assert_array_equal(reflect_actions(reflect_actions(indices)),indices)
    assert reflect_actions(indices)[1+2*576+20*18+3]==1+2*576+20*18+14
    np.testing.assert_array_equal(reflect_actions(indices)[2305:],indices[2305:])


def test_mirrored_mask_action_and_command_share_one_coordinate_transform():
    e=ContinuousPushEnv();e.reset(seed=2);e.next_enemy=10**9
    original=e._mask();obs=e._observe();e.reflected=True
    np.testing.assert_array_equal(e._mask(),reflect_actions(original))
    np.testing.assert_array_equal(e._observe()['planes'],reflect_observation(obs)['planes'])
    action=int(np.flatnonzero(e._mask()[1:])[0]+1);x=(action-1)%18
    e.step(action)
    command=e.pending[0]if e.pending else None
    assert command is not None and command[3]==17-x


def test_lane_reflection_is_disabled_on_validation_and_test():
    for split in ('validation','test'):
        e=ContinuousPushEnv(split=split,continuous_profile={'lane_reflection_fraction':1.});e.reset(seed=2)
        assert not e.reflected
