import numpy as np
import pytest

from sim.continuous_push import ContinuousPushEnv
from sim.continuous_tower_support import TowerSupportContinuousPushEnv


def test_optional_reward_preserves_observations_physics_and_resume():
    base = ContinuousPushEnv(seed=17, action_delay_ms=1350)
    candidate = TowerSupportContinuousPushEnv(seed=17, action_delay_ms=1350,
                                              tower_support_weight=.1)
    base.reset(seed=17); candidate.reset(seed=17)
    rng = np.random.default_rng(28)
    for _ in range(160):
        action = int(rng.choice(np.flatnonzero(base._mask())))
        left = base.step(action); right = candidate.step(action)
        for key in left[0]:
            np.testing.assert_array_equal(left[0][key], right[0][key])
        assert left[2:4] == right[2:4]
        assert right[1] == pytest.approx(left[1] + right[4]['tower_support_step_reward'])
        assert base.match.battle.damage_log == candidate.match.battle.damage_log
    state = candidate.snapshot()
    expected = candidate.step(0)
    candidate.restore(state)
    actual = candidate.step(0)
    assert actual[1:4] == expected[1:4]
    assert actual[4]['metrics'] == expected[4]['metrics']
    other = TowerSupportContinuousPushEnv(tower_support_weight=.2)
    with pytest.raises(ValueError, match='weight differs'):
        other.restore(state)
    candidate.reset(seed=18)
    assert candidate.tower_support.assisted_value == 0


@pytest.mark.parametrize('weight', [-1, float('nan'), float('inf'), True])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError):
        TowerSupportContinuousPushEnv(tower_support_weight=weight)


def test_training_factory_and_incompatible_modes():
    from sim.train_attack_ppo import make_training_env
    env=make_training_env(curriculum='continuous_v1',tower_support_weight=0.)
    assert isinstance(env,TowerSupportContinuousPushEnv)
    for options in ({'curriculum':'classic'},
                    {'curriculum':'continuous_v1','pending_inputs':True},
                    {'curriculum':'continuous_v1','history_frames':2}):
        with pytest.raises(ValueError):make_training_env(tower_support_weight=.1,**options)
