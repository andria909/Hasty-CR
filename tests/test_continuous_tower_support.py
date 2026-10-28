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


def test_real_tower_damage_rewards_all_mode_and_resume_without_recredit():
    from sim.entities import make_unit
    from sim import arena
    base=TowerSupportContinuousPushEnv(tower_support_weight=.4)
    all_damage=TowerSupportContinuousPushEnv(tower_support_weight=.4,continuous_profile={'tower_damage_all':1})
    for e in (base,all_damage):
        e.reset(seed=17);e._enemy=lambda:None
        enemy=make_unit(0,e.cards['giant'].unit,-1,arena.tile(3,22),0)
        enemy.elixir_value=5000;e.match.battle.add(enemy)
    total_extra=0.
    for _ in range(40):
        a=base.step(0);b=all_damage.step(0)
        assert a[2:4]==b[2:4]
        assert base.match.battle.damage_log==all_damage.match.battle.damage_log
        total_extra+=b[1]-a[1]
    assert all_damage.tower_support.total_paid_value>0
    assert base.tower_support.assisted_value==0
    assert total_extra==pytest.approx(.4*all_damage.tower_support.total_paid_value)
    state=all_damage.snapshot();expected=all_damage.step(0)
    all_damage.restore(state);actual=all_damage.step(0)
    assert expected[1:4]==actual[1:4]
    assert expected[4]['metrics']==actual[4]['metrics']
    with pytest.raises(ValueError):base.restore(state)
