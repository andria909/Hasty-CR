import numpy as np
import pytest

from sim.continuous_spell_waste import SpellWasteContinuousPushEnv
from sim.continuous_tactical_starts import TacticalStartContinuousPushEnv
from sim.env import DECK_26

OPTIONS = dict(action_delay_ms=1350, continuous_profile={'tower_damage_all':1},
               tower_support_weight=.15, spell_waste_weight=.1)


@pytest.mark.parametrize('split,fraction', [('train',0), ('validation',1), ('test',1)])
def test_control_and_heldout_splits_preserve_original_game(split, fraction):
    base = SpellWasteContinuousPushEnv(split=split, **OPTIONS)
    candidate = TacticalStartContinuousPushEnv(split=split, tactical_start_fraction=fraction, **OPTIONS)
    left, _ = base.reset(seed=32001)
    right, _ = candidate.reset(seed=32001)
    assert candidate.tactical_start is None
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])
    for _ in range(20):
        a, b = base.step(0), candidate.step(0)
        assert a[1:4] == b[1:4]
        assert base.match.play_log == candidate.match.play_log
        for key in a[0]:
            np.testing.assert_array_equal(a[0][key], b[0][key])


def test_paid_diverse_starts_keep_real_hand_and_continue_enemy_play():
    env = TacticalStartContinuousPushEnv(tactical_start_fraction=1, **OPTIONS)
    hands, banks, masks, lanes, families = set(), set(), set(), set(), set()
    for seed in range(32010, 32034):
        obs, info = env.reset(seed=seed)
        start = env.tactical_start
        assert len(obs['scalars']) == 309 and obs['planes'].shape[0] == 20
        hands.add(tuple(start['own_hand']));banks.add(start['own_elixir'])
        masks.add(start['own_princess_mask']);lanes.add(start['lane']);families.add(start['family'])
        assert sorted(env.match.players[1].hand+env.match.players[1].queue) == sorted(DECK_26)
        paid = sum(env.cards[r['card']].cost for r in start['enemy_setup'])
        assert env.match.players[-1].elixir_spent == paid
        assert paid <= 10
        for row in start['enemy_setup']:
            assert row['elixir_before']-row['elixir_after'] == env.cards[row['card']].cost*1000
        assert info['metrics']['tower_support_reward'] == 0
        assert env._previous['own_fraction_loss'] == start['initial_own_fraction_loss']
    assert len(hands)>10 and len(banks)>4 and masks==set(range(4)) and len(lanes)==2 and len(families)>=5
    before = len(env.enemy_actions)
    for _ in range(160):
        if env.done:break
        env.step(0)
    assert len(env.enemy_actions)>before


def test_snapshot_restores_setup_rewards_and_opponent_rng_exactly():
    env = TacticalStartContinuousPushEnv(tactical_start_fraction=1, **OPTIONS)
    env.reset(seed=33009)
    state = env.snapshot()
    expected = [env.step(0) for _ in range(30)]
    env.restore(state)
    actual = [env.step(0) for _ in range(30)]
    for a, b in zip(expected, actual):
        assert a[1:4] == b[1:4] and a[4]['metrics'] == b[4]['metrics']
        for key in a[0]:np.testing.assert_array_equal(a[0][key], b[0][key])
    other = TacticalStartContinuousPushEnv(tactical_start_fraction=.5, **OPTIONS)
    with pytest.raises(ValueError, match='fraction differs'):other.restore(state)


@pytest.mark.parametrize('fraction', [-1, 1.01, True, float('nan')])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError):TacticalStartContinuousPushEnv(tactical_start_fraction=fraction, **OPTIONS)


def test_training_factory_requires_explicit_coefficients():
    from sim.train_attack_ppo import make_training_env
    env=make_training_env(curriculum='continuous_v1', tactical_start_fraction=.25, **OPTIONS)
    assert isinstance(env,TacticalStartContinuousPushEnv)
    with pytest.raises(ValueError):make_training_env(curriculum='continuous_v1',tactical_start_fraction=.25)
