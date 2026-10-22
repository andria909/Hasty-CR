"""Algorithmic boundaries, not expensive local model training."""
import numpy as np
import pytest
import torch

from sim.train_defense_ppo import (action_masks, generalized_advantage,
                                  normalize_advantages, require_finite,
                                  rng_state, restore_rng, parser)


def test_terminal_never_bootstraps_or_leaks_reset_return():
    rewards = torch.tensor([[2.], [100.]])
    values = torch.tensor([[1.], [0.]])
    following = torch.tensor([[99.], [0.]])
    terminated = torch.tensor([[True], [True]])
    truncated = torch.zeros_like(terminated)
    advantage, returns = generalized_advantage(rewards, values, following, terminated, truncated, gamma=.9, gae_lambda=1)
    assert advantage[:, 0].tolist() == [1., 100.]
    assert returns[:, 0].tolist() == [2., 100.]


def test_truncation_bootstraps_final_observation_not_reset():
    advantage, returns = generalized_advantage(
        torch.tensor([[2.], [100.]]), torch.tensor([[1.], [0.]]),
        torch.tensor([[10.], [0.]]), torch.tensor([[False], [True]]),
        torch.tensor([[True], [False]]), gamma=.9, gae_lambda=1)
    assert returns[0, 0].item() == pytest.approx(11.)
    assert advantage[0, 0].item() == pytest.approx(10.)


def test_unfinished_rollout_bootstraps_and_propagates_gae():
    advantage, returns = generalized_advantage(
        torch.tensor([[1.], [2.]]), torch.tensor([[3.], [4.]]),
        torch.tensor([[4.], [5.]]), torch.zeros(2, 1, dtype=torch.bool),
        torch.zeros(2, 1, dtype=torch.bool), gamma=.9, gae_lambda=.8)
    assert advantage[1, 0].item() == pytest.approx(2.5)
    assert advantage[0, 0].item() == pytest.approx(3.4)
    assert returns[0, 0].item() == pytest.approx(6.4)


def test_normalization_uses_entire_batch_and_singleton_is_finite():
    values = torch.tensor([[0., 1.], [2., 9.]])
    normalized = normalize_advantages(values)
    assert normalized.mean().item() == pytest.approx(0, abs=1e-7)
    assert normalized.std(unbiased=False).item() == pytest.approx(1)
    assert normalize_advantages(torch.ones(1)).item() == 0


def test_empty_action_mask_and_nonfinite_fail_closed():
    with pytest.raises(ValueError, match='empty legal'):
        action_masks([{'action_mask': [False, False]}], 'cpu')
    with pytest.raises(FloatingPointError, match='reward'):
        require_finite(reward=torch.tensor([float('nan')]))


def test_rng_checkpoint_roundtrip(tmp_path):
    import random
    state = rng_state()
    path = tmp_path / 'rng.pt'
    torch.save(state, path)
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng(torch.load(path, weights_only=False))
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def test_defaults_are_finite_cuda_smoke_and_explicit_paths():
    args = parser().parse_args(['--init', 'baseline.pt', '--output', 'new-output'])
    assert (args.updates, args.envs, args.rollout, args.device) == (5, 2, 8, 'cuda')


def test_environment_snapshot_serialization_and_timeout_bootstrap(tmp_path):
    """Public API integration, with one action and no optimizer/training."""
    from sim.defense_env import DefenseEnv
    env = DefenseEnv(seed=42, split='train', max_ms=550)
    observations, info = env.reset(seed=42)
    action = int(np.flatnonzero(info['action_mask'])[1])
    path = tmp_path / 'environment.pt'
    torch.save(env.snapshot(), path)
    expected = env.step(action)
    restored_obs, restored_info = env.restore(torch.load(path, weights_only=False))
    assert all(np.array_equal(observations[k], restored_obs[k]) for k in observations)
    assert np.array_equal(info['action_mask'], restored_info['action_mask'])
    assert env.match.battle.unit_lookup.match is env.match
    actual = env.step(action)
    assert all(np.array_equal(expected[0][k], actual[0][k]) for k in observations)
    assert actual[1:4] == expected[1:4]
    assert actual[4]['metrics'] == expected[4]['metrics']
    assert np.array_equal(actual[4]['action_mask'], expected[4]['action_mask'])
    final_obs, _, terminated, truncated, final_info = env.step(0)
    assert truncated and not terminated
    assert final_info['terminal_reason'] == 'timeout'
    assert np.flatnonzero(final_info['action_mask']).tolist() == [0]
    assert all(np.isfinite(v).all() for v in final_obs.values())


def test_v25_loads_defense_observations_without_head_migration():
    """Inference only; the private baseline is optional in clean clones."""
    from pathlib import Path
    from sim.defense_env import DefenseEnv
    from sim.train_ppo import build_network, checkpoint_architecture, load_policy_state
    from sim.train_defense_ppo import tensor_observations
    path = Path(__file__).resolve().parents[1] / 'checkpoints/portable_v25/portable_v25_cannon_economy_best.pt'
    if not path.exists():
        pytest.skip('private v25 checkpoint not present')
    env = DefenseEnv(seed=42)
    obs, info = env.reset(seed=42)
    blob = torch.load(path, map_location='cpu', weights_only=False)
    network = build_network(obs['planes'].shape[0], obs['scalars'].shape[0],
                            len(info['action_mask']), checkpoint_architecture(blob))
    assert not load_policy_state(network, blob['state_dict'])
    with torch.no_grad():
        logits, values = network(*tensor_observations([obs], 'cpu'))
    assert logits.shape == (1, len(info['action_mask']))
    assert values.shape == (1,)
    require_finite(logits=logits, values=values)
