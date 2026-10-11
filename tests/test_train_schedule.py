import torch

from sim.train_ppo import (keep_appended_feature_gradients,
                           next_eval_boundary, parameter_is_trainable)


def test_eval_boundaries_do_not_drift_with_rollout_overshoot():
    assert next_eval_boundary(25_088, 25_000) == 50_000
    assert next_eval_boundary(50_176, 25_000) == 75_000
    assert next_eval_boundary(75_008, 25_000) == 100_000


def test_eval_boundary_is_strictly_after_an_exact_boundary():
    assert next_eval_boundary(25_000, 25_000) == 50_000


def test_nonpositive_interval_degenerates_to_every_step():
    assert next_eval_boundary(17, 0) == 18


def test_feature_only_mode_freezes_old_policy_but_trains_value():
    assert parameter_is_trainable(
        "scalar.0.weight", warmup=False, feature_only=True)
    assert parameter_is_trainable(
        "value.weight", warmup=False, feature_only=True)
    assert not parameter_is_trainable(
        "conv.0.weight", warmup=False, feature_only=True)
    assert not parameter_is_trainable(
        "scalar.0.weight", warmup=True, feature_only=True)


def test_feature_only_gradient_keeps_only_appended_columns():
    gradient = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    masked = keep_appended_feature_gradients(gradient, 2)
    assert torch.equal(masked[:, :-2], torch.zeros(2, 4))
    assert torch.equal(masked[:, -2:], gradient[:, -2:])
