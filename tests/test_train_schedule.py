from sim.train_ppo import next_eval_boundary


def test_eval_boundaries_do_not_drift_with_rollout_overshoot():
    assert next_eval_boundary(25_088, 25_000) == 50_000
    assert next_eval_boundary(50_176, 25_000) == 75_000
    assert next_eval_boundary(75_008, 25_000) == 100_000


def test_eval_boundary_is_strictly_after_an_exact_boundary():
    assert next_eval_boundary(25_000, 25_000) == 50_000


def test_nonpositive_interval_degenerates_to_every_step():
    assert next_eval_boundary(17, 0) == 18
