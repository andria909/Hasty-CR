import pytest

from sim.defense_checkpointing import checkpoint_due


def test_sparse_cadence_preserves_final_and_budget_boundary_recovery():
    assert [u for u in range(1,401) if checkpoint_due(u,50)]==list(range(50,401,50))
    assert checkpoint_due(393,50,final=True)
    assert checkpoint_due(393,50,budget_stop=True)
    assert not checkpoint_due(393,50)
    assert not checkpoint_due(0,50)
    assert all(checkpoint_due(u)for u in range(1,10))


@pytest.mark.parametrize('invalid',[0,-1,1.5,True])
def test_invalid_cadence_fails_before_training(invalid):
    with pytest.raises(ValueError):checkpoint_due(0,invalid)


def test_checkpoint_cadence_is_mutable_on_resume():
    from sim.train_defense_grpo import RESUME_KEYS
    from sim.train_defense_ppo import parser
    assert 'checkpoint_every' not in RESUME_KEYS
    assert parser().parse_args(['--init','x','--output','new']).checkpoint_every==1
    assert parser().parse_args(['--init','x','--output','new','--checkpoint-every','50']).checkpoint_every==50


def test_grpo_step_target_uses_carried_completed_steps_and_allows_group_overshoot():
    from sim.train_defense_grpo import environment_steps_target_reached,RESUME_KEYS
    assert not environment_steps_target_reached(1000000,None)
    assert not environment_steps_target_reached(999999,1000000)
    assert environment_steps_target_reached(1000123,1000000)
    assert environment_steps_target_reached(1000000,1000000)
    assert 'target_environment_steps' not in RESUME_KEYS
    for target in (0,-1,True):
        with pytest.raises(ValueError):environment_steps_target_reached(0,target)
