import pytest
from scripts.attack.audit_continuous_rewards import components, timing


def test_exact_reward_reconstruction_and_mismatch_rejection():
    profile=dict(enemy_hp_reward=3,own_hp_penalty=4,overflow_penalty=.15,
                 win_reward=20,loss_penalty=20)
    row=dict(split='train',outcome='win',enemy_fraction_damage=1,
             own_fraction_loss=.5,elixir_overflow=2,illegal_actions=1,episode_return=20.6)
    assert sum(components(row,profile).values())==pytest.approx(20.6)
    row['episode_return']=21
    with pytest.raises(ValueError,match='Return does not match'): components(row,profile)
    row['split']='test'
    with pytest.raises(ValueError,match='TRAIN'): components(row,profile)


def test_discount_uses_actual_decision_clock_and_separates_gae_trace():
    old=timing(.999,.95,500); new=timing(.999,.95,250)
    assert new['discounted_terminal_weight_at_300s']==pytest.approx(
        old['discounted_terminal_weight_at_300s']**2)
    assert new['discount_half_life_seconds']==pytest.approx(old['discount_half_life_seconds']/2)
    assert new['gae_residual_half_life_seconds']<new['discount_half_life_seconds']
