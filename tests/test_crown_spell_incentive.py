import pytest

from scripts.attack.probe_crown_spell_incentive import branch
from tests.test_calibrated_spell_pilot import protocol


@pytest.mark.parametrize('spell,cost',[('fireball',4),('zap',2)])
def test_empty_penalty_does_not_apply_to_actual_king_contact(spell,cost):
    trial=dict(protocol()['cases'][0],fireball_value_reward=.25,reserve_potential_weight=.1,
               own_hp_penalty=6.,barbarian_train_fraction=.35,lane_reflection_fraction=.5)
    weights=dict(spell_waste_weight=.1,crown_only_weight=0.,dormant_king_weight=0.)
    wait=branch(trial,spell,'king',weights,False)
    current=branch(trial,spell,'king',weights,True)
    empty=branch(trial,spell,'king',dict(weights,spell_waste_weight=.5),True)
    king=branch(trial,spell,'king',dict(weights,spell_waste_weight=.5,dormant_king_weight=.4),True)
    assert current['king_dormant_before']and not current['king_dormant_after']
    assert current['crown_only_elixir']==current['dormant_king_elixir']==cost
    assert current['empty_spell_elixir']==0
    assert len(current['hits'])==1 and current['hits'][0]['is_tower']
    assert current['reward']==empty['reward']
    assert current['reward']>wait['reward']>king['reward']
    assert current['reward']-king['reward']==pytest.approx(.4*cost)
    assert current['reward']-wait['reward']==pytest.approx(.3*cost+3*current['enemy_fraction_damage'])
