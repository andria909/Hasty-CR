from copy import deepcopy
import pytest
from scripts.kaggle.prepare_combined_reward_pilot import CASES,VARIANT,validate_protocol


def protocol():
    return dict(study_variant=VARIANT,pilot_seeds=[282,283],cases=[
        dict(id=name,tower_support_weight=tower,spell_waste_weight=spell,
             missed_fireball_penalty=.1 if spell is None else 0.,
             warmstart_updates=0,supervised_replay_weight=0.,
             action_delay_ms=1350,validation_action_delay_ms=1350,
             own_hp_weight=6.) for name,tower,spell in CASES])


def test_factorial_requires_identical_physics_and_other_rewards():
    good=protocol();validate_protocol(good)
    for change in [dict(own_hp_weight=4.),dict(action_delay_ms=225),
                   dict(validation_action_delay_ms=225),dict(warmstart_updates=800),
                   dict(supervised_replay_weight=1.),dict(tower_support_weight=0.),
                   dict(spell_waste_weight=None),dict(missed_fireball_penalty=.1)]:
        bad=deepcopy(good);bad['cases'][3].update(change)
        with pytest.raises(ValueError):validate_protocol(bad)


def test_factorial_requires_both_paired_seeds_and_all_four_cases():
    for change in [dict(pilot_seeds=[282]),dict(cases=protocol()['cases'][:3]),
                   dict(study_variant='other')]:
        bad=protocol();bad.update(change)
        with pytest.raises(ValueError):validate_protocol(bad)
