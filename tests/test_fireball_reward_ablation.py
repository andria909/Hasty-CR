from copy import deepcopy
import pytest
from scripts.kaggle.prepare_fireball_reward_ablation import CASES, VARIANT, validate_protocol


def protocol():
    return dict(study_variant=VARIANT, pilot_seeds=[272,273], cases=[dict(
        id=name,warmstart_updates=updates,fireball_value_reward=reward,
        slot_augmentation=bool(updates),warmstart_kl_weight=0.,supervised_replay_weight=0.,
        action_delay_ms=1350,validation_action_delay_ms=1350,own_hp_penalty=6.)
        for name,updates,reward in CASES])


def test_factorial_has_only_two_controlled_interventions():
    validate_protocol(protocol())


@pytest.mark.parametrize('key,value',[
    ('own_hp_penalty',9.),('action_delay_ms',225),('supervised_replay_weight',1.),
    ('warmstart_updates',200),('fireball_value_reward',.5),('slot_augmentation',False)])
def test_rejects_confounding_or_wrong_factor(key,value):
    p=deepcopy(protocol());p['cases'][2][key]=value
    with pytest.raises(ValueError):validate_protocol(p)


def test_rejects_wrong_seed_and_missing_cell():
    p=protocol();p['pilot_seeds']=[222,223]
    with pytest.raises(ValueError):validate_protocol(p)
    p=protocol();p['cases'].pop()
    with pytest.raises(ValueError):validate_protocol(p)
