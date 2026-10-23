import json
import pytest
from scripts.kaggle.pack_strategy_evaluation import supported_training_modes,digest


def contract():
    protocol=dict(schema='ppo-coefficient-study-v1',algorithm='ppo',modes=['defense','joint'],test_used_for_search=False,
                  full_environment_steps=2048000,short_environment_steps=102400)
    raw=json.dumps(protocol).encode()
    members={name:b'implementation' for name in ('sim/strategy_env.py','sim/train_defense_ppo.py','sim/train_attack_ppo.py','sim/reward_profiles.py')}
    members['data/ppo_tuning/protocol.json']=raw
    training=dict(schema='ppo-coefficient-source-v1',protocol_sha256=digest(raw),test_used_for_search=False)
    return training,members


def test_explicit_dual_regime_source_and_existing_joint_source_are_distinct():
    training,members=contract()
    assert supported_training_modes(training,members)==['defense','joint']
    assert supported_training_modes(dict(observation_contract='joint_attack_defense_timer_task_v1',num_scalars=309),{})==['joint']


def test_unpinned_or_test_tuned_source_cannot_be_relabelled_as_supported():
    training,members=contract();training['protocol_sha256']='different'
    with pytest.raises(ValueError,match='SHA'):supported_training_modes(training,members)
    training,members=contract();training['test_used_for_search']=True
    with pytest.raises(ValueError,match='dual-regime'):supported_training_modes(training,members)
    training,members=contract();members.pop('sim/train_defense_ppo.py')
    with pytest.raises(ValueError,match='implementation missing'):supported_training_modes(training,members)
    with pytest.raises(ValueError,match='Wrong training'):supported_training_modes({'schema':'arbitrary'},members)
