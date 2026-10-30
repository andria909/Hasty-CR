from copy import deepcopy
import pytest
from scripts.kaggle.prepare_crown_economy_pilot import validate_protocol
from scripts.attack.verify_crown_economy_pilot import audit_rewards
from sim.train_attack_ppo import make_training_env


def test_protocol_rejects_confounded_and_unsafe_cases():
    base=dict(warmstart_updates=0,supervised_replay_weight=0.,action_delay_ms=1350,
              validation_action_delay_ms=1350,cannon_center_only=1,tower_damage_all=1,
              source_radius_reach=1,hog_sight_clipping=1,tactical_start_fraction=0.,card_balance_weight=.03)
    p=dict(study_variant='crown_economy_v1',pilot_seeds=[382,383],total_pilot_steps=819200,
           cases=[dict(base,id=name,crown_only_weight=c,dormant_king_weight=k)
                  for name,(c,k)in zip(('control','low','medium','high'),[(0.,0.),(0.,.1),(0.,.4),(.1,.4)])])
    validate_protocol(p)
    bad=deepcopy(p);bad['cases'][1]['action_delay_ms']=225
    with pytest.raises(ValueError):validate_protocol(bad)
    bad=deepcopy(p);bad['cases'][2]['dormant_king_weight']=4
    with pytest.raises(ValueError):validate_protocol(bad)


def test_verifier_checks_actual_episode_economy_and_coefficients():
    trial=dict(crown_only_weight=.1,dormant_king_weight=.4,spell_waste_weight=.1,tower_support_weight=.15)
    env=make_training_env(curriculum='continuous_v1',**trial,
        continuous_profile=dict(cannon_center_only=1,tower_damage_all=1,source_radius_reach=1,hog_sight_clipping=1))
    env.reset(seed=2);row=env.step(0)[4]['metrics']
    audit_rewards(trial,[row],trial)
    bad=deepcopy(row);bad['spell_economy_penalty']=1
    with pytest.raises(ValueError,match='paid disjoint bins'):audit_rewards(trial,[bad],trial)
    with pytest.raises(ValueError,match='coefficient'):audit_rewards(dict(trial,dormant_king_weight=0),[row],trial)
