from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.kaggle.prepare_critic_weight_pilot import VARIANT,validate_protocol


def protocol():
    return json.loads(Path('data/ppo_tuning/critic_weight_design_20261104.json').read_text())


@pytest.mark.parametrize('key,value',[
    ('value_coef',0.),('gae_lambda',.98),('tower_support_weight',0.),('spell_waste_weight',.1),
    ('card_balance_weight',0.),('action_delay_ms',225),('focus','air_golem'),('policy_reduction','choice')])
def test_sweep_rejects_confounding_treatments(key,value):
    p=protocol();validate_protocol(p);p['cases'][3][key]=value
    with pytest.raises(ValueError):validate_protocol(p)


def test_actual_full_preflight_requires_critic_coefficient():
    from scripts.kaggle.preflight_global_full_pair import validate_specialized_preflight,variant_child_script
    trial=protocol()['cases'][3];model=dict(id='parent',sha256='fixed')
    entry=dict(config=dict(models=[model]),selected=dict(trial=trial))
    report=dict(explicit_initial_model=model,economy_metrics_verified=True,long_economy_full_verified=True,
        actual_focus='uniform',full_steps=4096000,actual_dormant_king_weight=0.,actual_spell_waste_weight=.3,
        actual_tower_support_weight=.15,crown_validation_games=12,actual_value_coef=.05)
    validate_specialized_preflight(entry,report,VARIANT)
    assert variant_child_script(VARIANT).name=='preflight_long_economy_full.py'
    with pytest.raises(ValueError,match='critic'):
        validate_specialized_preflight(entry,dict(report,actual_value_coef=.5),VARIANT)


def test_default_control_identity_keeps_old_failed_replicas():
    from scripts.attack.select_pilot_pool import economy_identity
    prep=dict(study_variant=VARIANT,config=dict(models=[dict(id='parent',sha256='fixed')]))
    trial=protocol()['cases'][0];old=deepcopy(trial);old.pop('value_coef')
    assert economy_identity(prep,trial)==economy_identity(dict(prep,study_variant='air_exposure_v1'),old)
    assert economy_identity(prep,dict(trial,value_coef=.05))!=economy_identity(prep,trial)


def test_selected_and_full_verifiers_reject_wrong_value_weight():
    from scripts.attack.verify_global_full_pair import audit_full_economy
    trial=protocol()['cases'][3]
    config={k:trial[k]for k in('tower_support_weight','spell_waste_weight','crown_only_weight','dormant_king_weight')}
    config.update(continuous_focus='uniform',value_coef=.5)
    with pytest.raises(ValueError,match='critic'):
        audit_full_economy(config,dict(config=config),'unused',trial,VARIANT)
