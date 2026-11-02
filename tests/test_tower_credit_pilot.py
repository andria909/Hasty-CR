from copy import deepcopy
import json
from pathlib import Path
import pytest

from scripts.kaggle.prepare_tower_credit_pilot import validate_protocol,VARIANT
from scripts.kaggle.prepare_air_exposure_pilot import validate_protocol as validate_air
from scripts.kaggle.preflight_global_full_pair import validate_specialized_preflight,variant_child_script


def protocol():
    p=json.loads(Path('data/ppo_tuning/tower_credit_design_20261102.json').read_text())
    p['study_variant']=VARIANT
    return p


def test_tower_factorial_keeps_positive_bonus_controls_and_two_independent_seeds():
    p=protocol();validate_protocol(p)
    for key,value in [('tower_support_weight',0.),('focus','air_golem'),('spell_waste_weight',1.),
                      ('dormant_king_weight',.4),('own_hp_penalty',10),('initial_id','bank')]:
        q=deepcopy(p);q['cases'][3][key]=value
        with pytest.raises(ValueError):validate_protocol(q)
    for key,value in [('pilot_steps_each',102400),('pilot_seeds',[462,462]),('paired_controls',{})]:
        q=deepcopy(p);q[key]=value
        with pytest.raises(ValueError):validate_protocol(q)
    air=json.loads(Path('data/ppo_tuning/air_exposure_design_20261102.json').read_text());air['study_variant']='air_exposure_v1'
    validate_air(air)
    for r in air['cases']:r['tower_support_weight']=.05
    with pytest.raises(ValueError):validate_air(air)


def test_full_run_needs_actual_tower_coefficient_not_generic_reward_smoke():
    model=dict(id='parent',sha256='same');trial=protocol()['cases'][3]
    entry=dict(config=dict(models=[model]),selected=dict(trial=trial))
    report=dict(explicit_initial_model=model,economy_metrics_verified=True,long_economy_full_verified=True,
        actual_focus='uniform',full_steps=4096000,actual_dormant_king_weight=0.,actual_spell_waste_weight=.3,
        actual_tower_support_weight=.05,crown_validation_games=12)
    assert variant_child_script(VARIANT).name=='preflight_long_economy_full.py'
    validate_specialized_preflight(entry,report,VARIANT)
    for wrong in (None,0.,.15):
        with pytest.raises(ValueError,match='tower coefficient'):
            validate_specialized_preflight(entry,dict(report,actual_tower_support_weight=wrong),VARIANT)


def test_dashboard_shows_third_series_without_claiming_unverified_completion(tmp_path):
    from scripts.defense.current_dashboard import render_air_exposure
    air=dict(kernel='a/air',root=str(tmp_path/'air'),pilot_target_steps=3276800,status='accepted')
    tower=dict(kernel='b/tower',root=str(tmp_path/'tower'),pilot_target_steps=3276800,status='training_steps_observed')
    (tmp_path/'tower_credit_training_20261102.json').write_text(json.dumps(tower))
    root=tmp_path/'tower/live_stream';root.mkdir(parents=True)
    (root/'progress.json').write_text(json.dumps(dict(kernel='b/tower',trials={'pilot_tower05_01_s462':dict(steps=204800,target_steps=409600)})))
    text=render_air_exposure(tmp_path,air)
    assert 'Бонус башен: b' in text and '| Башни 0,05; пустой спелл 0,1 | ≥50.0% | нет счётчика |'in text
    assert '100% проверено'not in text
    (root/'progress.json').write_text(json.dumps(dict(kernel='b/other',trials={'pilot_tower05_01_s462':dict(steps=409600,target_steps=409600)})))
    assert '100% проверено'not in render_air_exposure(tmp_path,air)


@pytest.mark.parametrize('where',['config','checkpoint','environment'])
@pytest.mark.parametrize('key',['tower_support_weight','spell_waste_weight','crown_only_weight','dormant_king_weight'])
def test_long_selected_weights_must_preserve_actual_reward_configuration(where,key):
    from scripts.attack.verify_air_exposure_pilot import audit_checkpoint_rewards
    trial=protocol()['cases'][3];c={k:trial[k]for k in ('tower_support_weight','spell_waste_weight','crown_only_weight','dormant_king_weight')}
    blob=dict(config=deepcopy(c),environment_states=[deepcopy(c),deepcopy(c)])
    audit_checkpoint_rewards(c,blob,trial)
    target={'config':c,'checkpoint':blob['config'],'environment':blob['environment_states'][1]}[where]
    target[key]+=1.
    with pytest.raises(ValueError,match='reward differs'):audit_checkpoint_rewards(c,blob,trial)
