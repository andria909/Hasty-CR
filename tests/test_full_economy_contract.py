from copy import deepcopy
import json

import pytest

from scripts.attack.verify_global_full_pair import assigned_initial, audit_full_economy
from scripts.kaggle.preflight_global_full_pair import variant_child_script,validate_specialized_preflight
from sim.train_attack_ppo import make_training_env


def test_assigned_policy_overrides_global_ranking_for_new_series_only():
    config=dict(models=[dict(id='parent',sha256='parent'),dict(id='bc800',sha256='bc800')])
    pilot=dict(initial_selection=dict(selected='parent'));trial=dict(initial_id='bc800')
    for variant in ('policy_adaptation_v1','calibrated_spell_cost_v1','group_entropy_v1'):
        assert assigned_initial(config,pilot,trial,variant)['sha256']=='bc800'
    assert assigned_initial(config,pilot,trial,'legacy')['sha256']=='parent'
    with pytest.raises(StopIteration):assigned_initial(config,pilot,dict(initial_id='missing'),'group_entropy_v1')


@pytest.mark.parametrize('variant,expected',[
    ('policy_adaptation_v1','preflight_policy_adaptation_pilot.py'),
    ('calibrated_spell_cost_v1','preflight_policy_adaptation_pilot.py'),
    ('group_entropy_v1','preflight_group_entropy_pilot.py'),
    ('legacy','preflight_global_full_pair.py'),
])
def test_full_preflight_uses_coefficient_and_initializer_aware_child(variant,expected):
    path=variant_child_script(variant)
    assert path.name==expected and path.is_file()


def test_generic_smoke_cannot_enable_new_full_experiment():
    model=dict(id='parent',sha256='correct');trial=dict(initial_id='parent',spatial_entropy_weight=.3)
    entry=dict(selected=dict(trial=trial),config=dict(models=[model]))
    report=dict(explicit_initial_model=model,economy_metrics_verified=True,spatial_entropy_verified=.3)
    validate_specialized_preflight(entry,report,'group_entropy_v1')
    for key in report:
        wrong=deepcopy(report);wrong.pop(key)
        with pytest.raises(ValueError):validate_specialized_preflight(entry,wrong,'group_entropy_v1')


@pytest.mark.parametrize('corruption',[None,'checkpoint_reward','run_entropy','checkpoint_entropy','objective','paid_spell','empty'])
def test_full_verifier_audits_real_environment_metrics(tmp_path,corruption):
    trial=dict(crown_only_weight=0.,dormant_king_weight=0.,spell_waste_weight=.5,
               tower_support_weight=.15,spatial_entropy_weight=.3)
    profile=dict(cannon_center_only=1,tower_damage_all=1,source_radius_reach=1,hog_sight_clipping=1,
                 missed_fireball_penalty=0.)
    env=make_training_env(curriculum='continuous_v1',continuous_profile=profile,
        **{k:v for k,v in trial.items()if k!='spatial_entropy_weight'})
    env.reset(seed=2);row=env.step(0)[4]['metrics']
    config=dict(trial,continuous_profile=profile);blob=dict(config=deepcopy(config))
    metric=dict(episodes=[row],entropy_components=dict(spatial_weight=.3,optimizer_batches=1,full=4.,group=1.,objective=1.9))
    if corruption=='checkpoint_reward':blob['config']['spell_waste_weight']=.1
    if corruption=='run_entropy':config['spatial_entropy_weight']=1.
    if corruption=='checkpoint_entropy':blob['config']['spatial_entropy_weight']=1.
    if corruption=='objective':metric['entropy_components']['objective']=4.
    if corruption=='paid_spell':row['empty_spell_penalty']+=1.
    if corruption=='empty':metric['episodes']=[]
    path=tmp_path/'metrics.jsonl';path.write_text(json.dumps(metric)+'\n')
    if corruption:
        with pytest.raises(ValueError):audit_full_economy(config,blob,path,trial,'group_entropy_v1')
    else:
        assert audit_full_economy(config,blob,path,trial,'group_entropy_v1')==dict(updates=1,episode_records=1,spatial_entropy_weight=.3)
