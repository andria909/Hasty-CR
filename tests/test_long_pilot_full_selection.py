from copy import deepcopy
import json

import pytest

from scripts.attack import select_pilot_pool as pool
from scripts.kaggle.preflight_global_full_pair import validate_specialized_preflight,variant_child_script
from tests.test_paired_ppo_behavior import records


def study(tmp_path,name,variant,seeds,cost):
    root=tmp_path/name;(root/'dataset').mkdir(parents=True);(root/'collected').mkdir()
    trial=dict(id=name,initial_id='parent',focus='uniform',spell_waste_weight=cost)
    protocol=dict(study_variant=variant,pilot_seeds=seeds,paired_controls={})
    (root/'dataset/protocol.json').write_text(json.dumps(protocol))
    preparation=dict(study_variant=variant,config=dict(models=[dict(id='parent',sha256='same')]))
    (root/'preparation.json').write_text(json.dumps(preparation))
    data=dict(validation=dict(records=records()))
    (root/'collected/parent_reference.json').write_text(json.dumps(data))
    for seed in seeds:(root/'collected'/f'pilot_{name}_s{seed}_summary.json').write_text(json.dumps(data))
    prefix=pool.LONG_VARIANTS[variant]
    proof=dict(kernel=name,pilot_steps_verified=3276800,pilot_models=[],promotion_blocked=True,promotion_block_reason='fewer_than_two_eligible_profiles',
        pilot_selection=dict(floor=.1,ranking=[dict(trial=trial,score=.3,minimum_family_points=.2,eligible=True)]))
    proof[prefix+'_audit']=dict(verified=True)
    proof[prefix+'_selection']=dict(eligible_profiles=[name],gate_met=False)
    return root,proof


def test_single_qualified_profile_from_each_long_cohort_can_make_full_pair(tmp_path,monkeypatch):
    a,pa=study(tmp_path,'air','air_exposure_v1',[442,443],.1)
    b,pb=study(tmp_path,'king','spell_interaction_v1',[452,453],.3)
    monkeypatch.setattr(pool,'verify_study',lambda root:{a:pa,b:pb}[root])
    monkeypatch.setattr(pool,'contract',lambda root:dict(core={'physics':'same'}))
    assert not pool.select([a])['gate_met']
    result=pool.select([a,b]);assert result['gate_met']and len(result['selected'])==2
    assert [r['pilot_steps_verified']for r in result['studies']]==[3276800]*2
    pa['pilot_steps_verified']=819200
    with pytest.raises(ValueError,match='Complete verified'):pool.select([a,b])


def test_long_global_pool_cannot_override_specialized_defense_gate(tmp_path,monkeypatch):
    root,proof=study(tmp_path,'air','air_exposure_v1',[442,443],.1)
    monkeypatch.setattr(pool,'verify_study',lambda root:proof)
    monkeypatch.setattr(pool,'contract',lambda root:dict(core={'physics':'same'}))
    proof['air_exposure_selection']['eligible_profiles']=[]
    with pytest.raises(ValueError,match='eligibility differ'):pool.select([root])
    proof['air_exposure_audit']['verified']=False
    with pytest.raises(ValueError,match='audit required'):pool.select([root])


def test_budget_distinguishes_replica_identity_but_long_studies_share_controls():
    p=dict(config=dict(models=[dict(id='parent',sha256='same')]))
    trial=dict(id='control',initial_id='parent',spell_waste_weight=.1)
    short=pool.economy_identity(p,trial)
    p['study_variant']='air_exposure_v1';air=pool.economy_identity(p,trial)
    p['study_variant']='spell_interaction_v1';king=pool.economy_identity(p,trial)
    assert short!=air and air==king and air['pilot_steps_each']==409600


@pytest.mark.parametrize('variant',['air_exposure_v1','spell_interaction_v1'])
def test_full_preflight_requires_full_branch_focus_and_actual_king_accounting(variant):
    assert variant_child_script(variant).name=='preflight_long_economy_full.py'
    model=dict(id='parent',sha256='same')
    trial=dict(initial_id='parent',focus='air_golem',dormant_king_weight=.4,spell_waste_weight=.3)
    entry=dict(selected=dict(trial=trial),config=dict(models=[model]))
    report=dict(explicit_initial_model=model,economy_metrics_verified=True,long_economy_full_verified=True,
        actual_focus='air_golem',full_steps=4096000,actual_dormant_king_weight=.4,
        actual_spell_waste_weight=.3,crown_validation_games=12)
    validate_specialized_preflight(entry,report,variant)
    for key in report:
        wrong=deepcopy(report);wrong.pop(key)
        with pytest.raises(ValueError):validate_specialized_preflight(entry,wrong,variant)
    wrong=dict(report,full_steps=409600)
    with pytest.raises(ValueError):validate_specialized_preflight(entry,wrong,variant)


def test_unrelated_promotion_block_is_never_ignored_for_long_study(tmp_path,monkeypatch):
    root,proof=study(tmp_path,'air','air_exposure_v1',[442,443],.1)
    proof['promotion_block_reason']='invalid physics'
    monkeypatch.setattr(pool,'verify_study',lambda root:proof)
    with pytest.raises(ValueError,match='invalid physics'):pool.select([root])


@pytest.mark.parametrize('corruption',[None,'focus','king_coefficient','paid_king'])
def test_long_full_audit_checks_checkpoint_and_paid_king_cost(tmp_path,corruption):
    from sim.train_attack_ppo import make_training_env
    from scripts.attack.verify_global_full_pair import audit_full_economy
    trial=dict(crown_only_weight=0.,dormant_king_weight=.4,spell_waste_weight=.3,
        tower_support_weight=.15,focus='uniform')
    profile=dict(cannon_center_only=1,tower_damage_all=1,source_radius_reach=1,hog_sight_clipping=1,
        missed_fireball_penalty=0.)
    env=make_training_env(curriculum='continuous_v1',continuous_profile=profile,
        **{k:v for k,v in trial.items()if k!='focus'})
    env.reset(seed=2);row=env.step(0)[4]['metrics']
    config=dict({k:v for k,v in trial.items()if k!='focus'},continuous_profile=profile,continuous_focus='uniform')
    blob=dict(config=deepcopy(config))
    if corruption=='focus':blob['config']['continuous_focus']='air_golem'
    if corruption=='king_coefficient':blob['config']['dormant_king_weight']=0.
    if corruption=='paid_king':row['spell_economy_penalty']+=1.
    path=tmp_path/'metrics.jsonl';path.write_text(json.dumps(dict(episodes=[row]))+'\n')
    if corruption:
        with pytest.raises(ValueError):audit_full_economy(config,blob,path,trial,'spell_interaction_v1')
    else:assert audit_full_economy(config,blob,path,trial,'spell_interaction_v1')['episode_records']==1
