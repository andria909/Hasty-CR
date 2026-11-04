from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.kaggle.prepare_policy_choice_pilot import VARIANT,validate_protocol
from scripts.attack.policy_choice_accounting import audit_update,audit_checkpoint


def protocol():
    p=json.loads(Path('data/ppo_tuning/policy_choice_design_20261104.json').read_text())
    p['study_variant']=VARIANT
    return p


def test_frozen_bundle_cannot_omit_the_entropy_dependency():
    p=protocol();p['training_sources'].pop('sim/group_entropy.py')
    with pytest.raises(ValueError,match='dependency pins'):validate_protocol(p)


@pytest.mark.parametrize('key,value',[
    ('policy_reduction','unknown'),('advantage_normalization','all'),('gae_lambda',.98),
    ('card_balance_weight',0.),('own_hp_penalty',1.),('action_delay_ms',225),('tower_support_weight',0.)])
def test_fixed_choice_grid_rejects_extra_treatments(key,value):
    p=protocol();validate_protocol(p);p['cases'][3][key]=value
    with pytest.raises(ValueError):validate_protocol(p)


def fixture():
    trial=dict(policy_reduction='choice',advantage_normalization='choice')
    config=dict(trial,envs=2,rollout=4,epochs=2,minibatch_size=4,target_kl=.02)
    row=dict(policy_decisions=dict(reduction='choice',advantage_normalization='choice',rollout_steps=8,
        choice_steps=3,forced_steps=5,optimizer_samples=16,choice_samples=6,no_choice_minibatches=1,
        optimizer_batches=4,actor_updates_enabled=True),optimizer_controls=dict(kl_stopped=False,value_only_warmup=False),
        entropy_components=dict(spatial_weight=1.,full=2.,group=1.,objective=2.,optimizer_batches=4))
    return row,config,trial


def test_early_kl_stop_is_allowed_but_incomplete_unstopped_epochs_are_rejected():
    row,c,t=fixture();audit_update(row,c,t)
    row['policy_decisions'].update(optimizer_samples=8,choice_samples=3,optimizer_batches=2,no_choice_minibatches=0)
    row['entropy_components']['optimizer_batches']=2
    with pytest.raises(ValueError,match='Unstopped'):audit_update(row,c,t)
    row['optimizer_controls']['kl_stopped']=True
    audit_update(row,c,t)
    with pytest.raises(ValueError,match='KL'):audit_update(row,dict(c,target_kl=0),t)


@pytest.mark.parametrize('key,value',[
    ('choice_steps',4),('optimizer_samples',17),('choice_samples',7),('actor_updates_enabled',False),
    ('reduction','all'),('advantage_normalization','all'),('optimizer_batches',5)])
def test_counter_or_treatment_corruption_is_rejected(key,value):
    row,c,t=fixture();row['policy_decisions'][key]=value
    with pytest.raises(ValueError):audit_update(row,c,t)


def test_no_choice_rollout_has_no_action_entropy_and_keeps_critic_batches():
    row,c,t=fixture();row['policy_decisions'].update(choice_steps=0,forced_steps=8,choice_samples=0,no_choice_minibatches=4)
    with pytest.raises(ValueError,match='action entropy'):audit_update(row,c,t)
    row['entropy_components'].update(full=0.,group=0.,objective=0.)
    audit_update(row,c,t)


@pytest.mark.parametrize('key',['policy_reduction','advantage_normalization'])
def test_selected_weights_cannot_silently_change_objective(key):
    _,c,t=fixture();blob=dict(config=deepcopy(c));audit_checkpoint(c,blob,t)
    blob['config'][key]='all'
    with pytest.raises(ValueError,match='Saved'):audit_checkpoint(c,blob,t)


def test_full_preflight_requires_actual_modes_not_generic_ppo_smoke():
    from scripts.kaggle.preflight_global_full_pair import variant_child_script,validate_specialized_preflight
    trial=protocol()['cases'][3];model=dict(id='parent',sha256='fixed')
    entry=dict(config=dict(models=[model]),selected=dict(trial=trial))
    report=dict(explicit_initial_model=model,economy_metrics_verified=True,long_economy_full_verified=True,
        actual_focus='uniform',full_steps=4096000,actual_dormant_king_weight=0.,actual_spell_waste_weight=.3,
        actual_tower_support_weight=.15,crown_validation_games=12,policy_choice_verified=True,
        full_environment_resume_equal=True,actual_policy_reduction='choice',actual_advantage_normalization='choice')
    assert variant_child_script(VARIANT).name=='preflight_policy_choice_full.py'
    validate_specialized_preflight(entry,report,VARIANT)
    for key in ('policy_choice_verified','full_environment_resume_equal','actual_policy_reduction','actual_advantage_normalization'):
        with pytest.raises(ValueError):validate_specialized_preflight(entry,dict(report,**{key:None}),VARIANT)


@pytest.mark.parametrize('corrupt',[None,'policy_choice_verified','full_environment_resume_equal',
                                  'actual_policy_reduction','actual_advantage_normalization'])
def test_submission_validates_choice_proof_before_accessing_credentials(tmp_path,monkeypatch,corrupt):
    import hashlib
    from scripts.kaggle.submit_prepared_pilot import submit
    p=protocol();model=dict(id='parent',sha256=p['initial_sha256'])
    (tmp_path/'dataset').mkdir();(tmp_path/'job').mkdir()
    (tmp_path/'dataset/protocol.json').write_text(json.dumps(p))
    bootstrap=tmp_path/'job/train.py';bootstrap.write_text('# inert fixture\n')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    prep=dict(study_variant=VARIANT,kernel='owner/choice',bootstrap_sha256=digest(bootstrap),
              config=dict(owner='owner',models=[model],pins={}))
    (tmp_path/'preparation.json').write_text(json.dumps(prep))
    children=[dict(explicit_initial_model=model,exact_resume_16_to_32=True,
        air_exposure_verified=True,pilot_steps=409600,actual_focus='uniform',actual_spell_waste_weight=.3,
        spell_interaction_verified=True,actual_dormant_king_weight=0.,crown_validation_games=12,
        actual_tower_support_weight=.15,policy_choice_verified=True,full_environment_resume_equal=True,
        actual_policy_reduction=t['policy_reduction'],actual_advantage_normalization=t['advantage_normalization'])
        for t in p['cases']]
    if corrupt:children[-1][corrupt]=None
    proof=dict(exact_resume_16_to_32=True,frozen_sources_verified=True,
               preparation_sha256=digest(tmp_path/'preparation.json'),bootstrap_sha256=digest(bootstrap),
               children=children,validation_games=48)
    (tmp_path/'preflight.json').write_text(json.dumps(proof))
    def credential_boundary():raise RuntimeError('credential boundary reached; no credentials read')
    monkeypatch.setattr(Path,'home',credential_boundary)
    if corrupt:
        with pytest.raises(ValueError,match='choice'):submit(tmp_path,0)
    else:
        with pytest.raises(RuntimeError,match='credential boundary'):submit(tmp_path,0)
