"""Audit longer pilot execution, rewards, final400 results and paid samples."""
import json
from pathlib import Path
from scripts.attack.verify_continuous_focus import verify,sha
from scripts.attack.verify_crown_economy_pilot import audit_rewards
from scripts.attack.verify_spell_waste_pilot import audit_reward
from scripts.attack.verify_global_full_pair import audit_training_execution
from scripts.attack.evaluation_execution import HOG_CLIPPED
from scripts.kaggle.prepare_air_exposure_pilot import validate_protocol,VARIANT


def audit_checkpoint_rewards(config,blob,trial):
    states=blob.get('environment_states',[])
    if not states:raise ValueError('Actual reward environment states required')
    for name in ('tower_support_weight','spell_waste_weight','crown_only_weight','dormant_king_weight'):
        expected=trial[name]
        if (config.get(name)!=expected or blob['config'].get(name)!=expected or
                any(state.get(name)!=expected for state in states)):
            raise ValueError('Checkpoint/environment reward differs: '+name)


def verify_exposure(collected,prepared,*,interaction=False,tower_credit=False,credit_assignment=False,policy_choice=False):
    root,prepared=Path(collected),Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text());p=json.loads((prepared/'dataset/protocol.json').read_text())
    validator=validate_protocol;variant=VARIANT;report_prefix='air_exposure'
    if sum((interaction,tower_credit,credit_assignment,policy_choice))>1:raise ValueError('One study variant required')
    if policy_choice:
        from scripts.kaggle.prepare_policy_choice_pilot import validate_protocol as validator,VARIANT as variant
        report_prefix='policy_choice'
    elif credit_assignment:
        from scripts.kaggle.prepare_credit_assignment_pilot import validate_protocol as validator,VARIANT as variant
        report_prefix='credit_assignment'
    elif tower_credit:
        from scripts.kaggle.prepare_tower_credit_pilot import validate_protocol as validator,VARIANT as variant
        report_prefix='tower_credit'
    elif interaction:
        from scripts.kaggle.prepare_spell_interaction_pilot import validate_protocol as validator,VARIANT as variant
        report_prefix='spell_interaction'
    validator(p)
    if policy_choice:
        for name,digest in p['training_sources'].items():
            if prep['config']['extensions'].get(name)!=digest:
                raise ValueError('Designed policy-choice source differs: '+name)
    models=prep['config']['models']
    if (prep['study_variant']!=variant or len(models)!=1 or models[0]['id']!='parent'or
            models[0]['sha256']!=p['initial_sha256']or
            sha(prepared/'dataset'/models[0]['file'])!=p['initial_sha256']):
        raise ValueError('Assigned BC800 initializer differs')
    result=verify(root,prepared);audited=[]
    import torch
    for trial in p['cases']:
        for seed in p['pilot_seeds']:
            name=f"pilot_{trial['id']}_s{seed}";run=root/name
            config=json.loads((run/'config.json').read_text())
            rows=[json.loads(x)for x in (run/'metrics.jsonl').read_text().splitlines()]
            validations=[r for r in rows if 'validation'in r]
            if [r['update']for r in validations]!=list(range(50,401,50)):
                raise ValueError('Every scheduled validation including final400 required')
            for r in rows:
                if policy_choice:
                    from scripts.attack.policy_choice_accounting import audit_update
                    audit_update(r,config,trial)
                if r['optimizer_controls']['value_only_warmup']!=(r['step']<=20480):
                    raise ValueError('Actual critic warmup differs')
                episodes=r['episodes']+r.get('validation',{}).get('records',[])
                if episodes:audit_rewards(config,episodes,trial);audit_reward(config,episodes,trial['spell_waste_weight'])
            selected=next(m for m in result['pilot_models']if m['id']==name)
            blob=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
            if policy_choice:
                from scripts.attack.policy_choice_accounting import audit_checkpoint
                audit_checkpoint(config,blob,trial)
            if credit_assignment:
                for key in ('gae_lambda','card_balance_weight','gamma'):
                    if config.get(key)!=trial[key] or blob['config'].get(key)!=trial[key]:
                        raise ValueError('Selected credit-assignment coefficient differs: '+key)
            audit_checkpoint_rewards(config,blob,trial)
            execution,count=audit_training_execution(config,blob,run/'metrics.jsonl')
            if execution!=HOG_CLIPPED:raise ValueError('Exposure execution mode differs')
            final=torch.load(run/'update_000400.pt',map_location='cpu',weights_only=False)
            if final['step']!=409600 or final['config']!=blob['config']:
                raise ValueError('Final400 checkpoint/config differs')
            final_records=validations[-1]['validation']['records']
            audited.append(dict(id=name,geometry_records=count,
                final_wins=sum(r['outcome']=='win'for r in final_records),
                final_defense_wins=sum(r['outcome']=='win'and r['task']=='defense'for r in final_records),
                final_validation={k:v for k,v in validations[-1]['validation'].items()if k!='records'},
                checkpoint100_validation={k:v for k,v in validations[1]['validation'].items()if k!='records'}))
            if policy_choice:
                audited[-1]['policy_choice_accounting']={k:sum(r['policy_decisions'][k]for r in rows)
                    for k in ('rollout_steps','choice_steps','forced_steps','optimizer_samples','choice_samples','no_choice_minibatches')}
                audited[-1]['kl_stopped_updates']=sum(r['optimizer_controls']['kl_stopped']for r in rows)
    from scripts.attack.compare_ppo_pilot_behavior import compare_records
    comparisons=[];eligible=[]
    for trial in p['cases']:
        stable=True
        for seed in p['pilot_seeds']:
            name=f"pilot_{trial['id']}_s{seed}"
            after=json.loads((root/(name+'_summary.json')).read_text())
            control=p['paired_controls'].get(trial['id'])
            reference=f'pilot_{control}_s{seed}_summary.json'if control else'parent_reference.json'
            before=json.loads((root/reference).read_text())
            compared=compare_records(before['validation']['records'],after['validation']['records'])
            passed=(after['selected_with_family_guard']and compared['all']['wins_delta']>=0
                    and compared['tasks']['defense']['wins_delta']>=0)
            stable=stable and passed
            comparisons.append(dict(id=name,reference=reference,passed=passed,comparison=compared))
        if stable:eligible.append(trial['id'])
    result[report_prefix+'_audit']=dict(verified=True,runs=audited)
    ranked=[r for r in result['pilot_selection']['ranking']if r['trial']['id']in eligible]
    result[report_prefix+'_selection']=dict(eligible_profiles=eligible,gate_met=len(eligible)>=2,
        selected=ranked[:4]if len(eligible)>=2 else[],
        seed_comparisons=comparisons,automatic_full_training=False)
    if credit_assignment or policy_choice:
        secondary=[]
        pairs=(p.get('secondary_comparisons',{}).items()if credit_assignment else
               [(r['candidate'],r['control'])for r in p.get('secondary_comparisons',[])])
        for candidate,control in pairs:
            for seed in p['pilot_seeds']:
                before=json.loads((root/f'pilot_{control}_s{seed}_summary.json').read_text())
                after=json.loads((root/f'pilot_{candidate}_s{seed}_summary.json').read_text())
                secondary.append(dict(candidate=candidate,control=control,seed=seed,
                    comparison=compare_records(before['validation']['records'],after['validation']['records'])))
        result[report_prefix+'_secondary_comparisons']=secondary
    result['promotion_blocked']=len(eligible)<2
    result['promotion_block_reason']='fewer_than_two_eligible_profiles' if len(eligible)<2 else None
    return result


def verify_interaction(collected,prepared):
    return verify_exposure(collected,prepared,interaction=True)


def verify_tower_credit(collected,prepared):
    return verify_exposure(collected,prepared,tower_credit=True)


def verify_credit_assignment(collected,prepared):
    return verify_exposure(collected,prepared,credit_assignment=True)


def verify_policy_choice(collected,prepared):
    return verify_exposure(collected,prepared,policy_choice=True)
