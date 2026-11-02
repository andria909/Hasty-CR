"""Audit longer pilot execution, rewards, final400 results and paid samples."""
import json
from pathlib import Path
from scripts.attack.verify_continuous_focus import verify,sha
from scripts.attack.verify_crown_economy_pilot import audit_rewards
from scripts.attack.verify_spell_waste_pilot import audit_reward
from scripts.attack.verify_global_full_pair import audit_training_execution
from scripts.attack.evaluation_execution import HOG_CLIPPED
from scripts.kaggle.prepare_air_exposure_pilot import validate_protocol,VARIANT


def verify_exposure(collected,prepared):
    root,prepared=Path(collected),Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text());p=json.loads((prepared/'dataset/protocol.json').read_text())
    validate_protocol(p)
    models=prep['config']['models']
    if (prep['study_variant']!=VARIANT or len(models)!=1 or models[0]['id']!='parent'or
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
                if r['optimizer_controls']['value_only_warmup']!=(r['step']<=20480):
                    raise ValueError('Actual critic warmup differs')
                episodes=r['episodes']+r.get('validation',{}).get('records',[])
                if episodes:audit_rewards(config,episodes,trial);audit_reward(config,episodes,trial['spell_waste_weight'])
            selected=next(m for m in result['pilot_models']if m['id']==name)
            blob=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
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
    result['air_exposure_audit']=dict(verified=True,runs=audited)
    ranked=[r for r in result['pilot_selection']['ranking']if r['trial']['id']in eligible]
    result['air_exposure_selection']=dict(eligible_profiles=eligible,gate_met=len(eligible)>=2,
        selected=ranked[:4]if len(eligible)>=2 else[],
        seed_comparisons=comparisons,automatic_full_training=False)
    result['promotion_blocked']=len(eligible)<2
    return result
