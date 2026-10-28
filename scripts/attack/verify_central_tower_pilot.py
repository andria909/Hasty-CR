"""Verify actual budgets and all three priority contracts in every validation."""
import json
from pathlib import Path
from scripts.attack.verify_continuous_focus import verify
from scripts.attack.verify_tower_support_pilot import audit_reward as tower_audit
from scripts.attack.verify_spell_waste_pilot import audit_reward as spell_audit
from scripts.kaggle.prepare_central_tower_pilot import VARIANT,validate_protocol


def audit_priority(config,records,weight):
    profile=config['continuous_profile']
    if any(profile.get(k)!=v for k,v in dict(cannon_center_only=1,action_cooldown_ms=0,tower_damage_all=1).items()):
        raise ValueError('Checkpoint priority contract differs')
    tower_audit(config,records,weight);spell_audit(config,records,.1)
    for row in records:
        if (row.get('cannon_center_only')!=1 or row.get('cannon_outside_center')!=0 or
            row.get('action_cooldown_ms')!=0 or row.get('tower_damage_reward_mode')!='all_damage'):
            raise ValueError('Validation priority contract differs')


def verify_central(collected,prepared):
    collected,prepared=Path(collected),Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text())
    if prep.get('study_variant')!=VARIANT:raise ValueError('Wrong central study')
    p=json.loads((prepared/'dataset/protocol.json').read_text());validate_protocol(p)
    result=verify(collected,prepared);audited=[]
    import torch
    for trial in p['cases']:
        for seed in p['pilot_seeds']:
            name=f"pilot_{trial['id']}_s{seed}";c=json.loads((collected/name/'config.json').read_text());boundaries=[]
            for line in (collected/name/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line)
                if 'validation'in row:
                    audit_priority(c,row['validation']['records'],trial['tower_support_weight']);boundaries.append(row['update'])
            if not boundaries:raise ValueError('Missing validation evidence')
            model=next(m for m in result['pilot_models']if m['id']==name)
            blob=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
            if blob['config']['continuous_profile']!=c['continuous_profile']:raise ValueError('Selected profile differs')
            if blob['config']['tower_support_weight']!=trial['tower_support_weight']:raise ValueError('Selected reward differs')
            audited.append(dict(id=name,validation_boundaries=boundaries))
    result['central_tower_audit']=dict(verified=True,runs=audited)
    return result
