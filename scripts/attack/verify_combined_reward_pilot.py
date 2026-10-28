"""Audit both shaping terms on every validation boundary and selected weight."""
import json
from pathlib import Path
from scripts.attack.verify_continuous_focus import verify
from scripts.attack.verify_tower_support_pilot import audit_reward as audit_tower
from scripts.attack.verify_spell_waste_pilot import audit_reward as audit_spell
from scripts.kaggle.prepare_combined_reward_pilot import validate_protocol,VARIANT


def verify_combined(collected,prepared):
    collected,prepared=Path(collected),Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text())
    if prep.get('study_variant')!=VARIANT:raise ValueError('Wrong combined reward study')
    protocol=json.loads((prepared/'dataset/protocol.json').read_text());validate_protocol(protocol)
    result=verify(collected,prepared);audited=[]
    import torch
    for trial in protocol['cases']:
        for seed in protocol['pilot_seeds']:
            name=f"pilot_{trial['id']}_s{seed}"
            config=json.loads((collected/name/'config.json').read_text());boundaries=[]
            for line in (collected/name/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line)
                if 'validation'in row:
                    audit_tower(config,row['validation']['records'],trial['tower_support_weight'])
                    audit_spell(config,row['validation']['records'],trial['spell_waste_weight'])
                    boundaries.append(row['update'])
            if not boundaries:raise ValueError('No paired reward validation evidence')
            selected=next(m for m in result['pilot_models']if m['id']==name)
            blob=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
            for key in ('tower_support_weight','spell_waste_weight'):
                if blob['config'].get(key)!=trial[key]:raise ValueError('Selected combined reward differs')
            audited.append(dict(id=name,validation_boundaries=boundaries))
    if len(audited)!=8:raise ValueError('Eight combined reward runs required')
    result['combined_reward_audit']=dict(runs=audited,verified=True)
    return result
