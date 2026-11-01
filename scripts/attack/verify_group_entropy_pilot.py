"""Verify trained entropy coefficients, reported objectives and fixed rewards."""
import json
import math
from pathlib import Path

from scripts.attack.verify_crown_economy_pilot import verify_crown
from scripts.attack.verify_spell_waste_pilot import audit_reward
from scripts.kaggle.prepare_group_entropy_pilot import validate_protocol


def audit_entropy(row,weight):
    if row.get('spatial_weight')!=weight or row.get('optimizer_batches',0)<=0:
        raise ValueError('Actual entropy optimizer accounting missing')
    full,group,objective=(row.get(k)for k in ('full','group','objective'))
    if any(not isinstance(v,(int,float))or not math.isfinite(v)or v< -1e-5 for v in (full,group,objective)):
        raise ValueError('Nonfinite entropy accounting')
    if group>full+1e-5 or abs(objective-(weight*full+(1-weight)*group))>1e-5:
        raise ValueError('Entropy decomposition differs')


def verify_entropy(collected,prepared):
    root,prepared=Path(collected),Path(prepared)
    protocol=json.loads((prepared/'dataset/protocol.json').read_text());validate_protocol(protocol)
    prep=json.loads((prepared/'preparation.json').read_text())
    if len(prep['config']['models'])!=1 or prep['config']['allow_full_training']:
        raise ValueError('One initializer; no automatic full continuation')
    result=verify_crown(root,prepared,group_entropy=True)
    import torch
    audits=[]
    for trial in protocol['cases']:
        for seed in protocol['pilot_seeds']:
            name=f"pilot_{trial['id']}_s{seed}";run=root/name;weight=trial['spatial_entropy_weight']
            config=json.loads((run/'config.json').read_text())
            if config.get('spatial_entropy_weight')!=weight:raise ValueError('Configured entropy weight differs')
            updates=0
            for line in (run/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line);audit_entropy(row['entropy_components'],weight);updates+=1
                records=row['episodes']+row.get('validation',{}).get('records',[])
                if records:audit_reward(config,records,trial['spell_waste_weight'])
            model=next(m for m in result['pilot_models']if m['id']==name)
            blob=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
            if blob['config'].get('spatial_entropy_weight')!=weight:raise ValueError('Saved entropy coefficient differs')
            audits.append(dict(id=name,spatial_weight=weight,updates=updates))
    result['group_entropy_audit']=dict(verified=True,runs=audits)
    return result
