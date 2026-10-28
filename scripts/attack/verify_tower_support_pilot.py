"""Audit tower reward coefficients as well as the standard full PPO evidence."""
import json
import math
from pathlib import Path
from scripts.attack.verify_continuous_focus import verify


def audit_reward(config, records, expected):
    if config.get('tower_support_weight')!=expected:
        raise ValueError('Tower support training coefficient differs')
    if not records:raise ValueError('Tower support validation missing')
    for row in records:
        value=row.get('tower_support_paid_value');reward=row.get('tower_support_reward')
        if (row.get('tower_support_weight')!=expected or
            not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or
            not isinstance(reward,(int,float)) or not math.isfinite(reward) or
            not math.isclose(reward,value*expected,rel_tol=1e-9,abs_tol=1e-9)):
            raise ValueError('Tower support validation accounting differs')


def verify_tower_support(collected, prepared):
    collected,prepared=Path(collected),Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text())
    if prep.get('study_variant')!='tower_support_reward_v1':raise ValueError('Wrong tower study')
    result=verify(collected,prepared)
    protocol=json.loads((prepared/'dataset/protocol.json').read_text());audited=[]
    import torch
    for trial in protocol['cases']:
        for seed in protocol['pilot_seeds']:
            name=f"pilot_{trial['id']}_s{seed}";weight=trial['tower_support_weight']
            config=json.loads((collected/name/'config.json').read_text())
            for line in (collected/name/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line)
                if 'validation' in row:audit_reward(config,row['validation']['records'],weight)
            selected=next(m for m in result['pilot_models'] if m['id']==name)
            blob=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
            if blob['config'].get('tower_support_weight')!=weight:raise ValueError('Selected weight differs')
            audited.append(name)
    if len(audited)!=8:raise ValueError('Eight tower reward runs required')
    result['tower_support_audit']=dict(runs=audited,verified=True)
    return result
