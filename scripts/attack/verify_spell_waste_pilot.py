"""Audit empty-spell coefficients as well as the standard full PPO evidence."""
import json
import math
from pathlib import Path
from scripts.attack.verify_continuous_focus import verify


def audit_reward(config, records, expected):
    if config.get('spell_waste_weight')!=expected:
        raise ValueError('Empty spell training coefficient differs')
    if not records:raise ValueError('Empty spell validation missing')
    expected_old=.1 if expected is None else 0.
    if config['continuous_profile']['missed_fireball_penalty']!=expected_old:
        raise ValueError('Legacy Fireball penalty differs')
    if expected is None:return
    for row in records:
        value=row.get('empty_spell_elixir');penalty=row.get('empty_spell_penalty')
        costs=row.get('empty_spell_elixir_by_card',{})
        if (row.get('spell_waste_weight')!=expected or
            not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or
            not isinstance(penalty,(int,float)) or not math.isfinite(penalty) or
            any(k not in ('zap','fireball') or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for k,v in costs.items()) or
            not math.isclose(sum(costs.values()),value,abs_tol=1e-9) or
            not math.isclose(penalty,value*expected,rel_tol=1e-9,abs_tol=1e-9)):
            raise ValueError('Empty spell accounting differs')


def verify_spell_waste(collected, prepared):
    collected,prepared=Path(collected),Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text())
    if prep.get('study_variant')!='spell_waste_reward_v1':raise ValueError('Wrong spell study')
    result=verify(collected,prepared)
    protocol=json.loads((prepared/'dataset/protocol.json').read_text());audited=[]
    import torch
    for trial in protocol['cases']:
        for seed in protocol['pilot_seeds']:
            name=f"pilot_{trial['id']}_s{seed}";weight=trial['spell_waste_weight']
            config=json.loads((collected/name/'config.json').read_text())
            for line in (collected/name/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line)
                if 'validation' in row:audit_reward(config,row['validation']['records'],weight)
            selected=next(m for m in result['pilot_models'] if m['id']==name)
            blob=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
            if blob['config'].get('spell_waste_weight')!=weight:raise ValueError('Selected weight differs')
            audited.append(name)
    if len(audited)!=8:raise ValueError('Eight empty spell runs required')
    result['spell_waste_audit']=dict(runs=audited,verified=True)
    return result
