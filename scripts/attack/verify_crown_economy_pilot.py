"""Verify actual reward coefficients, paid spell bins, physics and CUDA budgets."""
import json
from pathlib import Path

from scripts.attack.verify_continuous_focus import verify
from scripts.attack.verify_mana_campaign import audit_economy
from scripts.attack.verify_global_full_pair import audit_training_execution
from scripts.attack.evaluation_execution import HOG_CLIPPED, combat_modes
from scripts.kaggle.prepare_crown_economy_pilot import VARIANT, validate_protocol


def audit_rewards(config, rows, trial):
    for key in ('crown_only_weight','dormant_king_weight','spell_waste_weight','tower_support_weight'):
        if config.get(key)!=trial[key]:raise ValueError('Actual crown economy coefficient differs: '+key)
    if not rows:raise ValueError('Missing actual crown economy episodes')
    for row in rows:
        for key in ('crown_only_weight','dormant_king_weight','spell_waste_weight','tower_support_weight'):
            if row.get(key)!=trial[key]:raise ValueError('Episode crown economy coefficient differs: '+key)
        crown=row['nonlethal_crown_only_elixir'];king=row['dormant_at_request_king_only_elixir']
        if not 0<=king<=crown:raise ValueError('Invalid crown-only bins')
        expected=trial['crown_only_weight']*(crown-king)+trial['dormant_king_weight']*king
        if abs(row['spell_economy_penalty']-expected)>1e-7:raise ValueError('Crown penalty does not match paid disjoint bins')
        if row['cannon_outside_center'] or row['tower_damage_reward_mode']!='all_damage':
            raise ValueError('Cannon/tower contract differs')
        if any(row.get(k)!=v for k,v in combat_modes(HOG_CLIPPED).items()):
            raise ValueError('Crown study combat modes differ')
    audit_economy(rows)


def verify_crown(collected,prepared):
    collected,prepared=Path(collected),Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text())
    if prep.get('study_variant')!=VARIANT:raise ValueError('Wrong crown economy study')
    p=json.loads((prepared/'dataset/protocol.json').read_text());validate_protocol(p)
    result=verify(collected,prepared);audited=[]
    import torch
    for trial in p['cases']:
        for seed in p['pilot_seeds']:
            name=f"pilot_{trial['id']}_s{seed}";run=collected/name
            config=json.loads((run/'config.json').read_text());episodes=[]
            for line in (run/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line);episodes.extend(row['episodes'])
                if 'validation'in row:episodes.extend(row['validation']['records'])
            audit_rewards(config,episodes,trial)
            model=next(m for m in result['pilot_models']if m['id']==name)
            blob=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
            if any(blob['config'].get(k)!=trial[k]for k in ('crown_only_weight','dormant_king_weight')):
                raise ValueError('Selected checkpoint crown reward differs')
            execution,count=audit_training_execution(config,blob,run/'metrics.jsonl')
            if execution!=HOG_CLIPPED:raise ValueError('Crown checkpoint execution differs')
            audited.append(dict(id=name,episode_records=len(episodes),geometry_records=count))
    result['crown_economy_audit']=dict(verified=True,runs=audited)
    return result
