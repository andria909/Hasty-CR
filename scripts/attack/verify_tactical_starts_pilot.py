"""Verify paid TRAIN exposure and unmodified validation in the start-fraction study."""
from collections import Counter
import json
from pathlib import Path

from scripts.attack.verify_central_tower_pilot import audit_priority
from scripts.attack.verify_continuous_focus import verify
from scripts.kaggle.prepare_tactical_starts_pilot import VARIANT, validate_protocol, protocol_from


def audit_exposure(config, training, validation, fraction, weight):
    if config.get('tactical_start_fraction') != fraction:
        raise ValueError('Actual tactical-start fraction differs')
    audit_priority(config, training+validation, weight)
    for row in validation:
        if row.get('tactical_start_fraction') != fraction or row.get('tactical_start') is not None:
            raise ValueError('Validation includes tactical starts or lacks metadata')
    counts=Counter();masks=Counter();families=Counter()
    for row in training:
        if row.get('tactical_start_fraction') != fraction or row.get('split') != 'train':
            raise ValueError('Training exposure contract differs')
        start=row.get('tactical_start')
        counts['tactical' if start else 'ordinary']+=1
        if not start:continue
        if start.get('schema')!='paid-tactical-start-v1' or not start.get('synthetic_history'):
            raise ValueError('Unidentified tactical setup')
        if fraction==0:raise ValueError('Control has tactical starts')
        cycle=start['own_hand']+start['own_queue']
        if len(cycle)!=8 or len(set(cycle))!=8:raise ValueError('Invalid own hand/cycle')
        setup=start['enemy_setup']
        costs=[r['elixir_before']-r['elixir_after'] for r in setup]
        if not 1<=len(setup)<=2 or any(c<=0 or c%1000 for c in costs) or sum(costs)>10000:
            raise ValueError('Unpaid or over-budget starting push')
        masks[str(start['own_princess_mask'])]+=1;families[start['family']]+=1
    if not training or not counts['ordinary'] or (fraction and not counts['tactical']):
        raise ValueError('Missing actual ordinary/tactical training exposure')
    return dict(completed_episodes=len(training),counts=dict(counts),
                own_princess_masks=dict(masks),families=dict(families))


def verify_tactical(collected, prepared):
    collected,prepared=Path(collected),Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text())
    if prep.get('study_variant')!=VARIANT:raise ValueError('Wrong tactical-start study')
    p=json.loads((prepared/'dataset/protocol.json').read_text());validate_protocol(p)
    predecessor=json.loads((prepared/'dataset/predecessor_verified.json').read_text())
    expected=protocol_from(p,predecessor)
    if any(p[k]!=expected[k] for k in ('cases','predecessor_trial','predecessor_selection_score')):
        raise ValueError('Pilot differs from the verified predecessor selection')
    result=verify(collected,prepared);audited=[]
    import torch
    for trial in p['cases']:
        for seed in p['pilot_seeds']:
            name=f"pilot_{trial['id']}_s{seed}"
            config=json.loads((collected/name/'config.json').read_text())
            training=[];validation=[]
            for line in (collected/name/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line);training.extend(row['episodes'])
                if 'validation' in row:validation.extend(row['validation']['records'])
            exposure=audit_exposure(config,training,validation,trial['tactical_start_fraction'],trial['tower_support_weight'])
            model=next(m for m in result['pilot_models'] if m['id']==name)
            blob=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
            if blob['config']['tactical_start_fraction']!=trial['tactical_start_fraction']:
                raise ValueError('Selected checkpoint start fraction differs')
            audited.append(dict(id=name,**exposure))
    result['tactical_starts_audit']=dict(verified=True,runs=audited)
    return result
