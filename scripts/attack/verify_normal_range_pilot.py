"""Require actual corrected geometry throughout the paid-start pilot."""
import json
from pathlib import Path

from scripts.attack.verify_continuous_focus import verify
from scripts.attack.verify_tactical_starts_pilot import audit_exposure
from scripts.attack.verify_global_full_pair import audit_training_execution
from scripts.attack.evaluation_execution import CORRECTED,combat_modes,training_execution
from scripts.kaggle.prepare_normal_range_pilot import VARIANT,protocol_from,validate_protocol


def audit_geometry(config,records):
    if training_execution(config)!=CORRECTED:
        raise ValueError('Actual pilot training contract differs')
    if not records:raise ValueError('No geometry-tagged episodes')
    for row in records:
        if any(row.get(k)!=v for k,v in combat_modes(CORRECTED).items()):
            raise ValueError('Actual episode combat geometry differs')


def verify_normal_range(collected,prepared):
    collected,prepared=Path(collected),Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text())
    if prep.get('study_variant')!=VARIANT:raise ValueError('Wrong corrected-range study')
    p=json.loads((prepared/'dataset/protocol.json').read_text());validate_protocol(p)
    predecessor=json.loads((prepared/'dataset/predecessor_verified.json').read_text())
    expected=protocol_from(p,predecessor)
    for key in ('cases','constants','predecessor_trial','predecessor_selection_score'):
        if p[key]!=expected[key]:raise ValueError('Corrected pilot reward selection differs')
    result=verify(collected,prepared);audited=[]
    import torch
    for trial in p['cases']:
        for seed in p['pilot_seeds']:
            name=f"pilot_{trial['id']}_s{seed}";run=collected/name
            config=json.loads((run/'config.json').read_text());training=[];validation=[]
            for line in (run/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line);training.extend(row['episodes'])
                if 'validation'in row:validation.extend(row['validation']['records'])
            exposure=audit_exposure(config,training,validation,trial['tactical_start_fraction'],trial['tower_support_weight'])
            audit_geometry(config,training+validation)
            model=next(m for m in result['pilot_models']if m['id']==name)
            blob=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
            execution,count=audit_training_execution(config,blob,run/'metrics.jsonl')
            if execution!=CORRECTED or blob['config']['tactical_start_fraction']!=trial['tactical_start_fraction']:
                raise ValueError('Checkpoint geometry/start fraction differs')
            audited.append(dict(id=name,validation_geometry_records=count,**exposure))
    reference=json.loads((collected/'parent_reference.json').read_text())
    for row in reference['validation']['records']:
        if any(row.get(k)!=v for k,v in combat_modes(CORRECTED).items()):
            raise ValueError('Parent evaluated under different geometry')
    result['normal_range_audit']=dict(verified=True,execution=CORRECTED,runs=audited)
    return result
