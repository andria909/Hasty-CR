"""Verify budgets, actual new geometry, unchanged rewards and economy metrics."""
import json
from pathlib import Path

from scripts.attack.verify_continuous_focus import verify
from scripts.attack.verify_tactical_starts_pilot import audit_exposure
from scripts.attack.verify_global_full_pair import audit_training_execution
from scripts.attack.evaluation_execution import HOG_CLIPPED, combat_modes, training_execution
from scripts.kaggle.prepare_mana_campaign import validate_protocol, VARIANT


def audit_economy(rows):
    if not rows:
        raise ValueError('Missing economy metrics')
    for row in rows:
        counts = row.get('decision_economy', {})
        n = counts.get('decisions', 0)
        if (n <= 0 or sum(counts.get(k, -1) for k in ('forced_wait', 'voluntary_wait', 'plays', 'invalid')) != n
                or not 0 <= counts.get('native_below4', -1) <= counts.get('available_below4', -1) <= n):
            raise ValueError('Inconsistent decision economy counts')
        cards = counts.get('cards', {})
        if sum(c['in_hand'] for c in cards.values()) != 4*n:
            raise ValueError('Incomplete hand opportunity accounting')
        if sum(c['selected'] for c in cards.values()) != counts['plays']:
            raise ValueError('Chosen card count differs')
        if any(not 0 <= c['selected'] <= c['legal'] <= c['in_hand'] <= n for c in cards.values()):
            raise ValueError('Invalid card availability counts')


def verify_mana(collected, prepared):
    collected, prepared = Path(collected), Path(prepared)
    prep = json.loads((prepared/'preparation.json').read_text())
    if prep.get('study_variant') != VARIANT:
        raise ValueError('Wrong mana campaign')
    p = json.loads((prepared/'dataset/protocol.json').read_text()); validate_protocol(p)
    result = verify(collected, prepared)
    audited = []
    import torch
    for trial in p['cases']:
        for seed in p['pilot_seeds']:
            name = f"pilot_{trial['id']}_s{seed}"; run = collected/name
            config = json.loads((run/'config.json').read_text())
            training, validation = [], []
            for line in (run/'metrics.jsonl').read_text().splitlines():
                row = json.loads(line); training.extend(row['episodes'])
                if 'validation' in row:
                    validation.extend(row['validation']['records'])
            exposure = audit_exposure(config, training, validation, 0., .15)
            if training_execution(config) != HOG_CLIPPED:
                raise ValueError('Training physics differs')
            for row in training+validation:
                if any(row.get(k) != v for k, v in combat_modes(HOG_CLIPPED).items()):
                    raise ValueError('Episode physics differs')
            audit_economy(training+validation)
            model = next(m for m in result['pilot_models'] if m['id'] == name)
            blob = torch.load(model['checkpoint'], map_location='cpu', weights_only=False)
            execution, count = audit_training_execution(config, blob, run/'metrics.jsonl')
            if execution != HOG_CLIPPED:
                raise ValueError('Checkpoint physics differs')
            audited.append(dict(id=name, validation_geometry_records=count, **exposure))
    reference = json.loads((collected/'parent_reference.json').read_text())
    audit_economy(reference['validation']['records'])
    for row in reference['validation']['records']:
        if any(row.get(k) != v for k, v in combat_modes(HOG_CLIPPED).items()):
            raise ValueError('Parent reference physics differs')
    result['mana_decision_audit'] = dict(verified=True, experiment=p['experiment'],
                                        execution=HOG_CLIPPED, runs=audited)
    return result
