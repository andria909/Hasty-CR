"""Verify actual penalty accounting and PPO execution for the calibrated grid."""
import json
from pathlib import Path

from scripts.attack.verify_crown_economy_pilot import verify_crown
from scripts.attack.verify_spell_waste_pilot import audit_reward
from scripts.kaggle.prepare_calibrated_spell_pilot import validate_protocol


def verify_calibrated(collected, prepared):
    root, prepared = Path(collected), Path(prepared)
    protocol = json.loads((prepared/'dataset/protocol.json').read_text())
    validate_protocol(protocol)
    prep = json.loads((prepared/'preparation.json').read_text())
    if len(prep['config']['models']) != 1 or prep['config']['allow_full_training']:
        raise ValueError('One common initializer and manual full-training selection required')
    result = verify_crown(root, prepared, calibrated=True)
    for trial in protocol['cases']:
        for seed in protocol['pilot_seeds']:
            run = root/f"pilot_{trial['id']}_s{seed}"
            config = json.loads((run/'config.json').read_text())
            for line in (run/'metrics.jsonl').read_text().splitlines():
                row = json.loads(line)
                records = row['episodes'] + row.get('validation', {}).get('records', [])
                if records:
                    audit_reward(config, records, trial['spell_waste_weight'])
    result['calibrated_spell_cost_verified'] = True
    return result
