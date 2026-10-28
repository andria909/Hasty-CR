"""Freeze a 2x2 Fireball BC/reward experiment without changing simulator code."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path

from scripts.kaggle.prepare_motionfix_pilot import prepare as clone, sha


VARIANT = 'fireball_reward_ablation_v1'
CASES = [('control', 0, .25), ('noaux', 0, 0.),
         ('slot800', 800, .25), ('slot800noaux', 800, 0.)]


def validate_protocol(protocol):
    if protocol.get('study_variant') != VARIANT or protocol.get('pilot_seeds') != [272, 273]:
        raise ValueError('Wrong Fireball reward experiment identity')
    rows = protocol['cases']
    if len(rows) != 4:
        raise ValueError('Four factorial cases required')
    common = None
    varying = {'id', 'warmstart_updates', 'slot_augmentation', 'fireball_value_reward'}
    for row, (name, updates, reward) in zip(rows, CASES):
        if (row['id'], row['warmstart_updates'], row['fireball_value_reward']) != (name, updates, reward):
            raise ValueError('Fireball reward factorial differs')
        if (row['slot_augmentation'] is not bool(updates) or row['warmstart_kl_weight'] != 0.
                or row['supervised_replay_weight'] != 0. or row['action_delay_ms'] != 1350
                or row['validation_action_delay_ms'] != 1350):
            raise ValueError('Unexpected warmstart/replay/timing settings')
        fixed = {key: value for key, value in row.items() if key not in varying}
        if common is not None and fixed != common:
            raise ValueError('Additional confounding coefficient changed')
        common = fixed


def prepare(source, output):
    source, root = Path(source), Path(output)
    original = json.loads((source/'dataset/protocol.json').read_text())
    if original.get('study_variant') != 'native_fireball_slot_v1':
        raise ValueError('Expected verified native Fireball cohort')
    labels = json.loads((source/'dataset/air_manifest.json').read_text())
    from scripts.attack.sequence_physics import require_current_physics
    require_current_physics(labels)
    clone(source, root, 'hastycr-fireball-reward-ablation-1028')
    prep = json.loads((root/'preparation.json').read_text())
    protocol = json.loads((root/'dataset/protocol.json').read_text())
    template = next(row for row in protocol['cases'] if row['id'] == 'control')
    cases = []
    for name, updates, reward in CASES:
        row = deepcopy(template)
        row.update(id=name, warmstart_updates=updates, slot_augmentation=bool(updates),
                   warmstart_kl_weight=0., supervised_replay_weight=0., fireball_value_reward=reward)
        cases.append(row)
    protocol.update(study_variant=VARIANT, cases=cases, pilot_seeds=[272, 273],
        paired_comparison='2x2: no BC versus slot-augmented BC800; Fireball auxiliary value coefficient .25 versus 0. Same physics, parent, data, timing and other rewards.',
        rationale_source='reports/fireball_reward_ablation_design_20261028.md',
        limitations='Removing auxiliary value does not remove tower damage reward or resolved-miss penalty. Raw hit frequency is not a success criterion. No independent test selection.')
    validate_protocol(protocol)
    path = root/'dataset/protocol.json'
    path.write_text(json.dumps(protocol, indent=2)+'\n')
    prep['config']['pins']['protocol.json'] = sha(path)
    path = root/'job/train.py'
    tree = ast.parse(path.read_text())
    tree.body[-1].value.args = [ast.parse(repr(prep['config']), mode='eval').body]
    ast.fix_missing_locations(tree)
    path.write_text(ast.unparse(tree)+'\n')
    prep.update(study_variant=VARIANT, bootstrap_sha256=sha(path))
    (root/'preparation.json').write_text(json.dumps(prep, indent=2)+'\n')
    return dict(kernel=prep['kernel'], submitted=False, pilot_steps=819200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output)))
