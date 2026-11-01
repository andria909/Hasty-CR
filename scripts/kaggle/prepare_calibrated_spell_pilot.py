"""PPO waste-cost ablation against the measured overflow incentive."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path

from scripts.kaggle.prepare_motionfix_pilot import prepare as clone, sha

VARIANT = 'calibrated_spell_cost_v1'
CASES = (('control', .1), ('neutral', .3), ('medium', .5), ('strong', 1.))


def validate_protocol(protocol):
    if (protocol.get('study_variant') != VARIANT or protocol.get('pilot_seeds') != [422, 423]
            or protocol.get('pilot_steps_each') != 102400 or protocol.get('total_pilot_steps') != 819200
            or len(protocol['cases']) != 4):
        raise ValueError('Fixed calibrated spell experiment required')
    common = None
    for row, (name, weight) in zip(protocol['cases'], CASES):
        if (row['id'], row['spell_waste_weight'], row['initial_id']) != (name, weight, 'parent'):
            raise ValueError('Spell grid or common initializer differs')
        fixed = {k: v for k, v in row.items() if k not in ('id', 'spell_waste_weight')}
        if common is not None and fixed != common:
            raise ValueError('Only empty spell cost may vary')
        common = fixed
        required = dict(overflow_penalty=.3, tower_support_weight=.15, crown_only_weight=0.,
                        dormant_king_weight=0., missed_fireball_penalty=0., action_delay_ms=1350,
                        validation_action_delay_ms=1350, action_cooldown_ms=0, cannon_center_only=1,
                        tower_damage_all=1, source_radius_reach=1, hog_sight_clipping=1,
                        target_lock_in_range=0, tactical_start_fraction=0., warmstart_updates=0,
                        supervised_replay_weight=0., card_balance_weight=.03, history_frames=1)
        if any(row.get(k) != v for k, v in required.items()):
            raise ValueError('Fixed reward/execution contract differs')


def prepare(source, output):
    root = Path(output)
    clone(source, root, 'hastycr-spell-cost1102')
    prep = json.loads((root/'preparation.json').read_text())
    if prep['study_variant'] != 'policy_adaptation_v1':
        raise ValueError('Expected frozen adaptation source')
    config = prep['config']
    # BC800 was selected before this ablation: equal wins to BC8000 seed402,
    # materially less empty spell expenditure. No new test selection.
    initial = next(m for m in config['models'] if m['id'] == 'bc800')
    config['models'] = [dict(initial, id='parent')]
    config.update(owner='andriashadrin', allow_full_training=False)
    protocol_path = root/'dataset/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    base = protocol['cases'][0]
    protocol.update(schema='continuous-calibrated-spell-cost-v1', study_variant=VARIANT,
                    experiment='empty_spell_cost', pilot_seeds=[422, 423],
                    cases=[dict(deepcopy(base), id=name, initial_id='parent', spell_waste_weight=weight)
                           for name, weight in CASES],
                    rationale_source='reports/empty_spell_incentive_20261102.md',
                    paired_comparison='Same BC800 seed402, PPO seeds, paid TRAIN streams and physics; only empty spell coefficient differs',
                    initialization='Every trial uses BC800 seed402 with a fresh optimizer',
                    limitations=['Idle reward diagnostic is not a win-rate result',
                                 'BC800 chosen on reused development validation, not test',
                                 'No automatic full continuation; compare useful spell casts and family wins too'])
    validate_protocol(protocol)
    protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    config['pins']['protocol.json'] = sha(protocol_path)
    tree = ast.parse((root/'job/train.py').read_text())
    tree.body[-1].value.args = [ast.parse(repr(config), mode='eval').body]
    ast.fix_missing_locations(tree)
    (root/'job/train.py').write_text(ast.unparse(tree)+'\n')
    kernel = config['owner']+'/hastycr-spell-cost1102'
    for file, identity in [('dataset/dataset-metadata.json', config['owner']+'/'+config['slug']),
                           ('job/kernel-metadata.json', kernel)]:
        path = root/file
        row = json.loads(path.read_text())
        row.update(id=identity, title=identity.split('/')[-1])
        if file.startswith('job/'):
            row.update(dataset_sources=[config['owner']+'/'+config['slug']], enable_gpu=True)
        path.write_text(json.dumps(row, indent=2)+'\n')
    prep.update(kernel=kernel, study_variant=VARIANT, bootstrap_sha256=sha(root/'job/train.py'))
    prep.pop('initialization_sources', None)
    (root/'preparation.json').write_text(json.dumps(prep, indent=2)+'\n')
    return dict(kernel=kernel, submitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output)))
