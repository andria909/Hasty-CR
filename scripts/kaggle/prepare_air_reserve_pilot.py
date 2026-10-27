"""Paired joint-skills ablation with successful wait-then-deploy TRAIN labels."""
import argparse
import ast
import json
from pathlib import Path
import shutil

from scripts.kaggle.prepare_continuous_spell_study import prepare as prepare_joint
from scripts.kaggle.run_five_round_robin import sha


def prepare(output, owner='andriashadrin', data='data/air_reserve_examples_20261027', variant='reserve'):
    if variant not in ('reserve', 'alternatives'):
        raise ValueError('Unknown demonstration variant')
    root = Path(output)
    data = Path(data)
    manifest = json.loads((data / 'manifest.json').read_text())
    if manifest['split'] != 'train' or sha(data / 'examples.npz') != manifest['data_sha256']:
        raise ValueError('Verified TRAIN sequence data required')
    prepare_joint(root, owner, 'jointskills')
    prep = json.loads((root / 'preparation.json').read_text())
    config = prep['config']
    stem = 'hastycr-air-reserve-pilot' if variant == 'reserve' else 'hastycr-air-alternatives-pilot'
    slug = stem + '-inputs-20261027'
    kernel = owner + '/' + stem + '-20261027'
    config.update(slug=slug, source_slug=slug)
    for original, target in [('examples.npz', 'air_examples.npz'), ('manifest.json', 'air_manifest.json')]:
        shutil.copyfile(data / original, root / 'dataset' / target)
        config['pins'][target] = sha(root / 'dataset' / target)
    protocol_path = root / 'dataset/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol.update(study_variant=('air_reserve_wait_then_deploy_v1' if variant == 'reserve' else 'air_cannon_alternatives_v1'),
        rationale_source=('reports/defense_plan/balanced_air_response_trace_20261027.json' if variant == 'reserve' else 'reports/defense_plan/balloon_combinations_20261027.json'),
        rationale_split='fresh_train_full_games',
        paired_comparison='Same profiles/seeds/physics/rewards as joint-skills; only TRAIN demonstration dataset differs',
        limitations='Local successes; full-game validation still required; no action override')
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    config['pins']['protocol.json'] = sha(protocol_path)
    for name, identity in [('dataset/dataset-metadata.json', owner + '/' + slug),
                           ('job/kernel-metadata.json', kernel)]:
        path = root / name
        row = json.loads(path.read_text())
        row.update(id=identity, title=identity.split('/')[-1])
        if name.startswith('job/'):
            row['dataset_sources'] = [owner + '/' + slug]
        path.write_text(json.dumps(row, indent=2) + '\n')
    path = root / 'job/train.py'
    tree = ast.parse(path.read_text())
    if not isinstance(tree.body[-1], ast.Expr) or not isinstance(tree.body[-1].value, ast.Call):
        raise ValueError('Unexpected controller launch')
    tree.body[-1].value.args = [ast.parse(repr(config), mode='eval').body]
    ast.fix_missing_locations(tree)
    path.write_text(ast.unparse(tree) + '\n')
    prep.update(kernel=kernel, config=config, bootstrap_sha256=sha(path),
                paired_previous_study='joint_skills_pilot_20261027', study_variant=protocol['study_variant'])
    (root / 'preparation.json').write_text(json.dumps(prep, indent=2) + '\n')
    return dict(kernel=kernel, submitted=False, pilot_steps=819200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--owner', default='andriashadrin')
    parser.add_argument('--data', default='data/air_reserve_examples_20261027')
    parser.add_argument('--variant', choices=('reserve','alternatives'), default='reserve')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.owner, args.data, args.variant)))
