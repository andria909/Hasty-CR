"""Three paired, independent economy studies on the same pinned physics."""
import argparse
import ast
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile

from scripts.kaggle.prepare_motionfix_pilot import prepare as clone, sha

VARIANT = 'mana_decision_v1'
SEEDS = [342, 343]
GRIDS = dict(reserve=('reserve_potential_weight', [.1, .4, 1., 2.]),
             balance=('card_balance_weight', [.1, .03, .01, 0.]),
             wait=('card_exploration_epsilon', [0., .1, .25, .5]))
OWNERS = dict(reserve='andriashadrin', balance='vertel0', wait='andreiperelaiko')
PROFILE_KEYS = ('overflow_penalty', 'missed_fireball_penalty', 'fireball_value_reward',
    'reserve_potential_weight', 'own_hp_penalty', 'barbarian_train_fraction',
    'lane_reflection_fraction', 'action_cooldown_ms', 'cannon_center_only',
    'tower_damage_all', 'source_radius_reach', 'target_lock_in_range', 'hog_sight_clipping')


def protocol_from(previous, experiment):
    if experiment not in GRIDS:
        raise ValueError('Unknown economy study')
    base = deepcopy(previous['cases'][0])
    base.update(id='control', tactical_start_fraction=0., reserve_potential_weight=.1,
                card_balance_weight=.1, card_exploration_epsilon=0., hog_sight_clipping=1,
                tower_support_weight=.15, spell_waste_weight=.1)
    key, values = GRIDS[experiment]
    cases = [dict(base, **{'id': name, key: value})
             for name, value in zip(('control', 'low', 'medium', 'high'), values)]
    result = dict(schema='continuous-mana-decision-pilot-v1', study_variant=VARIANT,
        experiment=experiment, pilot_seeds=SEEDS, pilot_steps_each=102400,
        total_pilot_steps=819200, cases=cases, constants=base,
        initialization='Fixed balanced40d028 weights, fresh optimizer for every seed/profile',
        rationale_source='reports/balanced_offline_diagnosis_20261029.md',
        paired_comparison='Only '+key+' varies within this study; common control across all three studies.',
        evaluation_policy='Greedy learned logits; wait/card mixture is TRAIN-only, with matched PPO log probabilities.',
        limitations=['Hog river jump remains incorrect; this is an economy study on current physics.',
                    'No new live matches; synthetic validation is not a live win rate.',
                    'Validation is a reused development set, not an untouched test.'],
        full='Select qualified candidates across all three cohorts only after terminal artifact verification.')
    validate_protocol(result)
    return result


def validate_protocol(protocol):
    if (protocol.get('study_variant') != VARIANT or protocol.get('pilot_seeds') != SEEDS
            or protocol.get('pilot_steps_each') != 102400 or protocol.get('total_pilot_steps') != 819200):
        raise ValueError('Wrong study, seeds or budget')
    key, values = GRIDS[protocol['experiment']]
    rows = protocol['cases']
    if len(rows) != 4 or [r['id'] for r in rows] != ['control', 'low', 'medium', 'high']:
        raise ValueError('Four paired profiles required')
    common = {k: v for k, v in rows[0].items() if k not in ('id', key)}
    for row, value in zip(rows, values):
        if row[key] != value or {k: v for k, v in row.items() if k not in ('id', key)} != common:
            raise ValueError('Confounded economy comparison')
        fixed = dict(cannon_center_only=1, action_cooldown_ms=0, tower_damage_all=1,
            source_radius_reach=1, target_lock_in_range=0, hog_sight_clipping=1,
            tower_support_weight=.15, spell_waste_weight=.1, action_delay_ms=1350,
            validation_action_delay_ms=1350, tactical_start_fraction=0.,
            warmstart_updates=0, supervised_replay_weight=0., history_frames=1,
            gamma=.999, entropy_coef=.01)
        if any(row.get(k) != v for k, v in fixed.items()):
            raise ValueError('Shared physics/reward contract differs')


def prepare(source, output, experiment):
    source, root = Path(source), Path(output)
    previous = json.loads((source/'dataset/protocol.json').read_text())
    protocol = protocol_from(previous, experiment)
    slug = f'hastycr-mana-{experiment}-1029'
    clone(source, root, slug)
    prep = json.loads((root/'preparation.json').read_text())
    config = prep['config']
    owner = OWNERS[experiment]
    config.update(owner=owner, allow_full_training=False)
    # Refresh all simulator modules, and every explicit source extension.
    # New helper imports must not silently fall back to a stale base archive.
    base = json.loads((root/'dataset/manifest.json').read_text())['files']
    names = set(config['extensions']) | {str(p) for p in Path('sim').rglob('*.py')}
    names |= {name for name in base if name.startswith(('sim/', 'scripts/', 'data/'))
              and Path(name).is_file() and sha(name) != base[name]}
    archive = root/'dataset/extensions.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for name in sorted(names):
            path = Path(name)
            if path.is_symlink() or path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe frozen source')
            content = path.read_bytes()
            info = tarfile.TarInfo(name); info.size = len(content)
            stream.addfile(info, io.BytesIO(content))
    config['extensions'] = {name: sha(name) for name in sorted(names)}
    config['replaced'] = {name: base[name] for name in sorted(names) if name in base}
    config['pins']['extensions.tar.gz'] = sha(archive)
    protocol['physics_pins'] = {name: digest for name, digest in config['extensions'].items()
                                if name.startswith('sim/')}
    p = root/'dataset/protocol.json'; p.write_text(json.dumps(protocol, indent=2)+'\n')
    config['pins']['protocol.json'] = sha(p)
    path = root/'job/train.py'; tree = ast.parse(path.read_text())
    execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
    run = next(n for n in execute.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    i = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'args' for t in n.targets))
    run.body[i:i] = ast.parse("coefficients = json.loads(profile.read_text())\ncoefficients['hog_sight_clipping'] = trial['hog_sight_clipping']\nprofile.write_text(json.dumps(coefficients))").body
    reference_profile = {key: protocol['cases'][0][key] for key in PROFILE_KEYS}
    reference_profile.update(shaping_gamma=.999, decision_interval_ms=250)
    i = next(i for i, n in enumerate(execute.body) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'options' for t in n.targets))
    update = dict(continuous_profile=reference_profile, tower_support_weight=.15,
                  spell_waste_weight=.1, tactical_start_fraction=0.)
    execute.body[i+1:i+1] = ast.parse('options.update('+repr(update)+')').body
    tree.body[-1].value.args = [ast.parse(repr(config), mode='eval').body]
    ast.fix_missing_locations(tree); path.write_text(ast.unparse(tree)+'\n')
    kernel = owner+'/'+slug
    for name, identity in [('dataset/dataset-metadata.json', owner+'/'+config['slug']),
                           ('job/kernel-metadata.json', kernel)]:
        path = root/name; metadata = json.loads(path.read_text())
        metadata.update(id=identity, title=identity.split('/')[-1])
        if name.startswith('job/'):
            metadata['dataset_sources'] = [owner+'/'+config['slug']]
        path.write_text(json.dumps(metadata, indent=2)+'\n')
    prep.update(study_variant=VARIANT, experiment=experiment, kernel=kernel,
                bootstrap_sha256=sha(root/'job/train.py'))
    (root/'preparation.json').write_text(json.dumps(prep, indent=2)+'\n')
    return dict(kernel=kernel, root=str(root), pilot_runs=8, target_steps=819200, submitted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='tmp/kaggle/normal_range_pilot_20261028')
    p.add_argument('--output', required=True)
    p.add_argument('--experiment', choices=tuple(GRIDS), required=True)
    a = p.parse_args(); print(json.dumps(prepare(a.source, a.output, a.experiment)))
