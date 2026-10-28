"""Actual frozen PPO/resume and validation checks for the economy campaign."""
import argparse
import importlib.util
import json
from pathlib import Path


def child(path):
    import torch
    import sim.train_attack_ppo as trainer
    spec = json.loads(Path(path).read_text()); trial = spec['trial']
    evaluate = trainer.evaluate
    checks = []

    def configured(*args, **kwargs):
        kwargs['env_options'] = dict(kwargs.get('env_options', {}),
            tower_support_weight=trial['tower_support_weight'],
            spell_waste_weight=trial['spell_waste_weight'], tactical_start_fraction=0.)
        result = evaluate(*args, **kwargs)
        for row in result['records']:
            assert row['normal_range_mode'] == 'edge_to_edge_v1'
            assert row['hog_sight_mode'] == 'side_clip_v1'
            assert row['cannon_outside_center'] == 0 and row['cannon_center_only'] == 1
            assert row['tower_damage_reward_mode'] == 'all_damage'
            assert row['tower_support_weight'] == .15 and row['spell_waste_weight'] == .1
            d = row['decision_economy']
            assert d['decisions'] > 0
            assert d['decisions'] == sum(d[k] for k in ('forced_wait', 'voluntary_wait', 'plays', 'invalid'))
            assert d['native_below4'] <= d['available_below4'] <= d['decisions']
        checks.append(len(result['records']))
        return result

    trainer.evaluate = configured
    file = Path(__file__).with_name('preflight_global_full_pair.py')
    module_spec = importlib.util.spec_from_file_location('mana_preflight_shared', file)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    module.child(path)
    root = Path(spec['root'])
    blob = torch.load(root/'whole/update_000002.pt', map_location='cpu', weights_only=False)
    for key in ('card_balance_weight', 'card_exploration_epsilon'):
        assert blob['config'][key] == trial[key]
    for key in ('reserve_potential_weight', 'hog_sight_clipping', 'cannon_center_only'):
        assert blob['config']['continuous_profile'][key] == trial[key]
    assert blob['environment_states'][0]['match'].battle.hog_sight_mode == 'side_clip_v1'
    assert checks == [12]
    path = root/'child_preflight.json'; report = json.loads(path.read_text())
    report.update(mana_contract_verified=True, actual_validation_checks=checks,
                  hog_sight_mode='side_clip_v1', economy_metrics_verified=True)
    path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--prepared'); p.add_argument('--child')
    args = p.parse_args()
    if args.child:
        child(args.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(args.prepared, child_script=__file__)
