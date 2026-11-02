"""Exercise the actual400-update pilot branch with real PPO and exact resume."""
import argparse
import importlib.util
import json
from pathlib import Path


def child(path, *, full=False):
    path=Path(path);spec=json.loads(path.read_text())
    spec.update(controller_full=full,pilot_steps=409600);path.write_text(json.dumps(spec,indent=2)+'\n')
    source=Path(__file__).with_name('preflight_policy_adaptation_pilot.py')
    module_spec=importlib.util.spec_from_file_location('exposure_shared',source)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module);module.child(str(path))
    import torch
    root=Path(spec['root']);trial=spec['trial'];p=root/'child_preflight.json';report=json.loads(p.read_text())
    target=4096000 if full else 409600
    args=report['original_full_command' if full else 'original_pilot_command'];blob=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    if (args[args.index('--target-environment-steps')+1]!=str(target)or args[args.index('--updates')+1]!=str(target//1024)or
            args[args.index('--continuous-focus')+1]!=trial['focus']or
            float(args[args.index('--spell-waste-weight')+1])!=trial['spell_waste_weight']or
            blob['config']['continuous_focus']!=trial['focus']or
            blob['config']['spell_waste_weight']!=trial['spell_waste_weight']or
            blob['config']['tower_support_weight']!=trial['tower_support_weight']or
            blob['environment_states'][0]['tower_support_weight']!=trial['tower_support_weight']):
        raise ValueError('Actual longer pilot treatment differs')
    report.update(air_exposure_verified=True,pilot_steps=None if full else target,full_steps=target if full else None,actual_focus=trial['focus'],
                  actual_spell_waste_weight=trial['spell_waste_weight'],actual_tower_support_weight=trial['tower_support_weight'])
    p.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared');p.add_argument('--child');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
