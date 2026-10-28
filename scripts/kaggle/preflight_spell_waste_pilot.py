"""Exercise frozen empty-spell commands, exact resume and matching validation."""
import argparse
import importlib.util
import json
from pathlib import Path


def child(path):
    import torch
    import sim.train_attack_ppo as trainer
    spec=json.loads(Path(path).read_text());weight=spec['trial']['spell_waste_weight']
    evaluate=trainer.evaluate
    def configured_evaluate(*args,**kwargs):
        kwargs['env_options']=dict(kwargs.get('env_options',{}),spell_waste_weight=weight)
        return evaluate(*args,**kwargs)
    trainer.evaluate=configured_evaluate
    filename=Path(__file__).with_name('preflight_global_full_pair.py')
    module_spec=importlib.util.spec_from_file_location('shared_preflight',filename)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module);module.child(path)
    root=Path(spec['root']);report_path=root/'child_preflight.json';report=json.loads(report_path.read_text())
    command=report['original_full_command']
    captured=float(command[command.index('--spell-waste-weight')+1]) if '--spell-waste-weight' in command else None
    if captured!=weight:raise ValueError('Frozen reward command mismatch')
    checkpoint=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    if checkpoint['config']['spell_waste_weight']!=weight:
        raise ValueError('Actual trained coefficient mismatch')
    if checkpoint['environment_states'][0].get('spell_waste_weight')!=weight:
        raise ValueError('Actual environment coefficient mismatch')
    expected_old=.1 if weight is None else 0.
    if checkpoint['config']['continuous_profile']['missed_fireball_penalty']!=expected_old:
        raise ValueError('Missing or doubled old Fireball penalty')
    report.update(spell_waste_weight=weight,old_fireball_miss_weight=expected_old)
    report_path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared');p.add_argument('--child');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
