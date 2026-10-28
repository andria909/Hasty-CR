"""Exercise frozen tower-reward commands, exact resume and matching validation."""
import argparse
import importlib.util
import json
from pathlib import Path


def child(path):
    import torch
    import sim.train_attack_ppo as trainer
    spec=json.loads(Path(path).read_text());weight=spec['trial']['tower_support_weight']
    evaluate=trainer.evaluate
    def configured_evaluate(*args,**kwargs):
        kwargs['env_options']=dict(kwargs.get('env_options',{}),tower_support_weight=weight)
        return evaluate(*args,**kwargs)
    trainer.evaluate=configured_evaluate
    filename=Path(__file__).with_name('preflight_global_full_pair.py')
    module_spec=importlib.util.spec_from_file_location('shared_preflight',filename)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module);module.child(path)
    root=Path(spec['root']);report_path=root/'child_preflight.json';report=json.loads(report_path.read_text())
    command=report['original_full_command']
    if float(command[command.index('--tower-support-weight')+1])!=weight:
        raise ValueError('Frozen reward command mismatch')
    checkpoint=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    if checkpoint['config']['tower_support_weight']!=weight:
        raise ValueError('Actual trained coefficient mismatch')
    if checkpoint['environment_states'][0]['tower_support_weight']!=weight:
        raise ValueError('Actual environment coefficient mismatch')
    report.update(tower_support_weight=weight)
    report_path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared');p.add_argument('--child');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
