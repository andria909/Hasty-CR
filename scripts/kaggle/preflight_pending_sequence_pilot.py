"""Exercise frozen pending BC/replay commands and matching full-game validation."""
import argparse
import json
from pathlib import Path


def child(path):
    import scripts.attack.train_air_position_warmstart as warm
    import importlib.util
    file=Path(__file__).with_name('preflight_pending_pilot.py')
    module_spec=importlib.util.spec_from_file_location('pending_preflight',file)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    pending_child=module.child
    spec=json.loads(Path(path).read_text());original=warm.train;calls=[]
    def configured(*args,**kwargs):
        kwargs.update(pending_inputs=True,sequence_wait_fraction=spec['trial']['sequence_wait_fraction'])
        r=original(*args,**kwargs);calls.append(r);return r
    warm.train=configured
    pending_child(path)
    root=Path(spec['root']);report_path=root/'child_preflight.json';r=json.loads(report_path.read_text())
    for command in r['original_warm_commands']:
        if '--pending-inputs'not in command or command[command.index('--sequence-wait-fraction')+1]!='0.25':
            raise ValueError('Frozen warm command differs')
    if bool(calls)!=bool(spec['trial']['warmstart_updates']):raise ValueError('Missing warmstart')
    import torch
    checkpoint=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    weight=spec['trial']['supervised_replay_weight']
    if checkpoint['config']['supervised_replay_weight']!=weight:raise ValueError('Actual replay coefficient differs')
    metadata=checkpoint['config']['supervised_replay_metadata']
    if weight and (not metadata['pending_inputs'] or metadata['sequence_wait_fraction']!=.25):
        raise ValueError('Actual replay sampler differs')
    r.update(warmstart_reports=calls,supervised_replay_weight=weight,supervised_replay_metadata=metadata)
    report_path.write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared');p.add_argument('--child');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
