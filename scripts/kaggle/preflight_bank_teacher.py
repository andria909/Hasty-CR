"""Check actual50/50 teacher sampling as well as frozen PPO physics/resume."""
import argparse
import importlib.util
import json
from pathlib import Path


def child(path):
    import scripts.attack.train_air_position_warmstart as warm
    spec=json.loads(Path(path).read_text());original=warm.train;calls=[]
    def configured(*args,**kwargs):
        kwargs['sequence_wait_fraction']=.5
        result=original(*args,**kwargs);calls.append(result);return result
    warm.train=configured
    source=Path(__file__).with_name('preflight_mana_campaign.py')
    definition=importlib.util.spec_from_file_location('bank_physics_preflight',source)
    module=importlib.util.module_from_spec(definition);definition.loader.exec_module(module)
    module.child(path)
    p=Path(spec['root'])/'child_preflight.json';r=json.loads(p.read_text())
    if bool(calls)!=bool(spec['trial']['warmstart_updates']):raise ValueError('BC coverage differs')
    for command in r['original_warm_commands']:
        if float(command[command.index('--sequence-wait-fraction')+1])!=.5:raise ValueError('Frozen sampler differs')
    if calls and calls[0]['sequence_wait_fraction']!=.5:raise ValueError('Actual sampler differs')
    r.update(actual_warmstart_reports=calls,sequence_wait_fraction=.5)
    p.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child');p.add_argument('--prepared');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
