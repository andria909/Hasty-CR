"""Actual packaged slot augmentation/PPO/resume preflight; no submission."""
import argparse
import importlib.util
import json
from pathlib import Path

def child(path):
    spec=json.loads(Path(path).read_text());augment=spec['trial']['slot_augmentation']
    import scripts.attack.train_air_position_warmstart as warm
    original=warm.train
    calls=[]
    def configured(*args,**kwargs):
        kwargs['slot_augmentation']=augment
        result=original(*args,**kwargs);calls.append(result)
        return result
    warm.train=configured
    filename=Path(__file__).with_name('preflight_global_full_pair.py')
    module_spec=importlib.util.spec_from_file_location('shared_preflight',filename)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    module.child(path)
    report_path=Path(spec['root'])/'child_preflight.json';report=json.loads(report_path.read_text())
    for command in report['original_warm_commands']:
        if ('--slot-augmentation' in command)!=augment:
            raise ValueError('Frozen controller sampler differs from actual smoke')
    if bool(calls)!=bool(spec['trial']['warmstart_updates']):raise ValueError('Warmstart coverage differs')
    if calls and calls[0]['slot_augmentation']!=augment:raise ValueError('Actual sampler differs')
    report.update(slot_augmentation=augment,actual_warmstart_reports=calls)
    report_path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child');p.add_argument('--prepared');a=p.parse_args()
    if a.child:child(a.child)
    else:
        from scripts.kaggle.preflight_sequence_pilot import preflight
        preflight(a.prepared,child_script=__file__)
