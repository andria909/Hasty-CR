"""Exact new sequence sampler plus shared PPO/resume technical preflight."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def child(path):
    spec=json.loads(Path(path).read_text());fraction=spec['trial'].get('sequence_wait_fraction')
    import scripts.attack.train_air_position_warmstart as warm
    original=warm.train
    calls=[]
    def configured(*args,**kwargs):
        kwargs['sequence_wait_fraction']=fraction
        result=original(*args,**kwargs);calls.append(result)
        return result
    warm.train=configured
    filename=Path(__file__).with_name('preflight_global_full_pair.py')
    module_spec=importlib.util.spec_from_file_location('shared_preflight',filename)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    module.child(path)
    report_path=Path(spec['root'])/'child_preflight.json';report=json.loads(report_path.read_text())
    for command in report['original_warm_commands']:
        if float(command[command.index('--sequence-wait-fraction')+1])!=fraction:
            raise ValueError('Frozen controller sampler differs from actual smoke')
    if bool(calls)!=bool(spec['trial']['warmstart_updates']):raise ValueError('Warmstart coverage differs')
    if calls and calls[0]['sequence_wait_fraction']!=fraction:raise ValueError('Actual sampler differs')
    report.update(sequence_wait_fraction=fraction,actual_warmstart_reports=calls)
    report_path.write_text(json.dumps(report,indent=2)+'\n')


def preflight(prepared):
    from scripts.kaggle.preflight_global_full_pair import unpack
    from scripts.kaggle.local_artifact_transfer import file_sha
    root=Path(prepared).absolute();prep=json.loads((root/'preparation.json').read_text());config=prep['config']
    if file_sha(root/'job/train.py')!=prep['bootstrap_sha256']:raise ValueError('Bootstrap changed')
    protocol=json.loads((root/'dataset/protocol.json').read_text());out=root/'actual_preflight';out.mkdir()
    reports=[]
    for i,trial in enumerate(protocol['cases']):
        target=out/f'child_{i}';target.mkdir();project=target/'project';unpack(root/'dataset',config,project)
        spec=dict(root=str(target),inputs=str(root/'dataset'),trial=trial,config=config,
            initial_selection=dict(selected=config['models'][0]['id']),bootstrap=str(root/'job/train.py'))
        path=target/'spec.json';path.write_text(json.dumps(spec,indent=2)+'\n')
        with(target/'run.log').open('x')as log:
            subprocess.run([sys.executable,str(Path(__file__).absolute()),'--child',str(path)],cwd=project,
                env=dict(os.environ,PYTHONPATH=str(project)),stdout=log,stderr=subprocess.STDOUT,check=True)
        reports.append(json.loads((target/'child_preflight.json').read_text()))
        print(json.dumps(dict(profile=trial['id'],passed=True)),flush=True)
    report=dict(frozen_sources_verified=True,exact_resume_16_to_32=True,children=reports,validation_games=48,
        actual_cuda=False,quality_accepted=False,bootstrap_sha256=prep['bootstrap_sha256'],
        preparation_sha256=file_sha(root/'preparation.json'))
    (root/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child');p.add_argument('--prepared');a=p.parse_args()
    if a.child:child(a.child)
    else:preflight(a.prepared)
