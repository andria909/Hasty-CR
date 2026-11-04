"""CPU proof: actual time-limit stop, mock artifact transport, exact frozen resume.

This technical fixture cannot submit a job or qualify a model. Kaggle download
transport is simulated; the trainer, checkpoint, resume and comparison are real.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from scripts.kaggle import collect_global_full_recovery as collector
from scripts.kaggle.local_artifact_transfer import file_sha
from scripts.kaggle.preflight_global_full_pair import smoke_args,set_flag


COMPARE = '''
import importlib.util,json,sys,torch
from pathlib import Path
from sim.train_attack_ppo import validate_resume
torch.set_num_threads(1)
helper,whole,resumed,stopped,report=sys.argv[1:]
spec=importlib.util.spec_from_file_location('boundary_compare',helper)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
a=torch.load(whole,map_location='cpu',weights_only=False)
b=torch.load(resumed,map_location='cpu',weights_only=False)
c=torch.load(stopped,map_location='cpu',weights_only=False)
validate_resume(c,b['config'])
assert a['step']==b['step']==32 and c['step']==16
assert a['config']['target_environment_steps']==b['config']['target_environment_steps']==c['config']['target_environment_steps']==32
keys=('state_dict','optimiser','rng','sampler','environment_states','observations','infos','episode_returns','episode_counters','best_validation','step','update')
equal={k:module.equal_boundary(a[k],b[k])for k in keys}
if not all(equal.values()):raise ValueError('Resume differs: '+str(equal))
Path(report).write_text(json.dumps(dict(equal=equal,budget_unchanged=True,steps_before=16,steps_after=32),indent=2)+'\\n')
'''


def run(cases,output):
    if len(cases)!=2:raise ValueError('Two frozen technical cases required')
    output=Path(output).absolute();output.mkdir(parents=True,exist_ok=False)
    prepared=output/'prepared';prepared.mkdir()
    children=[];sources={};runs=[]
    for index,case in enumerate(map(lambda p:Path(p).absolute(),cases)):
        proof=json.loads((case/'child_preflight.json').read_text())
        if not proof.get('technical_fixture')or not proof.get('exact_resume_16_to_32'):
            raise ValueError('Existing frozen full technical preflight required')
        original=proof['original_full_command']
        if original[original.index('--target-environment-steps')+1]!='4096000':
            raise ValueError('Expected original full-budget command')
        project=case/'project';target=output/f'child_{index}';target.mkdir()
        command=smoke_args(original,target/'whole',32)
        stopped=set_flag(set_flag(command,'--output',target/'stopped'),'--wallclock-seconds',1e-9)
        env=dict(os.environ,PYTHONPATH=str(project))
        with(target/'training.log').open('x')as log:
            for cmd in(command,stopped):
                subprocess.run(cmd,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((target/'stopped/training_result.json').read_text())
        if result['target_met']or result['step']!=16 or result['target_environment_steps']!=32:
            raise ValueError('Actual time-limit interruption was not reproduced')
        child=dict(index=index,remote_output=f'global_full_child_{index}/continuous_focus',
                   selected=dict(trial=dict(id='technical')))
        prefix=child['remote_output']+'/full_technical_s64/'
        for name in('config.json','metrics.jsonl','training_result.json','update_000001.pt'):
            sources[prefix+name]=target/'stopped'/name
        children.append(child);runs.append((case,project,target,command,env))
    (prepared/'preparation.json').write_text(json.dumps(dict(kernel='technical/local-only',
        children=children,technical_fixture=True,submission_allowed=False),indent=2)+'\n')
    files={name:SimpleNamespace(url=name)for name in sources}
    def copy(url,target,**kwargs):shutil.copyfile(sources[url],target)
    with patch.object(collector,'list_outputs',return_value=(files,[])),patch.object(collector,'download_preserving',side_effect=copy):
        receipt=collector.collect(SimpleNamespace(kernels_status=lambda *a:'ERROR'),prepared,output/'collected')
    if receipt['complete']or not receipt['artifacts_complete']:raise ValueError('Recovery collection contract differs')
    reports=[]
    helper=Path(__file__).resolve().parents[1]/'attack/boundary_state.py'
    for index,(case,project,target,command,env)in enumerate(runs):
        entry=receipt['children'][index];checkpoint=output/'collected'/entry['latest_checkpoint']
        if entry['latest_update']!=1 or file_sha(checkpoint)!=file_sha(target/'stopped/update_000001.pt'):
            raise ValueError('Collected boundary differs')
        resumed=smoke_args(command,target/'resumed',32,checkpoint)
        with(target/'resume.log').open('x')as log:
            subprocess.run(resumed,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,'-c',COMPARE,str(helper),str(target/'whole/update_000002.pt'),
                str(target/'resumed/update_000002.pt'),str(checkpoint),str(target/'comparison.json')],
                cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        before=collector.logged_prefix((target/'stopped/metrics.jsonl').read_bytes())
        after=[json.loads(line)for line in(target/'resumed/metrics.jsonl').read_text().splitlines()]
        joined=before[:entry['latest_update']]+after
        if [r['update']for r in joined]!=[1,2]or[r['step']for r in joined]!=[16,32]:
            raise ValueError('Combined metrics contain a gap or duplicate')
        # New derived artifact; neither original log is overwritten.
        (target/'combined_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in joined))
        report=json.loads((target/'comparison.json').read_text())
        reports.append(dict(case=str(case),checkpoint_sha256=file_sha(checkpoint),
            actual_wallclock_interruption=True,combined_updates=[1,2],**report))
        print(json.dumps(dict(child=index,actual_stop_collect_resume_equal=True)),flush=True)
    result=dict(schema='full-recovery-collection-preflight-v1',technical_fixture=True,
        simulated_transport=True,actual_cuda=False,training_submitted=False,
        submission_allowed=False,quality_accepted=False,children=reports,
        scope='CPU 32-step proof; does not prove Kaggle terminal outputs exist or GPU exactness')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs=2,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.cases,a.output)
