"""Run all four actual paired commands from the immutable package on CPU."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.kaggle.local_artifact_transfer import file_sha
from scripts.kaggle.preflight_global_full_pair import unpack,equal_state


def preflight(root):
    import torch
    root=Path(root).absolute();prep=json.loads((root/'preparation.json').read_text());config=prep['config']
    if file_sha(root/'job/train.py')!=prep['bootstrap_sha256']:raise ValueError('Bootstrap changed')
    out=root/'actual_preflight';out.mkdir();project=out/'project';unpack(root/'dataset',config,project)
    base,corrections=project/'control_data',project/'correction_data'
    for target,prefix in ((base,'skill'),(corrections,'correction')):
        target.mkdir()
        for src,dst in ((prefix+'_examples.npz','examples.npz'),(prefix+'_manifest.json','manifest.json')):
            shutil.copyfile(root/'dataset'/src,target/dst)
    tree=ast.parse((root/'job/train.py').read_text());execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    calls=[n for n in ast.walk(execute)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='run']
    if len(calls)!=1:raise ValueError('One frozen training command required')
    command=eval(compile(ast.Expression(calls[0].args[0]),'frozen_pair','eval'),dict(sys=sys,base=base,corrections=corrections,inputs=root/'dataset'))
    command[command.index('--output')+1]=str(out/'result');command.append('--smoke')
    with(out/'run.log').open('x')as log:
        subprocess.run(command,cwd=project,env=dict(os.environ,PYTHONPATH=str(project),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
            stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((out/'result/result.json').read_text())
    if not result['smoke']or len(result['runs'])!=4 or result['supervised_updates']!=8 or result['environment_steps']:
        raise ValueError('Paired CPU execution coverage differs')
    null_checks=[]
    if result['technical_identical']:
        for tag in ('001','01'):
            a=out/f'result/control_kl{tag}_s402';b=out/f'result/corrections_kl{tag}_s402'
            x=torch.load(a/'policy.pt',map_location='cpu',weights_only=False)['state_dict']
            y=torch.load(b/'policy.pt',map_location='cpu',weights_only=False)['state_dict']
            if not equal_state(x,y)or json.loads((a/'updates.json').read_text())!=json.loads((b/'updates.json').read_text()):
                raise ValueError('Identical-data paired training differs')
            null_checks.append(dict(kl=tag,all_tensors_exact=True,update_history_exact=True))
    proof=dict(schema='bank-correction-pair-preflight-v1',bootstrap_sha256=prep['bootstrap_sha256'],
        preparation_sha256=file_sha(root/'preparation.json'),result_sha256=file_sha(out/'result/result.json'),
        frozen_sources_verified=True,smoke_runs=4,supervised_updates=8,environment_steps=0,
        technical_identical=result['technical_identical'],null_checks=null_checks,
        full_training_accepted=False,submission_allowed=not result['technical_identical'])
    (root/'preflight.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(preflight(a.root)))
