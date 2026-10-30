"""Execute the exact frozen BC module/arguments with a bounded CPU smoke."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.kaggle.preflight_global_full_pair import unpack
from scripts.kaggle.local_artifact_transfer import file_sha


def preflight(root):
    root=Path(root).absolute();prep=json.loads((root/'preparation.json').read_text());config=prep['config']
    if file_sha(root/'job/train.py')!=prep['bootstrap_sha256']:raise ValueError('Frozen bootstrap changed')
    out=root/'actual_preflight';out.mkdir();project=out/'project';unpack(root/'dataset',config,project)
    data=project/'initialization_data';data.mkdir()
    for src,dst in [('skill_examples.npz','examples.npz'),('skill_manifest.json','manifest.json')]:
        shutil.copyfile(root/'dataset'/src,data/dst)
    tree=ast.parse((root/'job/train.py').read_text());execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    calls=[n for n in ast.walk(execute)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='run']
    if len(calls)!=1:raise ValueError('One frozen child command required')
    command=eval(compile(ast.Expression(calls[0].args[0]),'frozen_bc_command','eval'),dict(sys=sys,data=data,inputs=root/'dataset'))
    command[command.index('--output')+1]=str(out/'result');command.append('--smoke')
    with(out/'run.log').open('x')as log:
        subprocess.run(command,cwd=project,env=dict(os.environ,PYTHONPATH=str(project),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((out/'result/result.json').read_text())
    if not result['smoke'] or len(result['runs'])!=4 or result['supervised_updates']!=8 or result['environment_steps']:
        raise ValueError('BC preflight coverage differs')
    for row in result['runs']:
        r=row['result']
        if r['actual_cuda'] is not None or r['heldout_seed']!=354 or r['train_seeds']!=[352,353]:
            raise ValueError('BC preflight device/fold contract differs')
        if set(r['final']['local_heldout']['by_skill'])!={'safe_bank_sequence','air_support_defense','fireball_group','spell_restraint'}:
            raise ValueError('Missing mixed skill coverage')
    proof=dict(schema='bank-initialization-preflight-v1',frozen_sources_verified=True,smoke_runs=4,
        supervised_updates=8,environment_steps=0,bootstrap_sha256=prep['bootstrap_sha256'],
        preparation_sha256=file_sha(root/'preparation.json'),result_sha256=file_sha(out/'result/result.json'),
        actual_cuda=False,quality_accepted=False)
    (root/'preflight.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(preflight(a.root)))
