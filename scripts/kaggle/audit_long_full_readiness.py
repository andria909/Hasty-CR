"""CPU-only rehearsal of each frozen long-pilot profile's full command.

This does not select candidates, create a submittable job, or spend GPU quota.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.kaggle.prepare_global_full_pair import isolated_child
from scripts.kaggle.preflight_global_full_pair import unpack,validate_specialized_preflight
from scripts.kaggle.local_artifact_transfer import file_sha


def audit(roots,output):
    output=Path(output).absolute();output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for origin in map(Path,roots):
        origin=origin.absolute();prep=json.loads((origin/'preparation.json').read_text());c=prep['config']
        if file_sha(origin/'job/train.py')!=prep['bootstrap_sha256']:raise ValueError('Frozen controller changed')
        protocol=json.loads((origin/'dataset/protocol.json').read_text());variant=protocol['study_variant']
        if variant=='air_exposure_v1':
            from scripts.kaggle.prepare_air_exposure_pilot import validate_protocol
        elif variant=='spell_interaction_v1':
            from scripts.kaggle.prepare_spell_interaction_pilot import validate_protocol
        else:raise ValueError('Expected a long pilot')
        validate_protocol(protocol)
        group=output/variant;group.mkdir()
        tree=ast.parse((origin/'job/train.py').read_text());tree.body.pop()
        original=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
        original_run=next(n for n in original.body if isinstance(n,ast.FunctionDef)and n.name=='run')
        for index,trial in enumerate(protocol['cases']):
            target=group/trial['id'];target.mkdir()
            project=target/'project';unpack(origin/'dataset',c,project)
            code=isolated_child(ast.unparse(tree),index%2)
            execute=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)and n.name=='execute')
            run=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run')
            if ast.dump(run)!=ast.dump(original_run):raise ValueError('Full trainer changed')
            bootstrap=target/'train.py';bootstrap.write_text(code)
            spec=dict(root=str(target),inputs=str(origin/'dataset'),trial=trial,config=c,
                initial_selection=dict(selected=c['models'][0]['id']),bootstrap=str(bootstrap),technical_fixture=True)
            path=target/'spec.json';path.write_text(json.dumps(spec,indent=2)+'\n')
            script=Path(__file__).absolute().with_name('preflight_long_economy_full.py')
            with(target/'run.log').open('x')as log:
                subprocess.run([sys.executable,str(script),'--child',str(path)],cwd=project,
                    env=dict(os.environ,PYTHONPATH=str(project)),stdout=log,stderr=subprocess.STDOUT,check=True)
            proof=json.loads((target/'child_preflight.json').read_text())
            validate_specialized_preflight(dict(selected=dict(trial=trial),config=c),proof,variant)
            reports.append(dict(variant=variant,trial=trial['id'],prepared=str(origin),
                bootstrap_sha256=prep['bootstrap_sha256'],full_run_ast_unchanged=True,proof=proof))
            print(json.dumps(dict(variant=variant,trial=trial['id'],passed=True)),flush=True)
    result=dict(schema='long-full-readiness-v1',checks=reports,validation_games=12*len(reports),
        actual_cuda=False,training_submitted=False,candidates_selected=False,quality_accepted=False,
        scope='Frozen full commands; real CPU16/32-step resume and full technical matches; no GPU full runs')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.prepared,a.output)
