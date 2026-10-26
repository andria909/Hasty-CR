"""Prepare two full runs from verified pilots without repeating their training.

No API calls. Existing frozen source, initial weights and profile selection are
retained. Only the controller's reference/pilot stages are replaced by pinned
verified outputs; the nested full-training function must remain AST-identical.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import shutil

from scripts.attack.verify_continuous_focus import verify
from scripts.kaggle.run_five_round_robin import sha


def full_only_bootstrap(source):
    tree=ast.parse(source)
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    original_run=ast.dump(next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run'))
    def assignment(name,node):
        return isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in node.targets)
    begin=next(i for i,n in enumerate(execute.body)if assignment('references',n))
    end=next(i for i,n in enumerate(execute.body)if assignment('protocol',n))
    execute.body[begin:end]=ast.parse("initial=json.loads((inputs/'initial_selection.json').read_text())\ninitial_model=next(m for m in config['models']if m['id']==initial['selected'])").body
    begin=next(i for i,n in enumerate(execute.body)if isinstance(n,ast.FunctionDef)and n.name=='pilots')
    end=next(i for i,n in enumerate(execute.body)if assignment('full',n))
    execute.body[begin:end]=ast.parse("selection=json.loads((inputs/'selection.json').read_text())\nassert selection['gate_met'] and len(selection['selected'])==2").body
    # The finished pilots are provenance, not newly consumed steps.
    for node in ast.walk(execute):
        if isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id=='dict':
            for kw in node.keywords:
                if kw.arg=='pilot_steps':kw.value=ast.Constant(0)
    remaining=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    if original_run!=ast.dump(remaining):raise ValueError('Full training function changed')
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)+'\n'


def prepare(pilot,output,owner):
    pilot=Path(pilot);output=Path(output)
    proof=verify(pilot/'collected',pilot)
    selection=proof['pilot_selection']
    if not selection['gate_met']or len(selection['selected'])!=2:
        raise ValueError('Two independently eligible profiles required; no full run prepared')
    original=json.loads((pilot/'preparation.json').read_text())
    code=(pilot/'job/train.py').read_text()
    if sha(pilot/'job/train.py')!=original['bootstrap_sha256']:raise ValueError('Pilot bootstrap changed')
    tree=ast.parse(code)
    # Remove only the final launch call; retain the exact frozen definitions.
    if not isinstance(tree.body[-1],ast.Expr)or not isinstance(tree.body[-1].value,ast.Call)or not isinstance(tree.body[-1].value.func,ast.Name)or tree.body[-1].value.func.id!='execute':
        raise ValueError('Unexpected frozen entrypoint')
    tree.body.pop();source=full_only_bootstrap(ast.unparse(tree))
    config=copy.deepcopy(original['config']);config['owner']=owner;config['allow_full_training']=True
    stem='hastycr-selected-full-'+sha(pilot/'job/train.py')[:12]
    slug=stem+'-inputs';kernel=owner+'/'+stem
    config.update(slug=slug,source_slug=slug)
    output.mkdir(parents=True,exist_ok=False);dataset=output/'dataset';dataset.mkdir();job=output/'job';job.mkdir()
    for name in ('source.tar.gz','manifest.json','extensions.tar.gz','protocol.json'):
        shutil.copyfile(pilot/'dataset'/name,dataset/name)
    for model in config['models']:
        if sha(pilot/'dataset'/model['file'])!=model['sha256']:raise ValueError('Initial weight changed')
        shutil.copyfile(pilot/'dataset'/model['file'],dataset/model['file'])
    for name,obj in [('selection.json',selection),('initial_selection.json',proof['initial_selection']),('pilot_verified.json',proof)]:
        (dataset/name).write_text(json.dumps(obj,indent=2)+'\n');config['pins'][name]=sha(dataset/name)
    source+='\nexecute('+repr(config)+')\n';compile(source,'train.py','exec');(job/'train.py').write_text(source)
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug,title=slug,licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title=kernel.split('/')[-1],code_file='train.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,dataset_sources=[owner+'/'+slug],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    result=dict(kernel=kernel,config=config,bootstrap_sha256=sha(job/'train.py'),pilot_kernel=proof['kernel'],submitted=False,full_steps=8192000,pilot_steps_repeated=0,quality_accepted=False)
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',required=True);p.add_argument('--output',required=True);p.add_argument('--owner',required=True)
    a=p.parse_args();r=prepare(a.pilot,a.output,a.owner);print(json.dumps(dict(kernel=r['kernel'],submitted=False,full_steps=r['full_steps'])))
