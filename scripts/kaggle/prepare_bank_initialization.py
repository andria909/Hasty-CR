"""Freeze a small mixed-skill BC ablation, separate from full-game PPO."""
import argparse
import ast
import io
import json
import re
from pathlib import Path
import shutil
import tarfile
from scripts.kaggle.prepare_motionfix_pilot import prepare as clone,sha
from scripts.attack.sequence_physics import require_current_physics


def prepare(source,data,output,slug='hastycr-bank-init1030'):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{5,65}',slug):raise ValueError('Valid distinct study slug required')
    root=Path(output);data=Path(data);m=json.loads((data/'manifest.json').read_text());require_current_physics(m)
    if m['schema']!='mixed-bank-public-sequences-v1' or sha(data/'examples.npz')!=m['data_sha256']:
        raise ValueError('Verified mixed development skills required')
    clone(source,root,slug)
    prep=json.loads((root/'preparation.json').read_text());c=prep['config']
    archive=root/'dataset/extensions.tar.gz'
    with tarfile.open(archive)as stream:payload={x.name:stream.extractfile(x).read()for x in stream.getmembers()if x.isfile()}
    original=json.loads((root/'dataset/manifest.json').read_text())['files']
    for name in ('scripts/attack/train_air_position_warmstart.py','scripts/attack/sequence_sampling.py',
                 'scripts/attack/train_bank_initialization_study.py','scripts/attack/sequence_physics.py',
                 'scripts/attack/sequence_execution.py','scripts/kaggle/local_artifact_transfer.py'):
        payload[name]=Path(name).read_bytes();c['extensions'][name]=sha(name)
        if name in original:c['replaced'][name]=original[name]
    with tarfile.open(archive,'w:gz')as stream:
        for name,raw in sorted(payload.items()):
            info=tarfile.TarInfo(name);info.size=len(raw);stream.addfile(info,io.BytesIO(raw))
    c['pins']['extensions.tar.gz']=sha(archive)
    for src,dst in [('examples.npz','skill_examples.npz'),('manifest.json','skill_manifest.json')]:
        shutil.copyfile(data/src,root/'dataset'/dst);c['pins'][dst]=sha(root/'dataset'/dst)
    parent=next(x for x in c['models']if x['id']=='parent')
    protocol=dict(schema='bank-initialization-study-v1',seeds=[392,393],updates_each=800,
        data_sha256=m['data_sha256'],initial_sha256=parent['sha256'],
        cases=[dict(id=name,kl_weight=kl,reset_actor_heads=reset)for name,kl,reset in
               [('retain01',.1,False),('retain001',.01,False),('retain0',0.,False),('zero01',.1,True)]],
        scope='BC-only local skill ablation; no environment steps, no PPO, no full-match promotion',
        full_game_teacher_ready=m['full_game_training_ready'],quality_accepted=False)
    (root/'dataset/initialization_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    c['pins']['initialization_protocol.json']=sha(root/'dataset/initialization_protocol.json')
    tree=ast.parse(Path('reports/defense_plan/bank_cannon_single_bootstrap_20261029.py').read_text())
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    kept=[];removed=0;commands=0
    for node in execute.body:
        text=ast.unparse(node)
        if isinstance(node,ast.Expr)and isinstance(node.value,ast.Call):
            call=node.value
            if (getattr(call.func,'attr',None)=='copyfile' and 'bank_builder.py'in text) or getattr(call.func,'attr',None)=='write_text':
                removed+=1;continue
            if getattr(call.func,'attr',None)=='run':
                kept+=ast.parse("data=project/'initialization_data'\ndata.mkdir()\nshutil.copyfile(inputs/'skill_examples.npz',data/'examples.npz')\nshutil.copyfile(inputs/'skill_manifest.json',data/'manifest.json')").body
                node=ast.parse("subprocess.run([sys.executable,'-u','-m','scripts.attack.train_bank_initialization_study','--data',str(data),'--initial',str(inputs/'parent.pt'),'--protocol',str(inputs/'initialization_protocol.json'),'--output','/kaggle/working/bank_initialization'],cwd=project,check=True)").body[0];commands+=1
        kept.append(node)
    if removed!=2 or commands!=1:raise ValueError('Unexpected frozen source loader')
    execute.body=kept;tree.body[-1].value.args=[ast.parse(repr(c),mode='eval').body]
    ast.fix_missing_locations(tree);(root/'job/train.py').write_text(ast.unparse(tree)+'\n')
    prep.update(study_variant='mixed_bank_initialization_v1',config=c,bootstrap_sha256=sha(root/'job/train.py'))
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n');return prep


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','data','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--slug',default='hastycr-bank-init1030')
    a=p.parse_args();r=prepare(a.source,a.data,a.output,a.slug);print(json.dumps(dict(kernel=r['kernel'],submitted=False)))
