"""Freeze a paired bank-teacher BC/PPO study on current physics."""
import argparse
import ast
from copy import deepcopy
import json
import io
import tarfile
from pathlib import Path
import shutil
import numpy as np
from scripts.kaggle.prepare_motionfix_pilot import prepare as clone,sha

VARIANT='bank_counterfactual_v1'


def validate_data(data):
    data=Path(data);m=json.loads((data/'manifest.json').read_text())
    if m.get('schema')!='bank-counterfactual-labels-v1' or m.get('split')!='train' or sha(data/'examples.npz')!=m['data_sha256']:
        raise ValueError('Pinned counterfactual TRAIN data required')
    if m['command_delay_ms']!=1350 or len(m['seeds'])!=3:raise ValueError('Three seed groups and live delay required')
    for name,digest in m['physics_pins'].items():
        if sha(name)!=digest:raise ValueError('Teacher physics changed')
    if sha('scripts/attack/build_bank_counterfactuals.py')!=m['builder_sha256']:raise ValueError('Teacher source changed')
    if sha('scripts/attack/add_bank_urgency.py')!=m.get('augmentation_source_sha256'):raise ValueError('Urgency source changed')
    with np.load(data/'examples.npz',allow_pickle=False)as a:
        labels=a['success_masks'];masks=a['masks']
        if (labels&~masks).any() or not labels.any(-1).all():raise ValueError('Illegal/empty labels')
        stats={}
        for seed in m['seeds']:
            ids=[r['array_index']for r in m['records']if r['accepted'] and r['seed']==seed]
            waits=sum(bool(labels[i,0]) and labels[i].sum()==1 for i in ids)
            deploy=sum(not bool(labels[i,0])for i in ids)
            if not waits or not deploy:raise ValueError(f'Seed{seed} lacks discriminative WAIT or deploy examples')
            stats[str(seed)]=dict(wait_only=int(waits),deploy_only=int(deploy),examples=len(ids))
    return m,stats


def prepare(output,data):
    m,stats=validate_data(data)
    if not m.get('cannon_first_audited'):
        raise ValueError('Cannon-first alternatives must be audited before further teacher training')
    root=Path(output);data=Path(data)
    clone('tmp/kaggle/mana_campaign_20261029/balance',root,'hastycr-bank-teacher-1029')
    prep=json.loads((root/'preparation.json').read_text());config=prep['config']
    archive=root/'dataset/extensions.tar.gz'
    with tarfile.open(archive)as stream:
        payload={m.name:stream.extractfile(m).read()for m in stream.getmembers()if m.isfile()}
    original=json.loads((root/'dataset/manifest.json').read_text())['files']
    for name in ('scripts/attack/sequence_sampling.py','scripts/attack/train_air_position_warmstart.py'):
        payload[name]=Path(name).read_bytes();config['extensions'][name]=sha(name)
        if name in original:config['replaced'][name]=original[name]
    with tarfile.open(archive,'w:gz')as stream:
        for name,raw in sorted(payload.items()):
            info=tarfile.TarInfo(name);info.size=len(raw);stream.addfile(info,io.BytesIO(raw))
    config['pins']['extensions.tar.gz']=sha(archive)
    p=json.loads((root/'dataset/protocol.json').read_text());base=deepcopy(p['cases'][0])
    base.update(card_balance_weight=.03)
    p.update(schema='continuous-bank-teacher-pilot-v1',study_variant=VARIANT,pilot_seeds=[362,363],
        experiment='paired_counterfactual_teacher',teacher_data_sha256=m['data_sha256'],teacher_groups=stats,
        cases=[dict(base,id=name,warmstart_updates=updates,supervised_replay_weight=replay,sequence_wait_fraction=.5)
               for name,updates,replay in [('control',0,0.),('low',200,0.),('medium',800,0.),('high',800,.1)]],
        physics_pins={k:v for k,v in config['extensions'].items()if k.startswith('sim/')},
        paired_comparison='Same balanced parent, balance.03, current physics/rewards/delay; only teacher pretraining and retention vary.',
        limitations=m['limitations'],full='Only two independently eligible profiles may receive full continuation.')
    for src,dest in [('examples.npz','air_examples.npz'),('manifest.json','air_manifest.json')]:
        shutil.copyfile(data/src,root/'dataset'/dest);config['pins'][dest]=sha(root/'dataset'/dest)
    shutil.copyfile('scripts/attack/build_bank_counterfactuals.py',root/'dataset/bank_builder.py')
    config['pins']['bank_builder.py']=sha(root/'dataset/bank_builder.py')
    shutil.copyfile('scripts/attack/add_bank_urgency.py',root/'dataset/bank_urgency_builder.py')
    config['pins']['bank_urgency_builder.py']=sha(root/'dataset/bank_urgency_builder.py')
    (root/'dataset/protocol.json').write_text(json.dumps(p,indent=2)+'\n');config['pins']['protocol.json']=sha(root/'dataset/protocol.json')
    tree=ast.parse((root/'job/train.py').read_text())
    changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='run' and node.args and isinstance(node.args[0],ast.List):
            if any(isinstance(v,ast.Constant) and v.value=='scripts.attack.train_air_position_warmstart'for v in node.args[0].elts):
                node.args[0]=ast.BinOp(left=node.args[0],op=ast.Add(),right=ast.parse("['--sequence-wait-fraction',str(trial['sequence_wait_fraction'])]",mode='eval').body);changed+=1
    if changed!=1:raise ValueError('One warmstart command required')
    if 'scripts/attack/sequence_sampling.py' not in config['extensions']:raise ValueError('Sampler must be frozen')
    tree.body[-1].value.args=[ast.parse(repr(config),mode='eval').body]
    ast.fix_missing_locations(tree);(root/'job/train.py').write_text(ast.unparse(tree)+'\n')
    prep.update(study_variant=VARIANT,bootstrap_sha256=sha(root/'job/train.py'))
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return dict(kernel=prep['kernel'],data_groups=stats,submitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--data',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.output,a.data)))
