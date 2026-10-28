"""Freeze a pending-data diversity/slot BC pilot; no submission."""
import argparse
import ast
import io
import json
from pathlib import Path
import tarfile
from scripts.kaggle.prepare_pending_sequence_pilot import prepare as base_prepare,sha


def prepare(source,output,data):
    root=Path(output);base_prepare(source,root,data)
    prep=json.loads((root/'preparation.json').read_text());config=prep['config'];dataset=root/'dataset'
    labels=json.loads((dataset/'air_manifest.json').read_text())
    if labels.get('expansion_kind')!='fixed_teacher_new_train_hands_v1':raise ValueError('Expanded pending data required')
    archive=dataset/'extensions.tar.gz'
    with tarfile.open(archive)as stream:payload={m.name:stream.extractfile(m).read()for m in stream.getmembers()}
    for name in ('sim/pending_slot_augmentation.py','sim/hand_slot_augmentation.py'):
        payload[name]=Path(name).read_bytes();config['extensions'][name]=sha(name)
    with tarfile.open(archive,'w:gz')as stream:
        for name,content in sorted(payload.items()):
            info=tarfile.TarInfo(name);info.size=len(content);stream.addfile(info,io.BytesIO(content))
    config['pins']['extensions.tar.gz']=sha(archive)
    p=dataset/'protocol.json';protocol=json.loads(p.read_text());template=protocol['cases'][0];cases=[]
    for name,updates,augment in [('control',0,False),('diverse200',200,False),('slot200',200,True),('slot800',800,True)]:
        trial=dict(template,id=name,warmstart_updates=updates,slot_augmentation=augment,supervised_replay_weight=0.)
        cases.append(trial)
    protocol.update(study_variant='pending_diversity_slots_v1',cases=cases,pilot_seeds=[262,263],
        paired_comparison='Expanded TRAIN hands, allpending1350ms. Control,BC200,BC200withconsistent slots,BC800withslots. No PPO replay. Heldout unchanged.',
        limitations='One grouped local holdout; report deploy separately from WAIT; fullgamevalidation gate required.')
    p.write_text(json.dumps(protocol,indent=2)+'\n');config['pins']['protocol.json']=sha(p)
    owner='andreiperelaiko';slug='hastycr-pending-diversity-1028'
    config.update(owner=owner,slug=slug+'-inputs',source_slug=slug+'-inputs')
    p=root/'job/train.py';tree=ast.parse(p.read_text());changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='run' and node.args and isinstance(node.args[0],ast.List):
            if any(isinstance(v,ast.Constant) and v.value=='scripts.attack.train_air_position_warmstart'for v in node.args[0].elts):
                node.args[0]=ast.BinOp(left=node.args[0],op=ast.Add(),right=ast.parse("(['--slot-augmentation'] if trial['slot_augmentation'] else [])",mode='eval').body);changed+=1
    if changed!=1:raise ValueError('Warmstart command count differs')
    tree.body[-1].value.args=[ast.parse(repr(config),mode='eval').body];ast.fix_missing_locations(tree);p.write_text(ast.unparse(tree)+'\n')
    for file,identity in [('job/kernel-metadata.json',owner+'/'+slug),('dataset/dataset-metadata.json',owner+'/'+config['slug'])]:
        path=root/file;r=json.loads(path.read_text());r.update(id=identity,title=identity.split('/')[-1])
        if file.startswith('job/'):r.update(dataset_sources=[owner+'/'+config['slug']],enable_gpu=False)
        path.write_text(json.dumps(r,indent=2)+'\n')
    prep.update(kernel=owner+'/'+slug,config=config,bootstrap_sha256=sha(p),study_variant=protocol['study_variant'])
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n');return dict(kernel=prep['kernel'],submitted=False)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('source','output','data'):p.add_argument('--'+arg,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.data)))
