"""Freeze a native Fireball slot-generalization pilot; never submit here."""
import argparse
import ast
from copy import deepcopy
import json
import io
from pathlib import Path
import shutil
import tarfile
from scripts.attack.sequence_physics import require_current_physics
from scripts.kaggle.prepare_native_latency_pilot import prepare as prepare_latency
from scripts.kaggle.local_artifact_transfer import file_sha as sha


def prepare(output, data, owner='andreiperelaiko'):
    root,data=Path(output),Path(data)
    labels=json.loads((data/'manifest.json').read_text())
    if labels['schema']!='native-fireball-development-examples-v1' or sha(data/'examples.npz')!=labels['data_sha256']:
        raise ValueError('Fresh native Fireball data required')
    require_current_physics(labels)
    prepare_latency(root,owner)
    prep=json.loads((root/'preparation.json').read_text());config=prep['config']
    dataset=root/'dataset';job=root/'job'
    source_manifest=json.loads((dataset/'manifest.json').read_text())
    for name,digest in labels['physics_pins'].items():
        if config['extensions'].get(name,source_manifest['files'].get(name))!=digest:
            raise ValueError('Packaged physics differs from sequential dataset: '+name)
    for src,dst in [('examples.npz','air_examples.npz'),('manifest.json','air_manifest.json')]:
        shutil.copyfile(data/src,dataset/dst);config['pins'][dst]=sha(dataset/dst)
    # prepare_latency freezes current warmstart; add its optional sampler dependency.
    name='sim/hand_slot_augmentation.py'
    target=dataset/'extensions'/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(name,target);config['extensions'][name]=sha(target)
    # Rebuild only this new package, not any submitted archive.
    archive=dataset/'extensions.tar.gz'
    if archive.exists():
        with tarfile.open(archive) as previous:
            payload={m.name:previous.extractfile(m).read() for m in previous.getmembers() if m.isfile()}
        payload[name]=target.read_bytes()
        if set(payload)!=set(config['extensions']):raise ValueError('Extension coverage differs')
        with tarfile.open(archive,'w:gz')as stream:
            for key in sorted(payload):
                info=tarfile.TarInfo(key);info.size=len(payload[key]);stream.addfile(info,io.BytesIO(payload[key]))
        config['pins']['extensions.tar.gz']=sha(archive)
    protocol=json.loads((dataset/'protocol.json').read_text())
    base=next(t for t in protocol['cases']if t['id']=='delay1350')
    cases=[]
    for name,updates,augment,kl in [('control',0,False,0.),('native800',800,False,0.),('slot800',800,True,0.),('slot800kl',800,True,.1)]:
        t=deepcopy(base);t.update(id=name,warmstart_updates=updates,warmstart_kl_weight=kl,
            slot_augmentation=augment,supervised_replay_weight=0.)
        cases.append(t)
    protocol.update(study_variant='native_fireball_slot_v1',cases=cases,pilot_seeds=[222,223],
        paired_comparison='Same balanced initializer/PPO/rewards/1350ms validation. Control, native Fireball BC800, online TRAIN-slot permutations, permutations with KL0.1. No supervised PPO replay.',
        rationale_source='reports/native_fireball_data_design_20261028.md',
        limitations='Narrow Fireball labels; local holdout18 dependent scenes. Full validation must check air defense and total outcomes; group hit counts alone do not justify promotion.')
    (dataset/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');config['pins']['protocol.json']=sha(dataset/'protocol.json')
    slug='hastycr-native-fireball-slot-inputs-20261028';kernel=owner+'/hastycr-native-fireball-slot-20261028'
    config.update(slug=slug,source_slug=slug)
    tree=ast.parse((job/'train.py').read_text());changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='run' and node.args and isinstance(node.args[0],ast.List):
            values=node.args[0].elts
            if any(isinstance(v,ast.Constant) and v.value=='scripts.attack.train_air_position_warmstart'for v in values):
                node.args[0]=ast.BinOp(left=node.args[0],op=ast.Add(),right=ast.parse("(['--slot-augmentation'] if trial.get('slot_augmentation') else [])",mode='eval').body);changed+=1
    if changed!=1:raise ValueError('Exactly one warmstart invocation required')
    tree.body[-1].value.args=[ast.parse(repr(config),mode='eval').body]
    ast.fix_missing_locations(tree);(job/'train.py').write_text(ast.unparse(tree)+'\n')
    for path,identity in [(dataset/'dataset-metadata.json',owner+'/'+slug),(job/'kernel-metadata.json',kernel)]:
        row=json.loads(path.read_text());row.update(id=identity,title=identity.split('/')[-1])
        if path.parent==job:row.update(dataset_sources=[owner+'/'+slug],enable_gpu=False)
        path.write_text(json.dumps(row,indent=2)+'\n')
    prep.update(kernel=kernel,config=config,bootstrap_sha256=sha(job/'train.py'),submitted=False,
        study_variant=protocol['study_variant'],requires_exact_preflight=True)
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return prep


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--data',required=True)
    a=p.parse_args();r=prepare(a.output,a.data);print(json.dumps(dict(kernel=r['kernel'],submitted=False)))
