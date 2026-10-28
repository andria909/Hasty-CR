"""Freeze paired pending-sequence BC/replay study; no GPU submission."""
import argparse
import ast
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
from scripts.kaggle.prepare_motionfix_pilot import prepare as refresh,sha


def prepare(source,output,data):
    root=Path(output)
    refresh(source,root,'hastycr-pending-sequence-1028',data)
    prep=json.loads((root/'preparation.json').read_text());config=prep['config'];dataset=root/'dataset'
    labels=json.loads((dataset/'air_manifest.json').read_text())
    if labels.get('schema')!='late-air-pending-sequence-labels-v1':raise ValueError('Pending sequences required')
    archive=dataset/'extensions.tar.gz'
    with tarfile.open(archive) as stream:
        payload={m.name:stream.extractfile(m).read() for m in stream.getmembers() if m.isfile()}
    base=json.loads((dataset/'manifest.json').read_text())
    names=('sim/train_attack_ppo.py','sim/supervised_replay.py','scripts/attack/train_air_position_warmstart.py',
           'scripts/attack/sequence_sampling.py','scripts/attack/sequence_physics.py','scripts/kaggle/local_artifact_transfer.py')
    for name in names:
        payload[name]=Path(name).read_bytes();config['extensions'][name]=sha(name)
        if name in base['files']:config['replaced'][name]=base['files'][name]
    with tarfile.open(archive,'w:gz') as stream:
        for name,content in sorted(payload.items()):
            info=tarfile.TarInfo(name);info.size=len(content);stream.addfile(info,io.BytesIO(content))
    config['pins']['extensions.tar.gz']=sha(archive)
    path=dataset/'protocol.json';protocol=json.loads(path.read_text())
    template=next(t for t in protocol['cases'] if t['id']=='pending1350')
    cases=[]
    for name,bc,replay in [('control',0,0.),('bc800',800,0.),('replay100',800,1.),('replay500',800,5.)]:
        trial=deepcopy(template);trial.update(id=name,pending_inputs=True,warmstart_updates=bc,
            sequence_wait_fraction=.25,warmstart_kl_weight=.1,supervised_replay_weight=replay)
        cases.append(trial)
    protocol.update(study_variant='pending_sequence_retention_v1',cases=cases,pilot_seeds=[252,253],
        paired_comparison='All24/313 own-pending inputs,1350ms delay/currentphysics/same rewards. No BC,BC800,BC800+replay1,BC800+replay5.',
        limitations='24 correlated teacher sequences from three TRAIN seeds; entire largest seed held out. Full-game validation required; no proof of live parity.')
    path.write_text(json.dumps(protocol,indent=2)+'\n');config['pins']['protocol.json']=sha(path)
    path=root/'job/train.py';tree=ast.parse(path.read_text());changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='run' and node.args and isinstance(node.args[0],ast.List):
            if any(isinstance(v,ast.Constant) and v.value=='scripts.attack.train_air_position_warmstart' for v in node.args[0].elts):
                node.args[0].elts += [ast.Constant('--pending-inputs'),ast.Constant('--sequence-wait-fraction'),ast.Constant('0.25')];changed+=1
    if changed!=1:raise ValueError('Exactly one warmstart command required')
    tree.body[-1].value.args=[ast.parse(repr(config),mode='eval').body];ast.fix_missing_locations(tree)
    path.write_text(ast.unparse(tree)+'\n')
    meta=root/'job/kernel-metadata.json';m=json.loads(meta.read_text());m['enable_gpu']=False;meta.write_text(json.dumps(m,indent=2)+'\n')
    prep.update(config=config,bootstrap_sha256=sha(path),study_variant=protocol['study_variant'],submitted=False)
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return dict(root=str(root),kernel=prep['kernel'],submitted=False)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('source','output','data'):p.add_argument('--'+arg,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.data)))
