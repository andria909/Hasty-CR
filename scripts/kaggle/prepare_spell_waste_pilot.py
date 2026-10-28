"""Freeze four empty-spell reward coefficients; no submission or auto full run."""
import argparse
import ast
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
from scripts.kaggle.prepare_native_latency_pilot import prepare as prepare_latency
from scripts.kaggle.local_artifact_transfer import file_sha as sha


def prepare(output,owner='vertel0'):
    root=Path(output);prepare_latency(root,owner)
    prep=json.loads((root/'preparation.json').read_text());config=prep['config'];dataset=root/'dataset';job=root/'job'
    archive=dataset/'extensions.tar.gz'
    with tarfile.open(archive)as source:
        payload={m.name:source.extractfile(m).read()for m in source.getmembers()if m.isfile()}
    for name in ('sim/tower_support.py','sim/continuous_tower_support.py','sim/train_attack_ppo.py','sim/spell_waste.py','sim/continuous_spell_waste.py'):
        payload[name]=Path(name).read_bytes();config['extensions'][name]=sha(Path(name))
    with tarfile.open(archive,'w:gz')as target:
        for name in sorted(payload):
            info=tarfile.TarInfo(name);info.size=len(payload[name]);target.addfile(info,io.BytesIO(payload[name]))
    config['pins']['extensions.tar.gz']=sha(archive)
    protocol=json.loads((dataset/'protocol.json').read_text());base=next(t for t in protocol['cases']if t['id']=='delay1350')
    cases=[]
    for name,weight in [('control',None),('small',.1),('medium',.5),('large',1.)]:
        trial=deepcopy(base);trial.update(id=name,spell_waste_weight=weight,
            action_delay_ms=1350,validation_action_delay_ms=1350,warmstart_updates=0,supervised_replay_weight=0.)
        cases.append(trial)
    protocol.update(study_variant='spell_waste_reward_v1',cases=cases,pilot_seeds=[242,243],
        paired_comparison='Same balanced initializer/stable physics/1350ms timing; no BC/replay, old Fireball-only control versus unified empty-spell cost coefficients; no tower-support bonus in this isolated study.',
        rationale_source='reports/spell_waste_training_design_20261028.md',
        limitations='Contact conservatively exempts a spell from empty-cast penalty; this does not prove good exchange value. Evaluate win rate and tower HP too.')
    (dataset/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');config['pins']['protocol.json']=sha(dataset/'protocol.json')
    slug='hastycr-spell-waste-inputs-20261028';kernel=owner+'/hastycr-spell-waste-pilot-20261028'
    config.update(slug=slug,source_slug=slug)
    tree=ast.parse((job/'train.py').read_text());execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    run=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    index=next(i for i,n in enumerate(run.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='process'for t in n.targets))
    run.body[index:index]=ast.parse("if trial['spell_waste_weight'] is not None:\n    args+=['--spell-waste-weight',str(trial['spell_waste_weight'])]").body
    tree.body[-1].value.args=[ast.parse(repr(config),mode='eval').body]
    ast.fix_missing_locations(tree);(job/'train.py').write_text(ast.unparse(tree)+'\n')
    for path,identity in [(dataset/'dataset-metadata.json',owner+'/'+slug),(job/'kernel-metadata.json',kernel)]:
        row=json.loads(path.read_text());row.update(id=identity,title=identity.split('/')[-1])
        if path.parent==job:row.update(dataset_sources=[owner+'/'+slug],enable_gpu=False)
        path.write_text(json.dumps(row,indent=2)+'\n')
    prep.update(kernel=kernel,config=config,bootstrap_sha256=sha(job/'train.py'),submitted=False,
        study_variant=protocol['study_variant'],requires_exact_preflight=True)
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n');return prep


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=prepare(a.output)
    print(json.dumps(dict(kernel=r['kernel'],submitted=False,pilot_steps=819200)))
