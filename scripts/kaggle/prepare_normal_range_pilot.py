"""Freeze the same paid-start grid with corrected ordinary attack geometry.

The initializer/rewards come from the verified central study, not unfinished
tactical weights. Previously submitted packages are never modified.
"""
import argparse
import ast
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile

from scripts.kaggle.prepare_tactical_starts_pilot import (
    prepare as prepare_base, protocol_from as base_protocol, validate_protocol as validate_base,
    VARIANT as BASE_VARIANT)
from scripts.kaggle.prepare_motionfix_pilot import sha

VARIANT='edge_to_edge_paid_tactical_v1'
FLAGS=dict(source_radius_reach=1,target_lock_in_range=0)


def protocol_from(previous,proof):
    result=base_protocol(previous,proof)
    result.update(study_variant=VARIANT,
        paired_comparison='Same reward coefficients, initializer, start fractions and seeds as the prior tactical grid; corrected ordinary range in every train/validation game.',
        rationale_source='reports/normal_range_correction_20261028.md',
        comparison_scope='Re-evaluate baselines on corrected physics; old-physics wins are not interchangeable.',
        after='Verify corrected geometry, all budgets and family gate before selecting at least two full-training profiles.')
    for trial in result['cases']:trial.update(FLAGS)
    result['constants'].update(FLAGS)
    validate_protocol(result)
    return result


def validate_protocol(protocol):
    if protocol.get('study_variant')!=VARIANT:raise ValueError('Wrong corrected-range study')
    original=deepcopy(protocol);original['study_variant']=BASE_VARIANT
    for row in [*original['cases'],original['constants']]:
        for key,value in FLAGS.items():
            if type(row.get(key)) is not int or row.pop(key)!=value:
                raise ValueError('Corrected combat geometry differs')
    validate_base(original)


def prepare(source,output):
    root=Path(output)
    prepare_base(source,root)
    previous=json.loads((root/'dataset/protocol.json').read_text())
    proof=json.loads((root/'dataset/predecessor_verified.json').read_text())
    protocol=protocol_from(previous,proof)
    prep=json.loads((root/'preparation.json').read_text());config=prep['config']
    slug='hastycr-edge-tactical-1028';owner='vertel0';kernel=owner+'/'+slug
    config.update(owner=owner,slug=slug+'-inputs',source_slug=slug+'-inputs',allow_full_training=False)
    archive=root/'dataset/extensions.tar.gz'
    with tarfile.open(archive)as stream:
        payload={m.name:stream.extractfile(m).read()for m in stream.getmembers()if m.isfile()}
    base=json.loads((root/'dataset/manifest.json').read_text())
    for name in ('sim/gamedata.py','sim/public_gamedata.py'):
        payload[name]=Path(name).read_bytes();config['extensions'][name]=sha(name)
        if name in base['files']:config['replaced'][name]=base['files'][name]
    with tarfile.open(archive,'w:gz')as stream:
        for name,value in sorted(payload.items()):
            info=tarfile.TarInfo(name);info.size=len(value);stream.addfile(info,io.BytesIO(value))
    config['pins']['extensions.tar.gz']=sha(archive)
    path=root/'dataset/protocol.json';path.write_text(json.dumps(protocol,indent=2)+'\n')
    config['pins']['protocol.json']=sha(path)
    path=root/'job/train.py';tree=ast.parse(path.read_text())
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    run=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    index=next(i for i,n in enumerate(run.body)if isinstance(n,ast.Assign)and
        any(isinstance(t,ast.Name)and t.id=='args'for t in n.targets))
    run.body[index:index]=ast.parse("coefficients = json.loads(profile.read_text())\ncoefficients.update({key: trial[key] for key in ('source_radius_reach', 'target_lock_in_range')})\nprofile.write_text(json.dumps(coefficients))").body
    tree.body[-1].value.args=[ast.parse(repr(config),mode='eval').body]
    ast.fix_missing_locations(tree);path.write_text(ast.unparse(tree)+'\n')
    for name,identity in [('dataset/dataset-metadata.json',owner+'/'+config['slug']),('job/kernel-metadata.json',kernel)]:
        path=root/name;record=json.loads(path.read_text());record.update(id=identity,title=identity.split('/')[-1])
        if name.startswith('job/'):record['dataset_sources']=[owner+'/'+config['slug']]
        path.write_text(json.dumps(record,indent=2)+'\n')
    prep.update(study_variant=VARIANT,kernel=kernel,bootstrap_sha256=sha(root/'job/train.py'))
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return dict(kernel=kernel,submitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.source,args.output)))
