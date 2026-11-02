"""Paired empty-spell/dormant-king costs on a separate authorized account."""
import argparse
import ast
import json
from pathlib import Path

from scripts.kaggle.prepare_motionfix_pilot import prepare as clone,sha
from scripts.kaggle.prepare_air_exposure_pilot import validate_shared_trial,STEPS

VARIANT='spell_interaction_v1'
SEEDS=[452,453]
CASES=(('control01',.1,0.),('control03',.3,0.),('king01',.1,.4),('king03',.3,.4))
CONTROLS={'control03':'control01','king01':'control01','king03':'control03'}


def validate_protocol(p):
    if (p.get('study_variant')!=VARIANT or p.get('pilot_seeds')!=SEEDS or
            p.get('pilot_steps_each')!=STEPS or p.get('total_pilot_steps')!=8*STEPS or
            p.get('paired_controls')!=CONTROLS or len(p['cases'])!=4):
        raise ValueError('Fixed spell-interaction experiment required')
    common=None
    for row,(name,cost,king)in zip(p['cases'],CASES):
        if (row['id'],row['spell_waste_weight'],row['dormant_king_weight'],row['initial_id'],row['focus'])!=(name,cost,king,'parent','uniform'):
            raise ValueError('Spell interaction grid or common initializer differs')
        fixed={k:v for k,v in row.items()if k not in ('id','spell_waste_weight','dormant_king_weight')}
        if common is not None and fixed!=common:raise ValueError('Only two spell costs may vary')
        common=fixed;validate_shared_trial(row,dormant_king_weight=king)


def prepare(source,output,design,*,variant=VARIANT,validator=validate_protocol,experiment='spell_interaction',slug='hastycr-spell-pair1102'):
    source,root,design=Path(source),Path(output),Path(design);planned=json.loads(design.read_text())
    if sha(source/'dataset/protocol.json')!=planned['source_protocol_sha256']:raise ValueError('Designed source changed')
    clone(source,root,slug)
    prep=json.loads((root/'preparation.json').read_text());c=prep['config']
    if prep['study_variant']!='air_exposure_v1'or len(c['models'])!=1 or c['models'][0]['sha256']!=planned['initial_sha256']:
        raise ValueError('Verified exposure controller and BC800 initial required')
    c.update(owner='vertel0',allow_full_training=False)
    p=json.loads((root/'dataset/protocol.json').read_text());p.update(planned)
    p.update(schema='continuous-'+experiment.replace('_','-')+'-pilot-v1',study_variant=variant,
        experiment=experiment,status='prepared',design_sha256=sha(design))
    validator(p)
    (root/'dataset/protocol.json').write_text(json.dumps(p,indent=2)+'\n');c['pins']['protocol.json']=sha(root/'dataset/protocol.json')
    tree=ast.parse((root/'job/train.py').read_text())
    # The400-update controller and imported selector are unchanged; only its hash-pinned inputs differ.
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    run=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    budget=next(n.value for n in run.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='steps'for t in n.targets))
    if not isinstance(budget,ast.IfExp)or budget.orelse.value!=STEPS:raise ValueError('Expected400-update pilot branch')
    tree.body[-1].value.args=[ast.parse(repr(c),mode='eval').body];ast.fix_missing_locations(tree)
    (root/'job/train.py').write_text(ast.unparse(tree)+'\n')
    kernel=c['owner']+'/'+slug
    for file,identity in [('dataset/dataset-metadata.json',c['owner']+'/'+c['slug']),('job/kernel-metadata.json',kernel)]:
        path=root/file;row=json.loads(path.read_text());row.update(id=identity,title=identity.split('/')[-1])
        if file.startswith('job/'):row['dataset_sources']=[c['owner']+'/'+c['slug']]
        path.write_text(json.dumps(row,indent=2)+'\n')
    prep.update(kernel=kernel,study_variant=variant,experiment=experiment,bootstrap_sha256=sha(root/'job/train.py'),design_sha256=sha(design))
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return dict(kernel=kernel,submitted=False,pilot_steps=8*STEPS)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','output','design'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.design)))
