"""Freeze choice-aware PPO as a separate study, preserving submitted jobs."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

from scripts.kaggle.prepare_air_exposure_pilot import validate_shared_trial,STEPS
from scripts.kaggle.prepare_motionfix_pilot import sha
from scripts.kaggle.prepare_spell_interaction_pilot import prepare as prepare_factorial

VARIANT='policy_choice_v1'
SEEDS=[482,483]
CASES=(('control','all','all'),('choice_loss','choice','all'),
       ('choice_advantages','all','choice'),('choice_both','choice','choice'))
CONTROLS={name:'control'for name,_,_ in CASES[1:]}
REQUIRED_TRAINING_SOURCES={'sim/train_attack_ppo.py','sim/policy_choice.py',
                           'sim/train_defense_ppo.py','sim/group_entropy.py'}


def validate_protocol(p):
    if not REQUIRED_TRAINING_SOURCES.issubset(p.get('training_sources',{})):
        raise ValueError('Complete policy-choice trainer dependency pins required')
    if (p.get('study_variant')!=VARIANT or p.get('pilot_seeds')!=SEEDS or
            p.get('pilot_steps_each')!=STEPS or p.get('total_pilot_steps')!=8*STEPS or
            p.get('paired_controls')!=CONTROLS or len(p['cases'])!=4):
        raise ValueError('Fixed policy-choice factorial required')
    common=None
    for row,(name,reduction,normalization)in zip(p['cases'],CASES):
        if (row['id'],row['policy_reduction'],row['advantage_normalization'],row['initial_id'],
                row['focus'],row['spell_waste_weight'],row['gae_lambda'])!=(name,reduction,normalization,'parent','uniform',.3,.95):
            raise ValueError('Choice grid or common initializer differs')
        fixed={k:v for k,v in row.items()if k not in ('id','policy_reduction','advantage_normalization')}
        if common is not None and fixed!=common:raise ValueError('Only actor reduction and advantage normalization may vary')
        common=fixed;validate_shared_trial(row)
        if any(row[k]!=v for k,v in dict(own_hp_penalty=6.,reserve_potential_weight=.1,
                                       fireball_value_reward=.25,card_balance_target='units').items()):
            raise ValueError('Common reward/auxiliary objective differs')


def prepare(source,output,design):
    planned=json.loads(Path(design).read_text())
    for name,digest in planned['training_sources'].items():
        if sha(name)!=digest:raise ValueError('Designed trainer source changed: '+name)
    prepare_factorial(source,output,design,variant=VARIANT,validator=validate_protocol,
                      experiment='policy_choice',slug='hastycr-choice1104')
    root=Path(output);prep=json.loads((root/'preparation.json').read_text());c=prep['config']
    c['owner']='andriashadrin'
    archive=root/'dataset/extensions.tar.gz';base=json.loads((root/'dataset/manifest.json').read_text())
    with tarfile.open(archive)as stream:
        payload={m.name:stream.extractfile(m).read()for m in stream.getmembers()if m.isfile()}
    for name in (*planned['training_sources'],'scripts/attack/policy_choice_accounting.py','scripts/attack/boundary_state.py'):
        payload[name]=Path(name).read_bytes();c['extensions'][name]=hashlib.sha256(payload[name]).hexdigest()
        if name in base['files']:c['replaced'][name]=base['files'][name]
    with tarfile.open(archive,'w:gz')as stream:
        for name,content in sorted(payload.items()):
            info=tarfile.TarInfo(name);info.size=len(content);stream.addfile(info,io.BytesIO(content))
    c['pins']['extensions.tar.gz']=sha(archive)
    tree=ast.parse((root/'job/train.py').read_text())
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    run=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    assignments=[n for n in run.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='args'for t in n.targets)]
    if len(assignments)!=1:raise ValueError('Expected one trainer command')
    patch=ast.parse("args += ['--policy-reduction', trial['policy_reduction'], '--advantage-normalization', trial['advantage_normalization']]").body[0]
    run.body.insert(run.body.index(assignments[0])+1,patch)
    tree.body[-1].value.args=[ast.parse(repr(c),mode='eval').body];ast.fix_missing_locations(tree)
    (root/'job/train.py').write_text(ast.unparse(tree)+'\n')
    kernel=c['owner']+'/hastycr-choice1104'
    for name,identity in [('dataset/dataset-metadata.json',c['owner']+'/'+c['slug']),('job/kernel-metadata.json',kernel)]:
        path=root/name;meta=json.loads(path.read_text());meta.update(id=identity,title=identity.split('/')[-1])
        if name.startswith('job/'):meta['dataset_sources']=[c['owner']+'/'+c['slug']]
        path.write_text(json.dumps(meta,indent=2)+'\n')
    prep.update(kernel=kernel,bootstrap_sha256=sha(root/'job/train.py'),trainer_flags=['policy_reduction','advantage_normalization'])
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return dict(kernel=kernel,submitted=False,pilot_steps=8*STEPS)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','output','design'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.design)))
