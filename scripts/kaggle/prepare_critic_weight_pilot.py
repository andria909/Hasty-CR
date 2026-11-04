"""Freeze a critic-weight sweep; same policy, rewards, physics and pilot budget."""
import argparse
import ast
import json
from pathlib import Path

from scripts.kaggle.prepare_air_exposure_pilot import validate_shared_trial,STEPS
from scripts.kaggle.prepare_spell_interaction_pilot import prepare as prepare_factorial
from scripts.kaggle.local_artifact_transfer import file_sha

VARIANT='critic_weight_v1'
SEEDS=[492,493]
CASES=(('control',.5),('value025',.25),('value010',.1),('value005',.05))
CONTROLS={name:'control'for name,_ in CASES[1:]}


def validate_protocol(p):
    if (p.get('study_variant')!=VARIANT or p.get('pilot_seeds')!=SEEDS or
            p.get('pilot_steps_each')!=STEPS or p.get('total_pilot_steps')!=8*STEPS or
            p.get('paired_controls')!=CONTROLS or len(p['cases'])!=4):
        raise ValueError('Fixed critic-weight sweep required')
    common=None
    for row,(name,coefficient)in zip(p['cases'],CASES):
        if (row['id'],row['value_coef'],row['initial_id'],row['focus'],row['spell_waste_weight'],row['gae_lambda'])!=(name,coefficient,'parent','uniform',.3,.95):
            raise ValueError('Critic coefficient grid or initializer differs')
        fixed={k:v for k,v in row.items()if k not in('id','value_coef')}
        if common is not None and fixed!=common:raise ValueError('Only critic coefficient may vary')
        common=fixed;validate_shared_trial(row)
        if any(row.get(k)!=v for k,v in dict(own_hp_penalty=6.,reserve_potential_weight=.1,
                fireball_value_reward=.25,card_balance_target='units').items()):
            raise ValueError('Common reward coefficients differ')
        if row.get('policy_reduction','all')!='all'or row.get('advantage_normalization','all')!='all':
            raise ValueError('Legacy actor reduction required')


def prepare(source,output,design):
    prepare_factorial(source,output,design,variant=VARIANT,validator=validate_protocol,
        experiment='critic_weight',slug='hastycr-critic1104')
    root=Path(output);p=root/'preparation.json';prep=json.loads(p.read_text());c=prep['config']
    c['owner']='andreiperelaiko'
    tree=ast.parse((root/'job/train.py').read_text())
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    run=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    assignments=[n for n in run.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='args'for t in n.targets)]
    if len(assignments)!=1:raise ValueError('Expected one frozen trainer command')
    run.body.insert(run.body.index(assignments[0])+1,ast.parse("args += ['--value-coef', str(trial['value_coef'])]").body[0])
    tree.body[-1].value.args=[ast.parse(repr(c),mode='eval').body];ast.fix_missing_locations(tree)
    (root/'job/train.py').write_text(ast.unparse(tree)+'\n')
    kernel=c['owner']+'/hastycr-critic1104'
    for name,identity in [('dataset/dataset-metadata.json',c['owner']+'/'+c['slug']),('job/kernel-metadata.json',kernel)]:
        path=root/name;meta=json.loads(path.read_text());meta.update(id=identity,title=identity.split('/')[-1])
        if name.startswith('job/'):meta['dataset_sources']=[c['owner']+'/'+c['slug']]
        path.write_text(json.dumps(meta,indent=2)+'\n')
    prep.update(kernel=kernel,bootstrap_sha256=file_sha(root/'job/train.py'),trainer_flags=['value_coef'])
    p.write_text(json.dumps(prep,indent=2)+'\n')
    return dict(kernel=kernel,submitted=False,pilot_steps=8*STEPS)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in('source','output','design'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.design)))
