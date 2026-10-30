"""Paired crown-spell reward study; no invalid bank-teacher demonstrations."""
import argparse
import ast
import json
from pathlib import Path
from scripts.kaggle.prepare_mana_campaign import prepare as clone
from scripts.kaggle.prepare_motionfix_pilot import sha

VARIANT='crown_economy_v1'


def validate_protocol(p):
    if (p.get('study_variant')!=VARIANT or p.get('pilot_seeds')!=[382,383] or
            p.get('total_pilot_steps')!=819200):
        raise ValueError('Unexpected crown economy study')
    cases=p['cases']
    pairs=[(0.,0.),(0.,.1),(0.,.4),(.1,.4)]
    if [r['id']for r in cases]!=['control','low','medium','high']:
        raise ValueError('Four ordered profiles required')
    constants={k:v for k,v in cases[0].items()if k not in ('id','crown_only_weight','dormant_king_weight')}
    for row,pair in zip(cases,pairs):
        if ((row['crown_only_weight'],row['dormant_king_weight'])!=pair or
                {k:v for k,v in row.items()if k not in ('id','crown_only_weight','dormant_king_weight')}!=constants):
            raise ValueError('Confounded reward study')
        if (row['warmstart_updates'] or row['supervised_replay_weight'] or
                row['action_delay_ms']!=1350 or row['validation_action_delay_ms']!=1350 or
                row['cannon_center_only']!=1 or row['tower_damage_all']!=1 or
                row['source_radius_reach']!=1 or row['hog_sight_clipping']!=1 or
                row['tactical_start_fraction']!=0 or row['card_balance_weight']!=.03):
            raise ValueError('Crown study physics/teacher contract differs')


def prepare(source,output):
    root=Path(output)
    clone(source,root,'balance')
    prep=json.loads((root/'preparation.json').read_text());config=prep['config']
    protocol=json.loads((root/'dataset/protocol.json').read_text())
    base=dict(protocol['cases'][0],card_balance_weight=.03)
    protocol.update(schema='continuous-crown-economy-pilot-v1',study_variant=VARIANT,
        experiment='nonlethal_crown_only_spell_cost',pilot_seeds=[382,383],
        cases=[dict(base,id=name,crown_only_weight=crown,dormant_king_weight=king)
               for name,(crown,king)in zip(('control','low','medium','high'),[(0.,0.),(0.,.1),(0.,.4),(.1,.4)])],
        paired_comparison='Shared parent/physics/economy; only disjoint crown-only spell costs vary.',
        rationale_source='reports/defense_plan/crown_only_spell_reward_hole_20261030.json',
        teacher_data_used=False,
        limitations=['Reward-only study; safe-bank teacher is not yet ready',
                    'Crown chip can be tactically useful; test finishing and mixed targets separately',
                    'Dormancy is measured at request, not a causal impact-time activation label',
                    'No live games; synthetic validation is not a live win rate'])
    validate_protocol(protocol)
    slug='hastycr-crown-econ1030';kernel=config['owner']+'/'+slug
    config.update(slug=slug+'-inputs',source_slug=slug+'-inputs',allow_full_training=False)
    (root/'dataset/protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    config['pins']['protocol.json']=sha(root/'dataset/protocol.json')
    tree=ast.parse((root/'job/train.py').read_text())
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    run=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    i=next(i for i,n in enumerate(run.body)if isinstance(n,ast.Assign)and
           any(isinstance(t,ast.Name)and t.id=='args'for t in n.targets))
    run.body[i+1:i+1]=ast.parse("args += ['--crown-only-weight',str(trial['crown_only_weight']),'--dormant-king-weight',str(trial['dormant_king_weight'])]").body
    i=next(i for i,n in enumerate(execute.body)if isinstance(n,ast.Assign)and
           any(isinstance(t,ast.Name)and t.id=='options'for t in n.targets))
    execute.body[i+1:i+1]=ast.parse('options.update(crown_only_weight=0.,dormant_king_weight=0.)').body
    tree.body[-1].value.args=[ast.parse(repr(config),mode='eval').body]
    ast.fix_missing_locations(tree);(root/'job/train.py').write_text(ast.unparse(tree)+'\n')
    for name,identity in [('dataset/dataset-metadata.json',config['owner']+'/'+config['slug']),('job/kernel-metadata.json',kernel)]:
        path=root/name;metadata=json.loads(path.read_text());metadata.update(id=identity,title=identity.split('/')[-1])
        if name.startswith('job/'):metadata['dataset_sources']=[config['owner']+'/'+config['slug']]
        path.write_text(json.dumps(metadata,indent=2)+'\n')
    prep.update(study_variant=VARIANT,kernel=kernel,config=config,bootstrap_sha256=sha(root/'job/train.py'))
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return dict(kernel=kernel,steps=819200,submitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output)))
