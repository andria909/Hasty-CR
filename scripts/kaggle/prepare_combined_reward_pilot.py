"""Freeze tower support x empty-spell shaping on identical full-game PPO tasks."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
from scripts.kaggle.prepare_motionfix_pilot import prepare as clone,sha

VARIANT='combined_tower_spell_reward_v1'
CASES=(('control',0.,None),('tower',.4,None),('spell',0.,.1),('combined',.4,.1))


def validate_protocol(protocol):
    if protocol.get('study_variant')!=VARIANT or protocol.get('pilot_seeds')!=[282,283]:
        raise ValueError('Wrong combined reward protocol')
    if len(protocol['cases'])!=4:raise ValueError('Four factorial cases required')
    common=None
    for row,(name,tower,spell)in zip(protocol['cases'],CASES):
        if (row['id'],row['tower_support_weight'],row['spell_waste_weight'])!=(name,tower,spell):
            raise ValueError('Combined reward factorial differs')
        if (row['missed_fireball_penalty']!=(.1 if spell is None else 0.) or
            row['warmstart_updates']!=0 or row['supervised_replay_weight']!=0. or
            row['action_delay_ms']!=1350 or row['validation_action_delay_ms']!=1350):
            raise ValueError('Unexpected pretraining/timing or duplicate spell penalty')
        fixed={k:v for k,v in row.items()if k not in ('id','tower_support_weight','spell_waste_weight','missed_fireball_penalty')}
        if common is not None and fixed!=common:raise ValueError('Confounding coefficient changed')
        common=fixed


def prepare(source,output):
    source,root=Path(source),Path(output)
    original=json.loads((source/'preparation.json').read_text())
    if original.get('study_variant')!='spell_waste_reward_v1':raise ValueError('Expected empty-spell source')
    clone(source,root,'hastycr-tower-spell-combined-1028')
    prep=json.loads((root/'preparation.json').read_text());config=prep['config']
    p=root/'dataset/protocol.json';protocol=json.loads(p.read_text());base=protocol['cases'][0]
    cases=[]
    for name,tower,spell in CASES:
        row=deepcopy(base);row.update(id=name,tower_support_weight=tower,spell_waste_weight=spell,
            missed_fireball_penalty=.1 if spell is None else 0.,warmstart_updates=0,supervised_replay_weight=0.)
        cases.append(row)
    protocol.update(study_variant=VARIANT,cases=cases,pilot_seeds=[282,283],
        paired_comparison='2x2 tower-support0/.4 and old Fireball miss versus unified Fireball/Zap waste. Same parent, corrected physics,1350ms and all other rewards. No BC/replay.',
        rationale_source='reports/combined_reward_pilot_design_20261028.md',
        limitations='No verified Cannon placement catalogue or live action mask. No claim to include all requested improvements. Validation gate required before full training.')
    validate_protocol(protocol);p.write_text(json.dumps(protocol,indent=2)+'\n');config['pins']['protocol.json']=sha(p)
    p=root/'job/train.py';tree=ast.parse(p.read_text())
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    run=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    index=next(i for i,n in enumerate(run.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='process'for t in n.targets))
    run.body[index:index]=ast.parse("args+=['--tower-support-weight',str(trial['tower_support_weight'])]").body
    tree.body[-1].value.args=[ast.parse(repr(config),mode='eval').body];ast.fix_missing_locations(tree)
    p.write_text(ast.unparse(tree)+'\n')
    prep.update(study_variant=VARIANT,bootstrap_sha256=sha(p))
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return dict(kernel=prep['kernel'],submitted=False,pilot_target_steps=819200)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output)))
