"""Freeze a longer factorial pilot without changing previous submitted bundles."""
import argparse
import ast
import io
import json
from pathlib import Path
import tarfile

from scripts.kaggle.prepare_motionfix_pilot import prepare as clone,sha

VARIANT='air_exposure_v1'
STEPS=409600
SEEDS=[442,443]
CASES=(('uniform01','uniform',.1),('uniform03','uniform',.3),
       ('focus01','air_golem',.1),('focus03','air_golem',.3))


def validate_protocol(p):
    if (p.get('study_variant')!=VARIANT or p.get('pilot_seeds')!=SEEDS or
            p.get('pilot_steps_each')!=STEPS or p.get('total_pilot_steps')!=8*STEPS or len(p['cases'])!=4):
        raise ValueError('Fixed longer factorial pilot required')
    if p.get('paired_controls')!={'uniform03':'uniform01','focus01':'uniform01','focus03':'uniform03'}:
        raise ValueError('Prespecified paired controls differ')
    common=None
    for row,(name,focus,cost)in zip(p['cases'],CASES):
        if (row['id'],row['focus'],row['spell_waste_weight'],row['initial_id'])!=(name,focus,cost,'parent'):
            raise ValueError('Factorial grid or initializer changed')
        fixed={k:v for k,v in row.items()if k not in ('id','focus','spell_waste_weight')}
        if common is not None and fixed!=common:raise ValueError('Only focus and empty cost may vary')
        common=fixed
        required=dict(overflow_penalty=.3,tower_support_weight=.15,crown_only_weight=0.,dormant_king_weight=0.,
            action_delay_ms=1350,validation_action_delay_ms=1350,action_cooldown_ms=0,cannon_center_only=1,
            tower_damage_all=1,source_radius_reach=1,hog_sight_clipping=1,target_lock_in_range=0,
            tactical_start_fraction=0.,warmstart_updates=0,supervised_replay_weight=0.,card_balance_weight=.03,
            history_frames=1,entropy_coef=.01,gamma=.999,missed_fireball_penalty=0.)
        if any(row.get(k)!=v for k,v in required.items()):raise ValueError('Frozen execution/reward contract differs')


def prepare(source,output,design):
    root=Path(output);design=Path(design);planned=json.loads(design.read_text())
    if sha(Path(source)/'dataset/protocol.json')!=planned['source_protocol_sha256']:
        raise ValueError('Designed source differs')
    clone(source,root,'hastycr-air-exposure1102')
    prep=json.loads((root/'preparation.json').read_text());c=prep['config']
    if c['owner']!='andriashadrin'or len(c['models'])!=1 or c['models'][0]['sha256']!=planned['initial_sha256']:
        raise ValueError('Fixed owner and BC800 initializer required')
    c['allow_full_training']=False
    p=json.loads((root/'dataset/protocol.json').read_text())
    p.update(planned);p.update(schema='continuous-air-exposure-pilot-v1',study_variant=VARIANT,
        experiment='air_exposure',status='prepared',design_sha256=sha(design))
    validate_protocol(p)
    (root/'dataset/protocol.json').write_text(json.dumps(p,indent=2)+'\n');c['pins']['protocol.json']=sha(root/'dataset/protocol.json')
    selector='scripts/attack/select_air_exposure.py'
    archive=root/'dataset/extensions.tar.gz'
    with tarfile.open(archive)as stream:payload={m.name:stream.extractfile(m).read()for m in stream.getmembers()if m.isfile()}
    payload[selector]=Path(selector).read_bytes();c['extensions'][selector]=sha(selector)
    with tarfile.open(archive,'w:gz')as stream:
        for name,content in sorted(payload.items()):
            info=tarfile.TarInfo(name);info.size=len(content);stream.addfile(info,io.BytesIO(content))
    c['pins']['extensions.tar.gz']=sha(archive)
    tree=ast.parse((root/'job/train.py').read_text());execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    run=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    replacements=0
    for n in run.body:
        if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='steps'for t in n.targets):
            if not isinstance(n.value,ast.IfExp)or n.value.orelse.value!=102400:raise ValueError('Unexpected source budget')
            n.value.orelse=ast.Constant(STEPS);replacements+=1
    for n in ast.walk(execute):
        if isinstance(n,ast.ImportFrom)and n.module=='scripts.attack.select_continuous_focus':
            n.names=[a for a in n.names if a.name!='select_profiles']
            execute.body.insert(execute.body.index(n)+1,ast.ImportFrom(module='scripts.attack.select_air_exposure',names=[ast.alias(name='select_profiles')],level=0))
            replacements+=1
        if isinstance(n,ast.keyword)and n.arg=='pilot_steps':
            if not isinstance(n.value,ast.Constant)or n.value.value!=819200:raise ValueError('Unexpected cohort budget')
            n.value=ast.Constant(8*STEPS);replacements+=1
    if replacements!=3:raise ValueError('Budget/selector patch coverage differs')
    tree.body[-1].value.args=[ast.parse(repr(c),mode='eval').body];ast.fix_missing_locations(tree)
    (root/'job/train.py').write_text(ast.unparse(tree)+'\n')
    prep.update(study_variant=VARIANT,experiment='air_exposure',pilot_steps=8*STEPS,
        bootstrap_sha256=sha(root/'job/train.py'),design_sha256=sha(design),automatic_full_training=False)
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return dict(kernel=prep['kernel'],submitted=False,pilot_steps=8*STEPS)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source','output','design'):p.add_argument('--'+k,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.design)))
