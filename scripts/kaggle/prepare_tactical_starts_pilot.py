"""Freeze a paired start-distribution study only after central pilot verification."""
import argparse
import ast
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile

from scripts.kaggle.prepare_motionfix_pilot import prepare as clone, sha

VARIANT='paid_tactical_fullgame_starts_v1'
FRACTIONS=(('control',0.),('low',.1),('medium',.25),('high',.5))


def protocol_from(previous, verified):
    if (not verified.get('central_tower_audit',{}).get('verified') or
            verified.get('pilot_steps_verified')!=819200):
        raise ValueError('Complete verified central pilot required')
    if verified.get('full_gate_met'):
        raise ValueError('Central candidates passed; promote them before another pilot')
    ranking=verified['pilot_selection']['ranking']
    if len(ranking)!=4:raise ValueError('Four central profiles required')
    fixed=deepcopy(ranking[0]['trial'])
    cases=[]
    for name,fraction in FRACTIONS:
        trial=deepcopy(fixed);trial.update(id=name,tactical_start_fraction=fraction)
        cases.append(trial)
    p=deepcopy(previous)
    p.update(study_variant=VARIANT,pilot_seeds=[322,323],cases=cases,
        predecessor_trial=fixed,predecessor_selection_score=ranking[0]['score'],
        rationale_source='reports/tactical_full_games_20261028.md',
        paired_comparison='Only the TRAIN tactical-start fraction changes. All rewards and the full validation are fixed.',
        limitations='Synthetic midgame states, not reconstructed real histories. No guarantee of improved live play.',
        constants={k:v for k,v in fixed.items() if k!='id'},
        penalty='Paid empty-spell cost after resolution; no per-card deployment bonus.',
        after='Verify all pilots, then promote only if at least two profiles pass the existing family gate.')
    validate_protocol(p)
    return p


def validate_protocol(p):
    if p.get('study_variant')!=VARIANT or p['pilot_seeds']!=[322,323] or len(p['cases'])!=4:
        raise ValueError('Wrong tactical-start study')
    common=None
    for row,(name,fraction) in zip(p['cases'],FRACTIONS):
        if (row['id'],row['tactical_start_fraction'])!=(name,fraction):raise ValueError('Wrong start-fraction grid')
        required=dict(cannon_center_only=1,action_cooldown_ms=0,tower_damage_all=1,
            action_delay_ms=1350,validation_action_delay_ms=1350,history_frames=1,
            warmstart_updates=0,supervised_replay_weight=0.,spell_waste_weight=.1,missed_fireball_penalty=0.)
        if any(row.get(k)!=v for k,v in required.items()):raise ValueError('Priority or training contract differs')
        fixed={k:v for k,v in row.items() if k not in ('id','tactical_start_fraction')}
        if common is not None and fixed!=common:raise ValueError('Confounded start-distribution comparison')
        common=fixed
    predecessor={k:v for k,v in p['predecessor_trial'].items() if k!='id'}
    if common!=predecessor or p['constants']!=predecessor:
        raise ValueError('Fixed coefficients differ from the verified predecessor')
    if p['pilot_steps_each']!=102400 or p['total_pilot_steps']!=819200:raise ValueError('Pilot budget differs')


def prepare(source, output):
    from scripts.attack.verify_central_tower_pilot import verify_central
    source,root=Path(source),Path(output)
    verified=verify_central(source/'collected',source)
    p=protocol_from(json.loads((source/'dataset/protocol.json').read_text()),verified)
    slug='hastycr-tactical-starts-1028'
    clone(source,root,slug)
    prep=json.loads((root/'preparation.json').read_text());c=prep['config']
    c.update(owner='andreiperelaiko',allow_full_training=False)
    archive=root/'dataset/extensions.tar.gz'
    with tarfile.open(archive) as stream:
        payload={m.name:stream.extractfile(m).read() for m in stream.getmembers() if m.isfile()}
    base=json.loads((root/'dataset/manifest.json').read_text())
    for name in ('sim/train_attack_ppo.py','sim/continuous_tactical_starts.py'):
        payload[name]=Path(name).read_bytes();c['extensions'][name]=sha(name)
        if name in base['files']:c['replaced'][name]=base['files'][name]
    with tarfile.open(archive,'w:gz') as stream:
        for name,value in sorted(payload.items()):
            info=tarfile.TarInfo(name);info.size=len(value);stream.addfile(info,io.BytesIO(value))
    c['pins']['extensions.tar.gz']=sha(archive)
    refreshed=json.loads((root/'dataset/protocol.json').read_text())
    p['physics_pins']=refreshed['physics_pins']
    for name,record in [('protocol.json',p),('predecessor_verified.json',verified)]:
        path=root/'dataset'/name;path.write_text(json.dumps(record,indent=2)+'\n');c['pins'][name]=sha(path)
    path=root/'job/train.py';tree=ast.parse(path.read_text())
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    run=next(n for n in execute.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    index=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and
               any(isinstance(t,ast.Name) and t.id=='args' for t in n.targets))
    run.body[index+1:index+1]=ast.parse("args += ['--tactical-start-fraction', str(trial['tactical_start_fraction'])]").body
    tree.body[-1].value.args=[ast.parse(repr(c),mode='eval').body]
    ast.fix_missing_locations(tree);path.write_text(ast.unparse(tree)+'\n')
    kernel=c['owner']+'/'+slug
    for name,identity in [('dataset/dataset-metadata.json',c['owner']+'/'+c['slug']),('job/kernel-metadata.json',kernel)]:
        path=root/name;record=json.loads(path.read_text());record.update(id=identity)
        if name.startswith('job/'):record.update(dataset_sources=[c['owner']+'/'+c['slug']])
        path.write_text(json.dumps(record,indent=2)+'\n')
    prep.update(study_variant=VARIANT,kernel=kernel,bootstrap_sha256=sha(root/'job/train.py'))
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return dict(kernel=kernel,submitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.source,args.output)))
