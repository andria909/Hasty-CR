"""Freeze the three priorities: no extra lock, central Cannon, own tower damage."""
import argparse
import ast
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
from scripts.kaggle.prepare_motionfix_pilot import prepare as clone,sha

VARIANT='central_tower_damage_v1'
WEIGHTS=(('control',0.),('small',.05),('medium',.15),('large',.4))
PROFILE_KEYS=('action_cooldown_ms','cannon_center_only','tower_damage_all')


def validate_protocol(p):
    if p.get('study_variant')!=VARIANT or p['pilot_seeds']!=[302,303]or len(p['cases'])!=4:
        raise ValueError('Wrong central tower study')
    common=None
    for row,(name,weight)in zip(p['cases'],WEIGHTS):
        if (row['id'],row['tower_support_weight'])!=(name,weight):raise ValueError('Wrong reward grid')
        if any(row[k]!=v for k,v in dict(action_cooldown_ms=0,cannon_center_only=1,tower_damage_all=1,
            spell_waste_weight=.1,missed_fireball_penalty=0.,action_delay_ms=1350,validation_action_delay_ms=1350,
            warmstart_updates=0,supervised_replay_weight=0.).items()):raise ValueError('Priority contract differs')
        fixed={k:v for k,v in row.items()if k not in ('id','tower_support_weight')}
        if common is not None and common!=fixed:raise ValueError('Confounded reward comparison')
        common=fixed


def prepare(source,output):
    root=Path(output);clone(source,root,'hastycr-central-tower-1028')
    prep=json.loads((root/'preparation.json').read_text());c=prep['config'];c['owner']='vertel0'
    archive=root/'dataset/extensions.tar.gz'
    with tarfile.open(archive)as s:payload={m.name:s.extractfile(m).read()for m in s.getmembers()if m.isfile()}
    base=json.loads((root/'dataset/manifest.json').read_text())
    for name in ('sim/cannon_placement.py','sim/continuous_push.py','sim/train_attack_ppo.py','sim/tower_support.py','sim/continuous_tower_support.py'):
        payload[name]=Path(name).read_bytes();c['extensions'][name]=sha(name)
        if name in base['files']:c['replaced'][name]=base['files'][name]
    with tarfile.open(archive,'w:gz')as s:
        for name,value in sorted(payload.items()):
            m=tarfile.TarInfo(name);m.size=len(value);s.addfile(m,io.BytesIO(value))
    c['pins']['extensions.tar.gz']=sha(archive)
    path=root/'dataset/protocol.json';p=json.loads(path.read_text());base=p['cases'][0];cases=[]
    for name,weight in WEIGHTS:
        row=deepcopy(base);row.update(id=name,tower_support_weight=weight,spell_waste_weight=.1,missed_fireball_penalty=0.,
            action_cooldown_ms=0,cannon_center_only=1,tower_damage_all=1)
        cases.append(row)
    p.update(study_variant=VARIANT,pilot_seeds=[302,303],cases=cases,
        rationale_source='reports/central_tower_design_20261028.md',
        paired_comparison='All cases: central Cannon mask and no extra slot lock, fixed empty-spell penalty. Only own-tower damage weight varies.',
        limitations='Central rectangle is a user-requested restriction, not proof of professional optimal placements. Remaining simulation differences unvalidated.')
    validate_protocol(p);path.write_text(json.dumps(p,indent=2)+'\n');c['pins']['protocol.json']=sha(path)
    path=root/'job/train.py';tree=ast.parse(path.read_text())
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    run=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    index=next(i for i,n in enumerate(run.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='args'for t in n.targets))
    run.body[index:index]=ast.parse("coefficients=json.loads(profile.read_text())\ncoefficients.update({k:trial[k] for k in ('action_cooldown_ms','cannon_center_only','tower_damage_all')})\nprofile.write_text(json.dumps(coefficients))").body
    tree.body[-1].value.args=[ast.parse(repr(c),mode='eval').body];ast.fix_missing_locations(tree);path.write_text(ast.unparse(tree)+'\n')
    kernel='vertel0/hastycr-central-tower-1028'
    for name,identity in [('dataset/dataset-metadata.json','vertel0/'+c['slug']),('job/kernel-metadata.json',kernel)]:
        path=root/name;r=json.loads(path.read_text());r.update(id=identity)
        if name.startswith('job/'):r.update(dataset_sources=['vertel0/'+c['slug']])
        path.write_text(json.dumps(r,indent=2)+'\n')
    prep.update(study_variant=VARIANT,kernel=kernel,bootstrap_sha256=sha(root/'job/train.py'))
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return dict(kernel=kernel,submitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(prepare(a.source,a.output)))
