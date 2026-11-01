"""Compare PPO adaptation of fixed initial policies, with identical rewards."""
import argparse
import ast
import io
import json
from pathlib import Path
import shutil
import tarfile

from scripts.kaggle.prepare_motionfix_pilot import prepare as clone,sha

VARIANT='policy_adaptation_v1'
IDS=('balanced','bank','bc800','bc8000')
INITIALS=('parent','bank','bc800','bc8000')


def validate_protocol(p):
    if (p.get('study_variant')!=VARIANT or p.get('pilot_seeds')!=[412,413]
            or p.get('pilot_steps_each')!=102400 or p.get('total_pilot_steps')!=819200
            or [t['id']for t in p['cases']]!=list(IDS)):
        raise ValueError('Fixed four-initializer PPO pilot required')
    common={k:v for k,v in p['cases'][0].items()if k not in ('id','initial_id')}
    for t,initial in zip(p['cases'],INITIALS):
        if t['initial_id']!=initial or {k:v for k,v in t.items()if k not in ('id','initial_id')}!=common:
            raise ValueError('Only initialization may vary')
        fixed=dict(action_delay_ms=1350,validation_action_delay_ms=1350,cannon_center_only=1,
            action_cooldown_ms=0,tower_damage_all=1,source_radius_reach=1,hog_sight_clipping=1,
            target_lock_in_range=0,tower_support_weight=.15,spell_waste_weight=.1,
            crown_only_weight=0.,dormant_king_weight=0.,tactical_start_fraction=0.,
            warmstart_updates=0,supervised_replay_weight=0.,history_frames=1,card_balance_weight=.03)
        if any(t.get(k)!=v for k,v in fixed.items()):raise ValueError('Shared PPO execution/reward contract differs')


def prepare(source,output,balanced,bank,short,long):
    root=Path(output);slug='hastycr-adapt-ppo1101';clone(source,root,slug)
    prep=json.loads((root/'preparation.json').read_text());c=prep['config'];c['allow_full_training']=False
    if c['owner']!='vertel0':raise ValueError('This experiment uses the authorized second account')
    models=[]
    for identity,path in zip(INITIALS,(balanced,bank,short,long)):
        filename='adapt_'+identity+'.pt';shutil.copyfile(path,root/'dataset'/filename)
        digest=sha(root/'dataset'/filename);c['pins'][filename]=digest
        models.append(dict(id=identity,file=filename,sha256=digest))
    c['models']=models
    protocol=json.loads((root/'dataset/protocol.json').read_text());base=protocol['cases'][0]
    protocol.update(schema='continuous-policy-adaptation-pilot-v1',study_variant=VARIANT,
        experiment='ppo_initialization',
        pilot_seeds=[412,413],pilot_steps_each=102400,total_pilot_steps=819200,
        cases=[dict(base,id=name,initial_id=initial)for name,initial in zip(IDS,INITIALS)],
        initialization='Every trial explicitly uses its named policy, with a fresh optimizer; reference ranking is informational only',
        paired_comparison='Same paid continuous TRAIN streams, PPO parameters, rewards and physics; only fixed initial policy varies',
        rationale_source='reports/bank_fit_depth_diagnosis_20261101.md',
        teacher_data_used=False,test_used_for_selection=False,
        limitations=['BC seeds402 chosen using reused development validation; PPO seeds412/413 are new optimizer runs, not independent BC replications',
            'No extra supervised examples or replay; no automatic large continuation',
            'Simulation physics is not fully validated; no claim of live win rate'])
    validate_protocol(protocol)
    # Freeze current dependencies; never edit a previously submitted archive.
    base_files=json.loads((root/'dataset/manifest.json').read_text())['files']
    names=set(c['extensions'])|{str(p)for p in Path('sim').rglob('*.py')}
    names|={name for name in base_files if name.startswith(('sim/','scripts/','data/'))
        and Path(name).is_file()and sha(name)!=base_files[name]}
    with tarfile.open(root/'dataset/extensions.tar.gz','w:gz')as archive:
        for name in sorted(names):
            path=Path(name)
            if path.is_symlink()or path.is_absolute()or '..'in path.parts:raise ValueError('Unsafe source member')
            content=path.read_bytes();info=tarfile.TarInfo(name);info.size=len(content)
            archive.addfile(info,io.BytesIO(content))
    c['extensions']={name:sha(name)for name in sorted(names)}
    c['replaced']={name:base_files[name]for name in sorted(names)if name in base_files}
    c['pins']['extensions.tar.gz']=sha(root/'dataset/extensions.tar.gz')
    (root/'dataset/protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');c['pins']['protocol.json']=sha(root/'dataset/protocol.json')
    tree=ast.parse((root/'job/train.py').read_text());execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    run=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run');changed=0
    for n in run.body:
        if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='run_initial'for t in n.targets):
            n.value=ast.parse("next(m for m in config['models'] if m['id']==trial['initial_id'])",mode='eval').body;changed+=1
    if changed!=1:raise ValueError('Exactly one initial policy assignment required')
    tree.body[-1].value.args=[ast.parse(repr(c),mode='eval').body];ast.fix_missing_locations(tree)
    (root/'job/train.py').write_text(ast.unparse(tree)+'\n')
    prep.update(study_variant=VARIANT,config=c,bootstrap_sha256=sha(root/'job/train.py'),
        initialization_sources={name:str(path)for name,path in zip(INITIALS,(balanced,bank,short,long))})
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n');return prep


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','output','balanced','bank','short','long'):p.add_argument('--'+key,required=True)
    a=p.parse_args();r=prepare(**vars(a));print(json.dumps(dict(kernel=r['kernel'],submitted=False)))
