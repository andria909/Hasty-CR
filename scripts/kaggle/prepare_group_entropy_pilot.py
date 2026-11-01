"""Freeze an isolated conditional spatial-entropy PPO ablation."""
import argparse
import ast
import io
import json
from pathlib import Path
import tarfile

from scripts.kaggle.prepare_motionfix_pilot import prepare as clone,sha

VARIANT='group_entropy_v1'
CASES=(('control',1.),('spatial03',.3),('spatial01',.1),('group_only',0.))


def validate_protocol(p):
    if (p.get('study_variant')!=VARIANT or p.get('pilot_seeds')!=[432,433]
            or p.get('pilot_steps_each')!=102400 or p.get('total_pilot_steps')!=819200
            or len(p['cases'])!=4):raise ValueError('Fixed group-entropy experiment required')
    common=None
    for row,(name,weight)in zip(p['cases'],CASES):
        if (row['id'],row['spatial_entropy_weight'],row['initial_id'])!=(name,weight,'parent'):
            raise ValueError('Entropy grid or common initializer differs')
        fixed={k:v for k,v in row.items()if k not in ('id','spatial_entropy_weight')}
        if common is not None and fixed!=common:raise ValueError('Only spatial entropy weight may vary')
        common=fixed
        expected=dict(entropy_coef=.01,spell_waste_weight=.1,overflow_penalty=.3,tower_support_weight=.15,
            crown_only_weight=0.,dormant_king_weight=0.,missed_fireball_penalty=0.,action_delay_ms=1350,
            validation_action_delay_ms=1350,action_cooldown_ms=0,cannon_center_only=1,tower_damage_all=1,
            source_radius_reach=1,hog_sight_clipping=1,target_lock_in_range=0,tactical_start_fraction=0.,
            warmstart_updates=0,supervised_replay_weight=0.,card_balance_weight=.03,history_frames=1)
        if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Execution/reward contract differs')


def prepare(source,output):
    root=Path(output);clone(source,root,'hastycr-group-entropy1102')
    prep=json.loads((root/'preparation.json').read_text());c=prep['config']
    if prep['study_variant']!='calibrated_spell_cost_v1'or len(c['models'])!=1:
        raise ValueError('Expected common BC800 source')
    c.update(owner='andreiperelaiko',allow_full_training=False)
    p=root/'dataset/protocol.json';protocol=json.loads(p.read_text());base=protocol['cases'][0]
    protocol.update(schema='continuous-group-entropy-pilot-v1',study_variant=VARIANT,experiment='conditional_spatial_entropy',
        pilot_seeds=[432,433],cases=[dict(base,id=name,spatial_entropy_weight=w)for name,w in CASES],
        rationale_source='reports/sampling_greedy_gap_20261102.md',
        paired_comparison='Only conditional spatial entropy coefficient changes; same BC800 initializer, full TRAIN streams and rewards',
        limitations=['Same flat action distribution, PPO log-prob and greedy evaluation; not a new hierarchical actor',
                    'The full entropy control is unchanged mathematically; fewer cells explored need not improve play',
                    'Reused development validation; no automatic full training or live deployment'])
    validate_protocol(protocol);p.write_text(json.dumps(protocol,indent=2)+'\n');c['pins']['protocol.json']=sha(p)
    archive=root/'dataset/extensions.tar.gz'
    with tarfile.open(archive)as stream:
        payload={m.name:stream.extractfile(m).read()for m in stream.getmembers()if m.isfile()}
    manifest=json.loads((root/'dataset/manifest.json').read_text())
    for name in ('sim/train_attack_ppo.py','sim/group_entropy.py'):
        payload[name]=Path(name).read_bytes();c['extensions'][name]=sha(name)
        if name in manifest['files']:c['replaced'][name]=manifest['files'][name]
    with tarfile.open(archive,'w:gz')as stream:
        for name,content in sorted(payload.items()):
            info=tarfile.TarInfo(name);info.size=len(content);stream.addfile(info,io.BytesIO(content))
    c['pins']['extensions.tar.gz']=sha(archive)
    p=root/'job/train.py';tree=ast.parse(p.read_text())
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    run=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    index=next(i for i,n in enumerate(run.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='process'for t in n.targets))
    run.body[index:index]=ast.parse("args+=['--spatial-entropy-weight',str(trial['spatial_entropy_weight'])]").body
    tree.body[-1].value.args=[ast.parse(repr(c),mode='eval').body];ast.fix_missing_locations(tree)
    p.write_text(ast.unparse(tree)+'\n');kernel=c['owner']+'/hastycr-group-entropy1102'
    for file,identity in [('dataset/dataset-metadata.json',c['owner']+'/'+c['slug']),('job/kernel-metadata.json',kernel)]:
        path=root/file;row=json.loads(path.read_text());row.update(id=identity,title=identity.split('/')[-1])
        if file.startswith('job/'):row.update(dataset_sources=[c['owner']+'/'+c['slug']],enable_gpu=True)
        path.write_text(json.dumps(row,indent=2)+'\n')
    prep.update(kernel=kernel,study_variant=VARIANT,bootstrap_sha256=sha(p))
    (root/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    return dict(kernel=kernel,submitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output)))
