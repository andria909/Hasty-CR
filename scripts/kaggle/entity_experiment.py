"""Matched fresh-initialization CNN / entity-transformer PPO pilot.

Standalone study, not an automatic promotion through previous CNN gates.
All architectures receive the same scenarios, rewards and environment budget.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ARCHS = {'cnn': 'factorized_spatial_v3_conditioned',
         'entity64': 'entity_transformer_d64_l2_v1',
         'entity96': 'entity_transformer_d96_l2_v1'}
ACCOUNTS = {'cnn': (0, 'andriashadrin'), 'entity64': (1, 'vertel0'),
            'entity96': (2, 'andreiperelaiko')}
STEPS = 409600
SEEDS = (512, 513)
CASE_OPTIONS = {}
PROFILE = dict(overflow_penalty=.3, missed_fireball_penalty=0., fireball_value_reward=.25,
               reserve_potential_weight=.1, own_hp_penalty=6., barbarian_train_fraction=.35,
               lane_reflection_fraction=.5, action_cooldown_ms=0, cannon_center_only=1,
               tower_damage_all=1, source_radius_reach=1, target_lock_in_range=0,
               hog_sight_clipping=1, shaping_gamma=.999, decision_interval_ms=250)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(architecture, seed, output, profile, *, device='cuda:0', technical=False,
            updates=None, resume=None, evaluation=False, case=None):
    total = 32 if technical else STEPS
    args = [sys.executable, '-c', "import torch,runpy;torch.set_num_threads(1);runpy.run_module('sim.train_attack_ppo',run_name='__main__')",
            '--output', str(output), '--device', device, '--architecture', architecture,
            '--task-mode', 'joint', '--curriculum', 'continuous_v1',
            '--geometry', 'native_anchors_candidate_v1', '--coordinate-mode', 'physical_live_cells_v2',
            '--continuous-profile', str(profile), '--continuous-focus', 'uniform',
            '--validation-selection', 'overall' if technical else 'worst_family',
            '--validation-seed', '900100', '--eval-episodes', '1' if technical else '36',
            '--tower-support-weight', '.15', '--spell-waste-weight', '.3',
            '--crown-only-weight', '0', '--dormant-king-weight', '0',
            '--card-balance-weight', '.03', '--card-balance-target', 'units',
            '--action-delay-ms', '1350', '--max-ms', '310000', '--seed', str(seed),
            '--envs', '2' if technical else '8', '--rollout', '8' if technical else '128',
            '--updates', str(updates if updates is not None else (2 if technical else STEPS//1024)),
            '--target-environment-steps', str(total), '--epochs', '1' if technical else '3',
            '--minibatch-size', '8' if technical else '256', '--lr', '.00005',
            '--gamma', '.999', '--gae-lambda', '.95', '--value-coef', '.25',
            '--entropy-coef', '.01', '--value-warmup-steps', '0', '--target-kl', '.02',
            '--checkpoint-every', '1' if technical else '50',
            '--eval-every', ('2' if evaluation else '0') if technical else '50',
            '--wallclock-seconds', '2400' if technical else '30000']
    if resume:
        args += ['--resume', str(resume)]
    args += CASE_OPTIONS.get(case, [])
    return args


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    predecessor=Path('tmp/kaggle/critic_weight_20261104_reuse_v2')
    prep=json.loads((predecessor/'preparation.json').read_text())
    payload={}
    for filename in ('source.tar.gz','extensions.tar.gz'):
        p=predecessor/'dataset'/filename
        if filename=='source.tar.gz':
            manifest_path=predecessor/'dataset/manifest.json'
            if sha(manifest_path)!=prep['config']['source_manifest_sha256']:
                raise ValueError('Base manifest changed')
            expected=json.loads(manifest_path.read_text())['archive_sha256']
        else:expected=prep['config']['pins'][filename]
        if sha(p)!=expected:raise ValueError('Base archive changed')
        with tarfile.open(p)as stream:
            for m in stream.getmembers():
                if not m.isfile() or Path(m.name).is_absolute() or '..'in Path(m.name).parts:
                    raise ValueError('Unsafe base archive')
                payload[m.name]=stream.extractfile(m).read()
    for folder in ('sim','scripts'):
        for p in Path(folder).rglob('*.py'):
            if '__pycache__'not in p.parts:payload[str(p)]=p.read_bytes()
    p=Path('data/entity_policy_vocab.json');payload[str(p)]=p.read_bytes()
    import re
    for name,content in payload.items():
        if re.search(rb'KGAT_[A-Za-z0-9]{16,}',content):raise ValueError('Secret pattern in package: '+name)
    source=root/'source';source.mkdir()
    for name,content in payload.items():
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
    dataset=root/'dataset';dataset.mkdir()
    archive=dataset/'source.tar.gz'
    with tarfile.open(archive,'w:gz')as stream:
        for name,content in sorted(payload.items()):
            m=tarfile.TarInfo(name);m.size=len(content);stream.addfile(m,io.BytesIO(content))
    manifest=dict(files={n:hashlib.sha256(v).hexdigest()for n,v in sorted(payload.items())},archive_sha256=sha(archive))
    (dataset/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    protocol=dict(schema='entity-architecture-pilot-v1',architectures=ARCHS,seeds=SEEDS,
        steps_each=STEPS,total_steps=6*STEPS,initialization='fresh_random_all_architectures',
        profile=PROFILE,value_coef=.25,lr=.00005,entity_vocabulary_sha256=sha('data/entity_policy_vocab.json'),
        selection='Same72validationgames; macro/worstfamily; report every seed, no automatic promotion',
        test_used=False,live_games=False,live_checkpoint_changed=False,
        limits='Representation and architecture change together; this is not an isolated attention-only ablation.',
        case_options=CASE_OPTIONS)
    (dataset/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (dataset/'profile.json').write_text(json.dumps(PROFILE,indent=2)+'\n')
    (root/'preparation.json').write_text(json.dumps(dict(
        pins={p.name:sha(p)for p in dataset.iterdir()},source=str(source.resolve()),
        source_manifest_sha256=sha(dataset/'manifest.json')),indent=2)+'\n')
    return root


def exact_equal(a,b):
    import numpy as np
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor)and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray)and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys()and all(exact_equal(a[k],b[k])for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b)and len(a)==len(b)and all(exact_equal(x,y)for x,y in zip(a,b))
    return a==b


def preflight(root):
    import torch
    root=Path(root).resolve();source=root/'source';profile=root/'dataset/profile.json'
    runs=root/'preflight';runs.mkdir(exist_ok=False);proof=[]
    for name,arch in ARCHS.items():
        paths={s:runs/(name+'_'+s)for s in ('whole','first','resume')}
        for mode in ('whole','first','resume'):
            cmd=command(arch,SEEDS[0],paths[mode],profile,device='cpu',technical=True,
                updates=1 if mode=='first'else 2,
                resume=paths['first']/'update_000001.pt' if mode=='resume' else None,evaluation=True,case=name)
            with (runs/(name+'_'+mode+'.log')).open('x')as out:
                subprocess.run(cmd,cwd=source,stdout=out,stderr=subprocess.STDOUT,check=True)
        a=torch.load(paths['whole']/'update_000002.pt',map_location='cpu',weights_only=False)
        b=torch.load(paths['resume']/'update_000002.pt',map_location='cpu',weights_only=False)
        keys=('state_dict','optimiser','rng','sampler','observations','infos','episode_counters','episode_returns')
        for key in keys:
            if not exact_equal(a[key],b[key]):raise ValueError('Resume differs: '+name+'/'+key)
        rows=[json.loads(l)for l in (paths['whole']/'metrics.jsonl').read_text().splitlines()]
        row=rows[-1]
        if a['step']!=32 or row['validation']['games']!=2:raise ValueError('Technical budget/eval incomplete')
        proof.append(dict(model=name,steps=32,exact_resume_keys=list(keys),validation_games=2,
                          architecture=a['architecture'],checkpoint_sha256=sha(paths['whole']/'update_000002.pt')))
        print(json.dumps(dict(event='PREFLIGHT_OK',**proof[-1])),flush=True)
    result=dict(technical_only=True,quality_claim=False,proof=proof,
                pins=json.loads((root/'preparation.json').read_text())['pins'])
    (root/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')


def remote(root,case):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    root=Path(root);torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA GPUs required')
    out=Path('/kaggle/working/entity_runs');out.mkdir(exist_ok=False)
    print(json.dumps(dict(event='CUDA_READY',gpus=[torch.cuda.get_device_name(i)for i in range(2)])),flush=True)
    def run(index,seed):
        identity=f'{case}_s{seed}';directory=out/identity
        print(json.dumps(dict(event='TRIAL_START',trial_id=identity,target_steps=STEPS,device=f'cuda:{index}')),flush=True)
        cmd=command(ARCHS[case],seed,directory,root/'profile.json',device=f'cuda:{index}',case=case)
        process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        with (out/(identity+'.log')).open('x')as log:
            for line in process.stdout:
                log.write(line);log.flush()
                try:r=json.loads(line)
                except ValueError:print(line,end='',flush=True);continue
                if 'update'in r and 'step'in r:
                    print(json.dumps(dict(event='TRIAL_PROGRESS',trial_id=identity,step=r['step'],update=r['update'],
                        target_steps=STEPS,percent=r['percent'],actual_cuda=r.get('actual_cuda'))),flush=True)
        if process.wait():raise RuntimeError('Training failed: '+identity)
        boundary=json.loads((directory/'training_result.json').read_text())
        if boundary.get('step')!=STEPS or not boundary.get('target_met'):raise ValueError('Incomplete pilot budget')
        rows=[json.loads(l)for l in (directory/'metrics.jsonl').read_text().splitlines()]
        evaluated=[r for r in rows if 'validation'in r]
        guarded=[r for r in evaluated if r['validation']['worst_archetype_points']>=.1]
        chosen=max(guarded or evaluated,key=lambda r:r['validation']['selection_score'])
        checkpoint=directory/f"update_{chosen['update']:06d}.pt"
        result=dict(id=identity,architecture=ARCHS[case],step=STEPS,seed=seed,
            selected_update=chosen['update'],checkpoint=str(checkpoint.relative_to(out)),checkpoint_sha256=sha(checkpoint),
            family_guard_met=bool(guarded),validation=chosen['validation'])
        (out/(identity+'_summary.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(event='TRIAL_COMPLETE',trial_id=identity,step=STEPS,score=chosen['validation']['selection_score'])),flush=True)
        return result
    with ThreadPoolExecutor(2)as pool:
        futures=[pool.submit(run,i,seed)for i,seed in enumerate(SEEDS)]
        results=[f.result()for f in futures]
    (out/'complete.json').write_text(json.dumps(dict(case=case,steps=2*STEPS,results=results),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','preflight','remote'))
    p.add_argument('--root',required=True);p.add_argument('--case',choices=tuple(ARCHS))
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.root)
    elif a.mode=='preflight':preflight(a.root)
    else:remote(a.root,a.case)


if __name__=='__main__':main()
