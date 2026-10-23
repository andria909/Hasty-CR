"""Train one validation-selected PPO profile to2048000 fresh-phase steps."""
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time

from scripts.kaggle.run_five_round_robin import sha,prepare_source
from scripts.kaggle.run_ppo_coefficient_study import training_command

LAUNCH_CONFIG=None
TARGET=2048000
UPDATES=2000


def verify_script_target_migration(config,inputs,project,manifest):
    """An explicit, narrowly pinned semantic change; never an exact-source resume."""
    migration=config['migration']
    if migration.get('schema')!='script-target-clamp-migration-v1':
        raise ValueError('Unsupported full-training migration')
    old=inputs/'original_manifest.json'
    if sha(old)!=migration['original_manifest_sha256']:
        raise ValueError('Original migration manifest differs')
    original=json.loads(old.read_text())
    if original['archive_sha256']!=migration['original_source_sha256']:
        raise ValueError('Original migration source differs')
    changed={name for name in set(original['files'])|set(manifest['files'])
             if original['files'].get(name)!=manifest['files'].get(name)}
    if changed!={'sim/defense_env.py'}:
        raise ValueError('Migration must change only scripted body targeting')
    new=(project/'sim/defense_env.py').read_text()
    replacement=migration['replacement']
    if replacement!='arena.clamp_to_arena(chosen.pos)' or new.count('return '+replacement)!=1:
        raise ValueError('Unexpected target-clamp migration')
    # Verify the whole original file, including all other environment semantics.
    reverted=new.replace('return '+replacement,'return chosen.pos')
    if hashlib.sha256(reverted.encode()).hexdigest()!=original['files']['sim/defense_env.py']:
        raise ValueError('Migration changes more than the declared target expression')
    if (migration['resume_update'],migration['resume_step'])!=(796,815104):
        raise ValueError('Expected reproduced, committed update796 boundary')
    boundary=inputs/'resume.pt'
    if sha(boundary)!=migration['resume_checkpoint_sha256']:
        raise ValueError('Migration resume checkpoint differs')
    return boundary


def full_command(project,inputs,output,mode,profile,device,seed,resume=None,completed_updates=0):
    if mode not in ('defense','joint') or completed_updates<0 or completed_updates>=UPDATES:
        raise ValueError('Invalid full-training mode or boundary')
    if bool(resume)!=(completed_updates>0):raise ValueError('Resume boundary and checkpoint must be paired')
    command=training_command(project,inputs,output,mode,profile,device,seed)
    def replace(flag,value):command[command.index(flag)+1]=str(value)
    # Defense interprets updates as additional; joint interprets them as final update.
    replace('--updates',UPDATES-completed_updates if mode=='defense' else UPDATES)
    replace('--checkpoint-every',100);replace('--wallclock-seconds',18000)
    if mode=='joint':replace('--target-environment-steps',TARGET)
    if resume:
        if mode=='joint':
            index=command.index('--init');del command[index:index+2]
        command+=['--resume',str(resume)]
    return command


def execute(config):
    owner,slug=config['dataset'].split('/')
    candidates=[p for p in (Path('/kaggle/input')/slug,Path('/kaggle/input/datasets')/owner/slug) if p.is_dir()]
    if len(candidates)!=1:raise ValueError('Expected one exact full-training input mount')
    inputs=candidates[0]
    for name,digest in config['pins'].items():
        if name not in ('manifest.json','teacher_defense.pt','teacher_joint.pt','original_manifest.json','resume.pt') or sha(inputs/name)!=digest:
            raise ValueError('Full-training input pin mismatch')
    output=Path('/kaggle/working/selected_full');output.mkdir(exist_ok=False)
    project=output/'project';manifest=prepare_source(inputs,project,config['pins']['manifest.json'])
    protocol=json.loads((project/'data/ppo_tuning/protocol.json').read_text())
    if protocol['full_environment_steps']!=TARGET:raise ValueError('Full-training budget differs from protocol')
    mode,profile=config['mode'],config['profile'];selection=config['selection']
    resume=None;completed_updates=0
    selection_source=manifest['archive_sha256']
    if 'migration'in config:
        if (mode,profile)!=('defense','economy_soft'):raise ValueError('Unexpected migrated full-training slot')
        resume=verify_script_target_migration(config,inputs,project,manifest)
        completed_updates=config['migration']['resume_update']
        selection_source=config['migration']['original_source_sha256']
    if (selection.get('stage')!='confirmed' or selection.get('source_sha256')!=selection_source or
        selection.get('protocol_sha256')!=sha(project/'data/ppo_tuning/protocol.json') or
        selection.get('test_used_for_selection') is not False or
        profile not in selection['modes'][mode]['full_training_profiles']):
        raise ValueError('Only confirmed validation-selected coefficients may train')
    sys.path[:0]=[str(project),str(project/'scripts')]
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU training fallback')
    seed=protocol['screen_training_seed'];device='cuda:0'
    report=dict(schema='selected-ppo-full-training-v1',status='running',mode=mode,profile=profile,seed=seed,
                target_environment_steps=TARGET,source_sha256=manifest['archive_sha256'],
                selection_sha256=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest(),
                actual_cuda=torch.cuda.get_device_name(0),test_used_for_selection=False,quality_accepted=False)
    if resume:
        boundary=torch.load(resume,map_location='cpu',weights_only=False)
        if (boundary['update'],boundary['step'])!=(796,815104)or not boundary.get('exact_resume_available'):
            raise ValueError('Migration boundary lacks complete saved optimizer/RNG/environment state')
        if torch.cuda.device_count()!=2 or len(boundary['rng']['cuda'])!=2 or len(boundary['environment_states'])!=8:
            raise ValueError('Migration must retain both CUDA RNG streams and eight environment snapshots')
        report.update(migration=config['migration'],source_unchanged=False,
                      same_source_exact_resume=False,resumed_environment_states=True)
    def save():(output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save();print(json.dumps(report),flush=True)
    environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    def logged(command,name):
        with (output/name).open('x') as stream:
            process=subprocess.Popen(command,cwd=project,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                stream.write(line);stream.flush()
                try:row=json.loads(line)
                except ValueError:continue
                if not isinstance(row,dict):continue
                step=row.get('step',row.get('steps',row.get('environment_steps')))
                if isinstance(step,int) and 'update' in row:
                    print(json.dumps(dict(event='FULL_TRAINING_PROGRESS',mode=mode,profile=profile,step=step,update=row['update'],percent=min(100,100*step/TARGET))),flush=True)
            if process.wait():raise RuntimeError('Full-training subprocess failed: '+name)
    try:
        logged(full_command(project,inputs,output/'train',mode,profile,device,seed,
                            resume=resume,completed_updates=completed_updates),'train.log')
        checkpoints=sorted((output/'train').glob('update_*.pt'))
        if not checkpoints:raise ValueError('Trainer produced no recoverable checkpoint')
        checkpoint=checkpoints[-1];blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if blob['config'].get('init_sha256')!=config['pins']['teacher_'+mode+'.pt']:raise ValueError('Full-training teacher changed')
        if not all(bool(torch.isfinite(v).all()) for v in blob['state_dict'].values()):raise ValueError('Nonfinite trained policy')
        report.update(step=blob['step'],update=blob['update'],checkpoint=str(checkpoint.relative_to(output)),checkpoint_sha256=sha(checkpoint),
                      exact_resume_available=bool(blob.get('exact_resume_available')))
        if blob['step']!=TARGET or blob['update']!=UPDATES:
            report['status']='incomplete_budget';save();print(json.dumps(report),flush=True);return report
        logged([sys.executable,'-m','sim.ppo_tuning','--checkpoint',str(checkpoint),'--profile',str(project/'data/ppo_tuning/profiles'/(profile+'.json')),
                '--mode',mode,'--device',device,'--contexts',str(project/'data/ppo_tuning/validation_contexts.json'),
                '--expected-steps',str(TARGET),'--output',str(output/'validation.json')],'validation.log')
        validation=json.loads((output/'validation.json').read_text())
        report.update(status='completed',validation_sha256=sha(output/'validation.json'),quality=validation['quality'])
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise
    save();print(json.dumps(report),flush=True);return report


if __name__=='__main__':
    if LAUNCH_CONFIG is None:raise RuntimeError('Verified coefficient selection required')
    execute(LAUNCH_CONFIG)
