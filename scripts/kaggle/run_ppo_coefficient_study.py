"""Two-GPU validation-only PPO coefficient screening; fixed102400 steps/trial."""
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

from scripts.kaggle.run_five_round_robin import sha,prepare_source

LAUNCH_CONFIG=None


def training_command(project,inputs,output,mode,profile,device,seed):
    module='sim.train_defense_ppo'if mode=='defense'else'sim.train_attack_ppo'
    args=[sys.executable,'-m',module,'--init',str(inputs/('teacher_'+mode+'.pt')),'--output',str(output),
        '--device',device,'--reward-profile',str(project/'data/ppo_tuning/profiles'/(profile+'.json')),
        '--updates','100','--envs','8','--rollout','128','--epochs','3','--minibatch-size','256',
        '--lr','0.00005','--gamma','0.997','--gae-lambda','0.95','--clip','0.2','--value-coef','0.5','--entropy-coef','0.01',
        '--seed',str(seed),'--eval-every','0','--checkpoint-every','100','--wallclock-seconds','7200',
        '--max-ms','240000','--action-delay-ms','150','--geometry','legacy_v1']
    if mode=='joint':args+=['--task-mode','joint','--target-environment-steps','102400','--initial-hand-mode','shuffle','--deadline-scale','3','--visible-fraction','0.5','--coordinate-mode','physical_live_cells_v2']
    else:args+=['--coordinate-mode','physical_live']
    return args


def execute(config):
    def mount(ref):
        owner,slug=ref.split('/');found=[p for p in (Path('/kaggle/input')/slug,Path('/kaggle/input/datasets')/owner/slug)if p.is_dir()]
        if len(found)!=1:raise ValueError('Expected one exact coefficient-study input mount')
        return found[0]
    inputs=mount(config['dataset']);output=Path('/kaggle/working/coefficient_study');output.mkdir(exist_ok=False)
    for name,digest in config['pins'].items():
        if name not in ('manifest.json','teacher_defense.pt','teacher_joint.pt')or sha(inputs/name)!=digest:
            raise ValueError('Pinned coefficient-study input SHA mismatch')
    project=output/'project';manifest=prepare_source(inputs,project,config['pins']['manifest.json'])
    protocol=json.loads((project/'data/ppo_tuning/protocol.json').read_text())
    if protocol['short_environment_steps']!=102400 or protocol['full_environment_steps']!=2048000:
        raise ValueError('Fixed1/20 screening budget changed')
    if sha(project/'data/ppo_tuning/validation_contexts.json')!=protocol['validation']['contexts_sha256']:
        raise ValueError('Fixed validation contexts changed')
    sys.path[:0]=[str(project),str(project/'scripts')]
    import torch
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices required; no CPU training fallback')
    report=dict(schema='ppo-coefficient-study-run-v1',status='running',source_sha256=manifest['archive_sha256'],
        gpu_names=[torch.cuda.get_device_name(i)for i in range(2)],training_steps_per_trial=102400,
        full_training_steps=2048000,test_used_for_search=False,quality_accepted=False,trials=[])
    def save():(output/'study_result.json').write_text(json.dumps(report,indent=2)+'\n')
    save();print(json.dumps(report),flush=True)
    environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    def logged(command,log,metadata):
        with log.open('x')as stream:
            process=subprocess.Popen(command,cwd=project,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                stream.write(line);stream.flush()
                try:row=json.loads(line)
                except ValueError:continue
                if not isinstance(row,dict):continue
                step=row.get('step',row.get('steps',row.get('environment_steps')))
                if isinstance(step,int)and 'update'in row:
                    print(json.dumps(dict(event='TRIAL_PROGRESS',**metadata,step=step,update=row['update'],percent=min(100,100*step/102400))),flush=True)
                elif row.get('event')=='TRIAL_VALIDATION':print(json.dumps(dict(row,**metadata)),flush=True)
            code=process.wait()
        if code:raise RuntimeError('Trial subprocess failed; inspect '+str(log))
    def worker(index):
        device=f'cuda:{index}';rows=[]
        for trial in config['workers'][index]:
            mode,profile,seed=trial['mode'],trial['profile'],trial.get('seed',42)
            label=f'{mode}_{profile}_s{seed}';folder=output/label;folder.mkdir()
            row=dict(mode=mode,profile=profile,seed=seed,device=device,trial_id=label,status='started')
            started=time.monotonic();print(json.dumps(dict(event='TRIAL_STARTED',**row)),flush=True)
            try:
                command=training_command(project,inputs,folder/'train',mode,profile,device,seed)
                logged(command,folder/'train.log',dict(trial_id=label,mode=mode,profile=profile))
                checkpoint=folder/'train/update_000100.pt'
                blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
                if blob.get('step')!=102400 or blob.get('update')!=100:raise ValueError('Incomplete screening train budget')
                if blob['config'].get('init_sha256')!=config['pins']['teacher_'+mode+'.pt']:
                    raise ValueError('Trial initialization differs from the paired teacher')
                if not all(bool(torch.isfinite(v).all())for v in blob['state_dict'].values()):raise ValueError('Nonfinite trial policy')
                validation=folder/'validation.json'
                logged([sys.executable,'-m','sim.ppo_tuning','--checkpoint',str(checkpoint),
                    '--profile',str(project/'data/ppo_tuning/profiles'/(profile+'.json')),'--mode',mode,
                    '--device',device,'--contexts',str(project/'data/ppo_tuning/validation_contexts.json'),'--output',str(validation)],
                    folder/'validation.log',dict(trial_id=label,mode=mode,profile=profile))
                measured=json.loads(validation.read_text())
                row.update(status='completed',training_steps=blob['step'],checkpoint=str(checkpoint.relative_to(output)),
                    checkpoint_sha256=sha(checkpoint),reward_profile_sha256=measured['reward_profile_sha256'],
                    validation=str(validation.relative_to(output)),validation_sha256=sha(validation),quality=measured['quality'])
            except Exception as error:
                row.update(status='failed',error_type=type(error).__name__,error=str(error))
            row['elapsed_seconds']=time.monotonic()-started
            (folder/'trial_result.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
            print(json.dumps(dict(event='TRIAL_COMPLETED',**row)),flush=True)
        return rows
    with ThreadPoolExecutor(2)as pool:
        futures=[pool.submit(worker,i)for i in (0,1)]
        for future in futures:
            report['trials']+=future.result();save()
    report['status']='completed'if all(r['status']=='completed'for r in report['trials'])else'completed_with_failures'
    save();print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    if LAUNCH_CONFIG is None:raise RuntimeError('Prepared pinned screening configuration required')
    execute(LAUNCH_CONFIG)
