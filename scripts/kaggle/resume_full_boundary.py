"""Resume one verified PPO boundary under its frozen project, without resetting budget.

Runs on the selected device but does not upload or submit Kaggle jobs. The caller
must supply a SHA-pinned specification. Invoke this file with PYTHONPATH set to
the original unpacked project so imports and checkpoint classes use frozen code.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read_metrics(path,start=0):
    rows=[];lines=Path(path).read_text().splitlines(keepends=True)
    for i,line in enumerate(lines):
        try:row=json.loads(line)
        except ValueError:
            if i==len(lines)-1 and not line.endswith('\n'):break
            raise ValueError('Corrupt interior metrics')
        if not isinstance(row,dict)or type(row.get('update'))is not int or row['update']!=start+len(rows)+1:
            raise ValueError('Metrics update gap or duplicate')
        rows.append(row)
    return rows


def selected_boundary(rows):
    validations=[r for r in rows if 'validation'in r]
    for row in validations:
        v=row['validation']
        if not isinstance(v,dict)or any(type(v.get(k))not in(int,float)or not math.isfinite(v[k])
                for k in('selection_score','worst_archetype_points')):
            raise ValueError('Invalid validation score')
    guarded=[r for r in validations if r['validation']['worst_archetype_points']>=.1]
    return (max(guarded or validations,key=lambda r:r['validation']['selection_score'])if validations else None,bool(guarded))


def flag(command,name):
    if command.count(name)!=1:raise ValueError('Missing or duplicated command flag: '+name)
    return command[command.index(name)+1]


def resumed_command(command,checkpoint,output):
    result=list(command)
    if '--resume'in result:raise ValueError('Expected original initialization command')
    flag(result,'--init');flag(result,'--output')
    i=result.index('--init');result[i:i+2]=['--resume',str(checkpoint)]
    result[result.index('--output')+1]=str(output)
    return result


def checked_inputs(spec):
    source=Path(spec['source']).resolve();paths={}
    for name,digest in spec['files'].items():
        path=source/name
        if Path(name).is_absolute()or '..'in Path(name).parts or path.is_symlink()or not path.resolve().is_relative_to(source):
            raise ValueError('Unsafe recovery source')
        if sha(path)!=digest:raise ValueError('Recovery source changed: '+name)
        paths[name]=path
    required={'config.json','metrics.jsonl',spec['checkpoint']}
    if not required<=paths.keys():raise ValueError('Missing pinned boundary inputs')
    return paths


def audit(spec):
    import torch
    from sim.train_attack_ppo import parser,validate_resume,code_fingerprint,load_reward_profile
    paths=checked_inputs(spec)
    config=json.loads(paths['config.json'].read_text())
    blob=torch.load(paths[spec['checkpoint']],map_location='cpu',weights_only=False)
    if config!=blob['config']:raise ValueError('Saved config and checkpoint differ')
    validate_resume(blob,config)
    command=spec['command']
    if command[1:3]!=['-c',"import torch,runpy;torch.set_num_threads(1);runpy.run_module('sim.train_attack_ppo',run_name='__main__')"]:
        raise ValueError('Unexpected frozen trainer invocation')
    args=vars(parser().parse_args(command[3:]))
    # Compare all original command options, including eval/checkpoint schedules,
    # not only the subset checked by the trainer's own resume guard.
    for key,value in args.items():
        if key in('init','resume','output','continuous_profile','reward_profile','wallclock_seconds'):continue
        if value!=config.get(key):raise ValueError('Original command differs: '+key)
    if args['resume']is not None or sha(args['init'])!=config['init_sha256']:
        raise ValueError('Original initialization differs')
    rewards,reward_sha=load_reward_profile(args['reward_profile'])
    if rewards!=config['reward_profile']or reward_sha!=config['reward_profile_sha256']:
        raise ValueError('Original reward profile differs')
    profile=json.loads(Path(args['continuous_profile']).read_text())
    if any(config['continuous_profile'].get(k)!=v for k,v in profile.items()):
        raise ValueError('Original profile differs')
    current=code_fingerprint(config['task_mode'],config['curriculum'],config['card_balance_weight'],
        config['history_frames'],config.get('pending_inputs',False),config['tower_support_weight'],
        config['spell_waste_weight'],config['tactical_start_fraction'],config['crown_only_weight'],
        config['dormant_king_weight'],config['spatial_entropy_weight'])
    if current!=config['source_code_sha256']:raise ValueError('Frozen trainer or simulator changed')
    target=config['target_environment_steps'];per_update=config['envs']*config['rollout']
    if type(target)is not int or target<=0 or target%per_update or config['updates']!=target//per_update:
        raise ValueError('Inconsistent original full target')
    technical=spec.get('technical_fixture',False)
    if not math.isfinite(args['wallclock_seconds'])or not 0<args['wallclock_seconds']<=(600 if technical else 34000):
        raise ValueError('Recovery wallclock exceeds session cap')
    if technical:
        if config['device']!='cpu' or target>1024:raise ValueError('CPU-only bounded technical fixture required')
    elif target!=4096000 or per_update!=1024 or config['device']not in('cuda:0','cuda:1'):
        raise ValueError('Actual full recovery requires unchanged 4,096,000-step CUDA budget')
    update=blob['update'];step=blob['step']
    if type(update)is not int or not 1<=update<=config['updates'] or step!=update*per_update:
        raise ValueError('Checkpoint counters differ')
    if spec['checkpoint']!=f'update_{update:06d}.pt':raise ValueError('Checkpoint filename and counters differ')
    rows=read_metrics(paths['metrics.jsonl'])
    if len(rows)<update:raise ValueError('Checkpoint has no complete logged boundary')
    if len(rows)>config['updates']:raise ValueError('Metrics exceed original target')
    rows=rows[:update]
    for i,row in enumerate(rows,1):
        if row.get('step')!=i*per_update or row.get('target_environment_steps')!=target:
            raise ValueError('Logged budget differs from checkpoint')
        if not technical and not row.get('actual_cuda'):raise ValueError('Missing actual CUDA evidence')
    if not all(torch.isfinite(t).all()for t in blob['state_dict'].values()):raise ValueError('Nonfinite model weights')
    chosen,guarded=selected_boundary(rows)
    if chosen:
        name=f"update_{chosen['update']:06d}.pt"
        if name not in paths:raise ValueError('Historical selected checkpoint not pinned')
        old=torch.load(paths[name],map_location='cpu',weights_only=False)
        validate_resume(old,config)
        if old['update']!=chosen['update']or old['step']!=chosen['step']:
            raise ValueError('Historical selected checkpoint counters differ')
        if any(old['config'].get(k)!=config.get(k)for k in('target_environment_steps','updates','init_sha256','architecture')):
            raise ValueError('Historical selected checkpoint belongs to another run')
        if not all(torch.isfinite(t).all()for t in old['state_dict'].values()):raise ValueError('Nonfinite historical weights')
    return dict(paths=paths,config=config,rows=rows,update=update,step=step,target=target,
        per_update=per_update,selected=chosen,guarded=guarded,technical=technical)


def execute(spec,output):
    state=audit(spec);output=Path(output).absolute()
    if output.exists():raise FileExistsError('Preserve previous recovery attempt')
    config=state['config'];paths=state['paths'];completed=state['step']==state['target']
    if completed:
        if 'training_result.json'not in paths:raise ValueError('Completed sibling result must be pinned')
        result=json.loads(paths['training_result.json'].read_text())
        if not result.get('target_met')or result.get('step')!=state['target']:
            raise ValueError('Completed sibling result differs')
        output.mkdir(parents=True)
        for name in('config.json','training_result.json'):shutil.copyfile(paths[name],output/name)
        # Keep the original raw log too, including any ignored trailing bytes.
        shutil.copyfile(paths['metrics.jsonl'],output/'previous_metrics.jsonl')
        rows=state['rows'];returncode=0;command=None
    else:
        command=resumed_command(spec['command'],paths[spec['checkpoint']],output)
        # The trainer creates a new output directory. Keep the driver log outside it.
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.with_name(output.name+'_driver.log').open('x')as log:
            result=subprocess.run(command,cwd=Path.cwd(),env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        returncode=result.returncode
        if not output.exists():raise RuntimeError('Resume exited before creating output; inspect driver log')
        shutil.copyfile(paths['metrics.jsonl'],output/'previous_metrics.jsonl')
        suffix=read_metrics(output/'metrics.jsonl',state['update'])if(output/'metrics.jsonl').exists()else[]
        for i,row in enumerate(suffix,state['update']+1):
            if row.get('step')!=i*state['per_update']or row.get('target_environment_steps')!=state['target']:
                raise ValueError('Resume changed total budget or counters')
            if not state['technical']and not row.get('actual_cuda'):raise ValueError('Missing resumed CUDA evidence')
        rows=state['rows']+suffix
        if len(rows)>config['updates']:raise ValueError('Resume exceeded total budget')
        if(output/'metrics.jsonl').exists():(output/'metrics.jsonl').rename(output/'resume_segment_metrics.jsonl')
    # Historical checkpoints remain candidates for final selection. Never rewrite
    # either the downloaded source or a newly produced checkpoint.
    for name,path in paths.items():
        if name.endswith('.pt'):
            target=output/name
            if target.exists():
                if sha(target)!=sha(path):raise ValueError('Conflicting checkpoint at same update')
            else:shutil.copyfile(path,target)
    with(output/'metrics.jsonl').open('x')as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    chosen,guarded=selected_boundary(rows)
    selected=f"update_{chosen['update']:06d}.pt"if chosen else None
    if selected and not(output/selected).exists():raise ValueError('Combined selection checkpoint missing')
    result_path=output/'training_result.json'
    result=json.loads(result_path.read_text())if result_path.exists()else{}
    last_step=rows[-1]['step'];last_update=rows[-1]['update']
    budget_met=(returncode==0 and result.get('target_met')is True and result.get('step')==state['target']
        and last_step==state['target'] and last_update==config['updates'])
    report=dict(schema='full-boundary-continuation-v1',technical_fixture=state['technical'],
        source_checkpoint_sha256=sha(paths[spec['checkpoint']]),command=command,
        completed_sibling_reused=completed,optimizer_process_started=not completed,
        previous_steps=state['step'],new_steps=last_step-state['step'],total_steps=last_step,
        target_steps=state['target'],total_budget_met=budget_met,returncode=returncode,
        selected_checkpoint=selected,selected_with_family_guard=guarded,
        quality_accepted=False,submission_allowed=False)
    (output/'continuation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(execute(json.loads(Path(a.spec).read_text()),a.output)))
