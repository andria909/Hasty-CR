"""Pinned two-GPU coefficient search followed by two long full-game PPO runs."""
import argparse
import inspect
import json
from pathlib import Path
import tarfile

from scripts.kaggle.run_five_round_robin import sha,prepare_source


def execute(config):
    import subprocess,sys
    from concurrent.futures import ThreadPoolExecutor
    def mount(slug):
        root=Path('/kaggle/input')
        candidates=[p for p in (root/slug,root/'datasets'/config['owner']/slug)if p.is_dir()]
        if len(candidates)!=1:raise ValueError('Ambiguous input mount')
        return candidates[0]
    source=mount(config['source_slug']);inputs=mount(config['slug'])
    for name,digest in config['source_pins'].items():
        if sha(source/name)!=digest:raise ValueError('Original source pin mismatch')
    if sha(inputs/'protocol.json')!=config['protocol_sha256']:raise ValueError('Protocol pin mismatch')
    project=Path('/kaggle/working/continuous_project')
    manifest=prepare_source(source,project,config['source_pins']['manifest.json'])
    for name,digest in config['replaced'].items():
        if sha(project/name)!=digest:raise ValueError('Unexpected source migration')
    if (inputs/'extensions.tar.gz').exists():
        if sha(inputs/'extensions.tar.gz')!=config['extensions_archive_sha256']:raise ValueError('Extension archive changed')
        with tarfile.open(inputs/'extensions.tar.gz')as stream:
            members=stream.getmembers()
            if {m.name for m in members}!=set(config['extensions']) or len(members)!=len(config['extensions']):raise ValueError('Unexpected extension members')
            if any(not m.isfile() or PurePosixPath(m.name).is_absolute() or '..'in PurePosixPath(m.name).parts for m in members):raise ValueError('Unsafe archive')
            stream.extractall(project,filter='data')
    else:
        base=(inputs/'extensions').resolve()
        for name in config['extensions']:
            original=base/name;target=project/name
            if original.is_symlink()or not original.resolve().is_relative_to(base):raise ValueError('Unsafe expanded extension')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
    for name,digest in config['extensions'].items():
        if sha(project/name)!=digest:raise ValueError('Migrated source changed')
    sys.path.insert(0,str(project))
    import torch
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices required')
    output=Path('/kaggle/working/continuous_ppo');output.mkdir(exist_ok=False)
    print(json.dumps(dict(event='INPUTS_VERIFIED',source_manifest_sha256=config['source_pins']['manifest.json'],
        extensions=config['extensions'],protocol_sha256=config['protocol_sha256'])),flush=True)
    print(json.dumps(dict(event='CUDA_READY',gpus=[torch.cuda.get_device_name(i)for i in range(2)])),flush=True)
    trials=config['trials'];records=[]
    def run(trial,device,seed,full=False):
        stage='full' if full else 'pilot'
        run_id=f"{stage}_{trial['id']}_s{seed}"
        directory=output/run_id
        profile=output/(run_id+'_profile.json')
        profile.write_text(json.dumps(dict(overflow_penalty=trial['leak'],action_cooldown_ms=2100)))
        steps=4096000 if full else 102400
        args=[sys.executable,'-c',"import torch,runpy; torch.set_num_threads(1); runpy.run_module('sim.train_attack_ppo',run_name='__main__')",
            '--init',str(source/config.get('parent_file','parent.pt')),'--output',str(directory),'--device',device,
            '--task-mode','joint','--curriculum','continuous_v1',
            '--geometry','native_anchors_candidate_v1','--coordinate-mode','physical_live_cells_v2',
            '--continuous-profile',str(profile),'--action-delay-ms','225','--max-ms','310000',
            '--seed',str(seed),'--envs','8','--rollout','128','--updates',str(steps//1024),
            '--target-environment-steps',str(steps),'--epochs','3','--minibatch-size','256',
            '--lr','0.00005','--gamma','.999','--value-warmup-steps','20480','--target-kl','.02',
            '--card-balance-weight',str(trial['balance']),'--card-balance-target','units',
            '--checkpoint-every','250' if full else '100',
            '--eval-every','250' if full else '50','--eval-episodes','24',
            '--wallclock-seconds','34000' if full else '7200']
        process=subprocess.Popen(args,cwd=project,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        print(json.dumps(dict(event='TRIAL_START',trial_id=run_id,target_steps=steps,device=device)),flush=True)
        with (output/(run_id+'.log')).open('x')as log:
            for line in process.stdout:
                log.write(line);log.flush()
                try:r=json.loads(line)
                except ValueError:
                    print(line,end='',flush=True);continue
                if isinstance(r,dict)and 'update'in r and 'actual_cuda'in r:
                    print(json.dumps(dict(event='TRIAL_PROGRESS',trial_id=run_id,update=r['update'],
                        step=r['step'],percent=r['percent'],actual_cuda=r['actual_cuda'])),flush=True)
        if process.wait():raise RuntimeError('Training failed: '+run_id)
        result=json.loads((directory/'training_result.json').read_text())
        if not result.get('target_met'):raise RuntimeError('Training budget not completed: '+run_id)
        validations=[json.loads(line)for line in (directory/'metrics.jsonl').read_text().splitlines()]
        validations=[r for r in validations if 'validation'in r]
        if not validations:raise RuntimeError('No validation results')
        chosen=max(validations,key=lambda r:r['validation']['selection_score'])
        best=directory/f"best_validation_{chosen['update']:06d}.pt"
        # A tie retains the first best-save, exactly as the trainer does.
        if not best.exists():raise RuntimeError('Selected best checkpoint missing')
        record=dict(id=run_id,trial=trial,seed=seed,stage=stage,target_steps=steps,
            target_met=True,checkpoint=str(best.relative_to(output)),checkpoint_sha256=sha(best),
            validation=chosen['validation'],quality_accepted=False)
        (output/(run_id+'_summary.json')).write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(event='TRIAL_COMPLETE',trial_id=run_id,score=chosen['validation']['selection_score'])),flush=True)
        return record
    def pilots(device,assigned):
        return [run(trial,device,seed)for trial in assigned for seed in (42,43)]
    with ThreadPoolExecutor(2)as pool:
        a=pool.submit(pilots,'cuda:0',trials[::2]);b=pool.submit(pilots,'cuda:1',trials[1::2])
        records=a.result()+b.result()
    ranks=[]
    for trial in trials:
        runs=[r for r in records if r['trial']['id']==trial['id']]
        score=sum(r['validation']['selection_score']for r in runs)/len(runs)
        ranks.append(dict(trial=trial,score=score))
    ranks.sort(key=lambda x:(-x['score'],x['trial']['balance'],x['trial']['leak']))
    (output/'selection.json').write_text(json.dumps(dict(ranking=ranks,selected=ranks[:2],
        rule='mean of best validation points over two independent training seeds; ties prefer weaker shaping',
        test_used_for_selection=False),indent=2)+'\n')
    with ThreadPoolExecutor(2)as pool:
        futures=[pool.submit(run,r['trial'],f'cuda:{i}',44,True)for i,r in enumerate(ranks[:2])]
        full=[f.result()for f in futures]
    result=dict(schema='continuous-ppo-sweep-and-full-v1',status='complete',
        pilot_steps=819200,full_steps=8192000,all_target_budgets_met=True,
        selected=ranks[:2],full_runs=full,quality_accepted=False,live_checkpoint_changed=False,
        protocol_sha256=config['protocol_sha256'])
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='CONTINUOUS_TRAINING_COMPLETE',quality_accepted=False)),flush=True)


def prepare(output,owner='andriashadrin',source_slug='hastycr-fireball-balanced-inputs-20261025',parent_file='parent.pt'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dataset=output/'dataset';dataset.mkdir();job=output/'job';job.mkdir()
    frozen=Path('tmp/kaggle/fireball_balanced_20261025/dataset')
    manifest=json.loads((frozen/'manifest.json').read_text())
    names=('sim/match.py','sim/train_attack_ppo.py','sim/continuous_push.py','sim/card_balance.py',
        'sim/env.py','sim/kite_geometry.py','sim/projectile_calibration.py','sim/entities.py',
        'sim/gamedata.py','sim/public_gamedata.py','sim/spells.py','sim/continuous_validation.py','sim/continuous_sampling.py')
    extensions={n:sha(n)for n in names}
    with tarfile.open(dataset/'extensions.tar.gz','w:gz')as stream:
        for name in names:stream.add(name,arcname=name,recursive=False)
    trials=[dict(id='control',leak=0.,balance=0.),dict(id='leak',leak=.15,balance=0.),
            dict(id='balanced',leak=.15,balance=.1),dict(id='strong',leak=.35,balance=.3)]
    protocol=dict(schema='continuous-ppo-sweep-and-full-protocol-v1',trials=trials,
        pilot_seeds=[42,43],pilot_steps_each=102400,full_seed=44,full_steps_each=4096000,
        full_models=2,total_target_steps=9011200,parent_sha256=sha(frozen/'parent.pt'),
        original_source_archive_sha256=manifest['archive_sha256'],extensions=extensions,
        selection='validation full-game points, two seeds, never test reward',
        enemy_rules='normal four-card hand, eight-card deck/cycle, normal income and costs, no bonus elixir',
        opponents='six archetypes, random-legal and reactive scripted styles; validation/test swap deck cards and use disjoint seeds',
        observation_contract='joint_public_presence_continuous_slots_v2',input_scalars=309,
        timing=dict(command_delay_ms=225,used_slot_refill_delay_ms=2100,decision_interval_ms=250,
            global_post_execution_cooldown_ms=0,
            basis='1.5x measured input path; ACK/refill blocks only the used slot, other cards remain playable with reserved mana'),
        physics_limits=['Projectile XY calibration only, targeting/impact/levels still require independent validation',
            'Close kite candidates do not prove all enemy units can be pulled',
            'No live games authorized by training run'],fresh_optimizer_per_run=True,
        value_head_only_warmup_steps=20480,target_kl=.02,quality_accepted=False,live_checkpoint_changed=False)
    (dataset/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    slug='hastycr-continuous-slots-inputs-20261026'
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug,title=slug,licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    config=dict(owner=owner,slug=slug,source_slug=source_slug,parent_file=parent_file,
        source_pins={'manifest.json':sha(frozen/'manifest.json'),parent_file:sha(frozen/'parent.pt')},
        extensions=extensions,replaced={n:manifest['files'][n]for n in names if n in manifest['files']},
        extensions_archive_sha256=sha(dataset/'extensions.tar.gz'),protocol_sha256=sha(dataset/'protocol.json'),trials=trials)
    code='import hashlib,json,shutil,tarfile\nfrom pathlib import Path,PurePosixPath\n'+inspect.getsource(sha)+'\n'+inspect.getsource(prepare_source)+'\n'+inspect.getsource(execute)+'\nexecute('+repr(config)+')\n'
    compile(code,'train.py','exec');(job/'train.py').write_text(code)
    kernel=owner+'/hastycr-continuous-slots-20261026'
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title='hastycr continuous slots 20261026',
        code_file='train.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,
        dataset_sources=[owner+'/'+slug,owner+'/'+config['source_slug']],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    (output/'preparation.json').write_text(json.dumps(dict(kernel=kernel,config=config,bootstrap_sha256=sha(job/'train.py'),submitted=False),indent=2)+'\n')
    return dict(kernel=kernel,target_steps=9011200,submitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--owner',default='andriashadrin')
    p.add_argument('--source-slug',default='hastycr-fireball-balanced-inputs-20261025');p.add_argument('--parent-file',default='parent.pt')
    a=p.parse_args();print(json.dumps(prepare(a.output,a.owner,a.source_slug,a.parent_file)))
