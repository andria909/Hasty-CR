"""Freeze the next short-to-full cohort; bind weights only after v2 finishes."""
import argparse
import inspect
import json
from pathlib import Path
import shutil
import tarfile

from scripts.kaggle.run_five_round_robin import sha,prepare_source


def execute(config):
    import subprocess,sys
    from concurrent.futures import ThreadPoolExecutor
    def mount(slug):
        root=Path('/kaggle/input');paths=[p for p in(root/slug,root/'datasets'/config['owner']/slug)if p.is_dir()]
        if len(paths)!=1:raise ValueError('Ambiguous dataset mount')
        return paths[0]
    inputs=mount(config['slug']);source=mount(config['source_slug'])
    for name,digest in config['pins'].items():
        if name=='extensions.tar.gz'and not(inputs/name).exists():continue
        if sha(inputs/name)!=digest:raise ValueError('Frozen focus input differs')
    project=Path('/kaggle/working/focus_project')
    prepare_source(source,project,config['source_manifest_sha256'])
    for name,digest in config['replaced'].items():
        if sha(project/name)!=digest:raise ValueError('Unexpected source migration')
    if(inputs/'extensions.tar.gz').exists():
        with tarfile.open(inputs/'extensions.tar.gz')as stream:
            members=stream.getmembers()
            if len(members)!=len(config['extensions'])or{m.name for m in members}!=set(config['extensions']):raise ValueError('Unexpected extension set')
            if any(not m.isfile()or PurePosixPath(m.name).is_absolute()or '..'in PurePosixPath(m.name).parts for m in members):raise ValueError('Unsafe extension')
            stream.extractall(project,filter='data')
    else:
        base=(inputs/'extensions').resolve()
        for name in config['extensions']:
            original=base/name;target=project/name
            if original.is_symlink()or not original.resolve().is_relative_to(base):raise ValueError('Unsafe expanded extension')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
    for name,digest in config['extensions'].items():
        if sha(project/name)!=digest:raise ValueError('Focus source differs')
    sys.path.insert(0,str(project))
    import torch
    from sim.train_attack_ppo import initialize_attack_policy,evaluate
    from sim.continuous_push import CONTRACT
    from scripts.attack.select_continuous_focus import select_initial,select_profiles
    torch.set_num_threads(1)
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices required')
    output=Path('/kaggle/working/continuous_focus');output.mkdir(exist_ok=False)
    print(json.dumps(dict(event='INPUTS_VERIFIED',protocol_sha256=config['pins']['protocol.json'])),flush=True)
    print(json.dumps(dict(event='CUDA_READY',gpus=[torch.cuda.get_device_name(i)for i in(0,1)])),flush=True)
    options=dict(curriculum='continuous_v1',validation_selection='worst_family',continuous_focus='uniform',
        geometry='native_anchors_candidate_v1',coordinate_mode='physical_live_cells_v2',action_delay_ms=225,max_ms=310000)
    def reference(model,index):
        blob=torch.load(inputs/model['file'],map_location='cpu',weights_only=False)
        net,_,_=initialize_attack_policy(blob,device=f'cuda:{index%2}',num_scalars=309,observation_contract=CONTRACT)
        report=evaluate(net,f'cuda:{index%2}',episodes=36,seed=900100,env_options=options,task_mode='joint')
        result=dict(id=model['id'],sha256=model['sha256'],validation=report)
        (output/(model['id']+'_reference.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(event='FOCUS_REFERENCE_COMPLETE',model=model['id'],score=report['selection_score'])),flush=True)
        return result
    references=[]
    # At most one worker per device, including the second reference on GPU0.
    def ref_group(index):return[reference(m,index)for m in config['models'][index::2]]
    with ThreadPoolExecutor(2)as pool:
        futures=[pool.submit(ref_group,i)for i in(0,1)]
        for f in futures:references.extend(f.result())
    initial=select_initial(references)
    (output/'initial_selection.json').write_text(json.dumps(initial,indent=2)+'\n')
    initial_model=next(m for m in config['models']if m['id']==initial['selected'])
    protocol=json.loads((inputs/'protocol.json').read_text())
    if 'air_examples.npz'in config['pins']:
        air_data=project/'air_warmstart_data';air_data.mkdir()
        shutil.copyfile(inputs/'air_examples.npz',air_data/'examples.npz')
        shutil.copyfile(inputs/'air_manifest.json',air_data/'manifest.json')
    def run(trial,device,seed,full=False):
        run_id=('full_'if full else'pilot_')+trial['id']+'_s'+str(seed)
        directory=output/run_id;steps=4096000 if full else 102400
        run_initial=initial_model
        if trial.get('warmstart_updates',0):
            warm=output/(run_id+'_warmstart')
            subprocess.run([sys.executable,'-m','scripts.attack.train_air_position_warmstart',
                '--data',str(air_data),'--init',str(inputs/initial_model['file']),'--output',str(warm),
                '--device',device,'--updates',str(trial['warmstart_updates']),
                '--kl-weight',str(trial['warmstart_kl_weight']),'--seed',str(seed)],cwd=project,check=True)
            run_initial=dict(id=run_id+'_warmstart',file=str(warm/'policy.pt'),sha256=sha(warm/'policy.pt'))
            print(json.dumps(dict(event='WARMSTART_COMPLETE',trial_id=run_id,updates=trial['warmstart_updates'])),flush=True)
        profile=output/(run_id+'_profile.json')
        profile.write_text(json.dumps(dict(overflow_penalty=trial['overflow_penalty'],
            missed_fireball_penalty=trial.get('missed_fireball_penalty',0.),
            board_potential_weight=trial.get('board_potential_weight',0.),shaping_gamma=trial.get('gamma',.999),
            action_cooldown_ms=2100,decision_interval_ms=250)))
        args=[sys.executable,'-c',"import torch,runpy;torch.set_num_threads(1);runpy.run_module('sim.train_attack_ppo',run_name='__main__')",
            '--init',str(inputs/run_initial['file']),'--output',str(directory),'--device',device,
            '--task-mode','joint','--curriculum','continuous_v1','--geometry','native_anchors_candidate_v1',
            '--coordinate-mode','physical_live_cells_v2','--continuous-profile',str(profile),
            '--continuous-focus',trial['focus'],'--validation-selection','worst_family','--validation-seed','900100',
            '--eval-episodes','36','--entropy-coef',str(trial['entropy_coef']),
            '--history-frames',str(trial.get('history_frames',1)),
            '--card-balance-weight',str(trial['card_balance_weight']),'--card-balance-target','units',
            '--policy-temperature',str(trial.get('policy_temperature',1.)),
            '--action-delay-ms','225','--max-ms','310000','--seed',str(seed),'--envs','8','--rollout','128',
            '--updates',str(steps//1024),'--target-environment-steps',str(steps),'--epochs','3','--minibatch-size','256',
            '--lr','0.00005','--gamma',str(trial.get('gamma',.999)),
            '--gae-lambda',str(trial.get('gae_lambda',.95)),'--value-warmup-steps','20480','--target-kl','.02',
            '--checkpoint-every','250'if full else'50','--eval-every','250'if full else'50',
            '--wallclock-seconds','34000'if full else'7200']
        process=subprocess.Popen(args,cwd=project,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        print(json.dumps(dict(event='TRIAL_START',trial_id=run_id,target_steps=steps,device=device)),flush=True)
        with(output/(run_id+'.log')).open('x')as log:
            for line in process.stdout:
                log.write(line);log.flush()
                try:r=json.loads(line)
                except ValueError:print(line,end='',flush=True);continue
                if isinstance(r,dict)and'update'in r and'actual_cuda'in r:
                    print(json.dumps(dict(event='TRIAL_PROGRESS',trial_id=run_id,step=r['step'],update=r['update'],
                        target_steps=steps,percent=r['percent'],actual_cuda=r['actual_cuda'])),flush=True)
        if process.wait():raise RuntimeError('Focus training failed: '+run_id)
        boundary=json.loads((directory/'training_result.json').read_text())
        if not boundary.get('target_met')or boundary.get('step')!=steps:raise ValueError('Incomplete focus budget')
        rows=[json.loads(line)for line in(directory/'metrics.jsonl').read_text().splitlines()]
        validations=[r for r in rows if'validation'in r]
        eligible=[r for r in validations if r['validation']['worst_archetype_points']>=.1]
        chosen=max(eligible or validations,key=lambda r:r['validation']['selection_score'])
        # All evaluation boundaries also save a regular checkpoint, even if a
        # lower-scoring eligible boundary did not beat an earlier specialist.
        checkpoint=directory/f"update_{chosen['update']:06d}.pt"
        if not checkpoint.exists():raise ValueError('Selected boundary was not saved')
        result=dict(id=run_id,trial=trial,seed=seed,step=steps,target_met=True,selected_update=chosen['update'],
            selected_with_family_guard=bool(eligible),checkpoint=str(checkpoint.relative_to(output)),
            checkpoint_sha256=sha(checkpoint),initial_model=run_initial,validation=chosen['validation'],quality_accepted=False)
        (output/(run_id+'_summary.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(event='FOCUS_TRIAL_COMPLETE',trial_id=run_id,score=chosen['validation']['selection_score'],family_guard=bool(eligible))),flush=True)
        return result
    def pilots(index):
        return[run(trial,f'cuda:{index}',seed)for trial in protocol['cases'][index::2]for seed in protocol.get('pilot_seeds',(62,63))]
    runs=[]
    with ThreadPoolExecutor(2)as pool:
        futures=[pool.submit(pilots,i)for i in(0,1)]
        for f in futures:runs.extend(f.result())
    selection=select_profiles(runs,seeds=protocol.get('pilot_seeds',(62,63)))
    (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    full=[]
    if selection['gate_met']and config.get('allow_full_training',True):
        with ThreadPoolExecutor(2)as pool:
            futures=[pool.submit(run,r['trial'],f'cuda:{i}',64,True)for i,r in enumerate(selection['selected'])]
            full=[f.result()for f in futures]
    result=dict(schema='continuous-focus-cohort-result-v1',status='complete'if full else
        'pilot_complete'if not config.get('allow_full_training',True)else'pilot_complete_no_eligible_pair',
        pilot_steps=819200,full_steps=4096000*len(full),pilot_budgets_met=True,full_gate_met=selection['gate_met'],
        full_runs=full,initial_selection=initial,protocol_sha256=config['pins']['protocol.json'],
        quality_accepted=False,live_checkpoint_changed=False,test_used_for_selection=False)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


def freeze_template(output,extra_files=()):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    frozen=Path('tmp/kaggle/fireball_balanced_20261025/dataset');manifest=json.loads((frozen/'manifest.json').read_text())
    names=('sim/match.py','sim/train_attack_ppo.py','sim/continuous_push.py','sim/card_balance.py','sim/env.py',
        'sim/kite_geometry.py','sim/projectile_calibration.py','sim/entities.py','sim/gamedata.py','sim/public_gamedata.py',
        'sim/spells.py','sim/continuous_sampling.py','sim/continuous_validation.py','scripts/attack/select_continuous_focus.py',
        'sim/public_frame_history.py','sim/continuous_history.py','sim/board_potential.py','sim/pathfind.py')
    names=names+tuple(extra_files)
    if len(set(names))!=len(names):raise ValueError('Duplicate frozen extension')
    with tarfile.open(output/'extensions.tar.gz','w:gz')as stream:
        for name in names:stream.add(name,arcname=name,recursive=False)
    shutil.copyfile('data/ppo_tuning/continuous_focus/protocol.json',output/'protocol.json')
    code='import hashlib,json,shutil,tarfile\nfrom pathlib import Path,PurePosixPath\n'+inspect.getsource(sha)+'\n'+inspect.getsource(prepare_source)+'\n'+inspect.getsource(execute)
    compile(code,'focus_template.py','exec');(output/'bootstrap.txt').write_text(code)
    binder='import argparse,hashlib,json,shutil\nfrom pathlib import Path\n'+inspect.getsource(sha)+'\n'+inspect.getsource(bind)+'''\n
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--training-report',required=True);p.add_argument('--output',required=True);p.add_argument('--owner',default='vertel0')
    a=p.parse_args();print(json.dumps(bind(a.template,a.training_report,a.output,a.owner)))
'''
    compile(binder,'bind.py','exec');(output/'bind.py').write_text(binder)
    template=dict(schema='frozen-continuous-focus-template-v1',extensions={n:sha(n)for n in names},
        replaced={n:manifest['files'][n]for n in names if n in manifest['files']},
        source_manifest_sha256=sha(frozen/'manifest.json'),
        files={n:sha(output/n)for n in('extensions.tar.gz','protocol.json','bootstrap.txt','bind.py')})
    (output/'template_manifest.json').write_text(json.dumps(template,indent=2)+'\n')
    return template


def bind(template,training_report,output,owner='vertel0'):
    template=Path(template);manifest=json.loads((template/'template_manifest.json').read_text())
    if any(sha(template/n)!=h for n,h in manifest['files'].items()):raise ValueError('Frozen template changed')
    report=json.loads(Path(training_report).read_text())
    if(report.get('schema')!='verified-continuous-ppo-training-v1'or not report.get('all_training_budgets_verified')or
            report.get('kernel')!='andreiperelaiko/hastycr-continuous-slots-20261026'):raise ValueError('Corrected-slot training proof required')
    if len(report.get('models',[]))!=2:raise ValueError('Two corrected full-model references required')
    import torch
    for model in report['models']:
        original=Path(model['checkpoint'])
        if sha(original)!=model['checkpoint_sha256']:raise ValueError('Corrected weight changed')
        blob=torch.load(original,map_location='cpu',weights_only=False)
        if(blob.get('config',{}).get('observation_contract')!='joint_public_presence_continuous_slots_v2'or
                blob['state_dict']['scalar.0.weight'].shape[1]!=309):raise ValueError('Corrected slot input contract required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);dataset=output/'dataset';dataset.mkdir();job=output/'job';job.mkdir()
    models=[]
    for model in report['models']:
        original=Path(model['checkpoint']);name='corrected_'+model['id'];digest=model['checkpoint_sha256']
        if sha(original)!=digest:raise ValueError('Corrected weight changed')
        shutil.copyfile(original,dataset/(name+'.pt'));models.append(dict(id=name,file=name+'.pt',sha256=digest))
    parent=Path('checkpoints/ppo_joint_economy_soft_20261023/policy.pt')
    if sha(parent)!='9d6ac7d1d41ff00df6a9c74354fa8447793e0be8d69f723b27ff43b23c4c73e8':raise ValueError('Original parent changed')
    shutil.copyfile(parent,dataset/'parent.pt');models.append(dict(id='parent',file='parent.pt',sha256=sha(parent)))
    for name in('extensions.tar.gz','protocol.json'):shutil.copyfile(template/name,dataset/name)
    frozen=Path('tmp/kaggle/fireball_balanced_20261025/dataset')
    if sha(frozen/'manifest.json')!=manifest['source_manifest_sha256']:raise ValueError('Original frozen source changed')
    for name in('manifest.json','source.tar.gz'):shutil.copyfile(frozen/name,dataset/name)
    slug='hastycr-continuous-focus-inputs-20261026'
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug,title=slug,licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    config=dict(owner=owner,slug=slug,source_slug=slug,models=models,
        source_manifest_sha256=manifest['source_manifest_sha256'],extensions=manifest['extensions'],replaced=manifest['replaced'],
        pins={n:sha(dataset/n)for n in('manifest.json','extensions.tar.gz','protocol.json',*[m['file']for m in models])})
    code=(template/'bootstrap.txt').read_text()+'\nexecute('+repr(config)+')\n'
    compile(code,'train.py','exec');(job/'train.py').write_text(code)
    kernel=owner+'/hastycr-continuous-focus-20261026'
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title='hastycr continuous focus 20261026',code_file='train.py',
        language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,
        dataset_sources=[owner+'/'+slug],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    (output/'preparation.json').write_text(json.dumps(dict(kernel=kernel,config=config,bootstrap_sha256=sha(job/'train.py'),
        template_manifest_sha256=sha(template/'template_manifest.json'),submitted=False),indent=2)+'\n')
    return dict(kernel=kernel,submitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-template');p.add_argument('--template');p.add_argument('--training-report');p.add_argument('--output');p.add_argument('--owner',default='vertel0')
    a=p.parse_args()
    if a.freeze_template:print(json.dumps(freeze_template(a.freeze_template)))
    else:print(json.dumps(bind(a.template,a.training_report,a.output,a.owner)))
