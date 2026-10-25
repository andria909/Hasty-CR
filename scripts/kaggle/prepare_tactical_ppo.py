"""Small extension package for a fresh public-input tactical PPO pilot."""
import argparse
import inspect
import json
from pathlib import Path
import tarfile

from scripts.kaggle.run_five_round_robin import sha,prepare_source

BASELINE='''import json,sys,torch
from pathlib import Path
from sim.train_attack_ppo import initialize_attack_policy,evaluate
from sim.tactical_strategy import CONTRACT
torch.set_num_threads(1)
blob=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
net,_,_=initialize_attack_policy(blob,device='cuda:1',num_scalars=309,observation_contract=CONTRACT)
options=dict(curriculum='tactical_v1',geometry='native_anchors_candidate_v1',coordinate_mode='physical_live_cells_v2',initial_hand_mode='shuffle',deadline_scale=3.,visible_fraction=.5)
report=evaluate(net,'cuda:1',episodes=16,seed=900000,env_options=options,task_mode='joint')
Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\\n')
print(json.dumps(dict(event='TACTICAL_BASELINE_COMPLETE',skills=report['tactical_skills'],selection_score=report['selection_score'])),flush=True)
'''


def execute(config):
    import subprocess,sys
    root=Path('/kaggle/input')
    def mount(slug):
        paths=[p for p in (root/slug,root/'datasets'/config['owner']/slug)if p.is_dir()]
        if len(paths)!=1:raise ValueError('Ambiguous tactical input mount')
        return paths[0]
    source=mount(config['source_slug']);inputs=mount(config['slug'])
    for name,digest in config['source_pins'].items():
        if sha(source/name)!=digest:raise ValueError('Frozen source/parent pin differs')
    for name,digest in config['input_pins'].items():
        if name=='extensions.tar.gz'and not (inputs/name).exists():continue
        if sha(inputs/name)!=digest:raise ValueError('Tactical input pin differs')
    project=Path('/kaggle/working/tactical_project')
    manifest=prepare_source(source,project,config['source_pins']['manifest.json'])
    for name,digest in config['replaced'].items():
        if sha(project/name)!=digest:raise ValueError('Unexpected pre-migration source')
    if (inputs/'extensions.tar.gz').exists():
        with tarfile.open(inputs/'extensions.tar.gz')as stream:
            members=stream.getmembers()
            if len(members)!=len(config['extensions'])or {m.name for m in members}!=set(config['extensions'])or any(not m.isfile()for m in members):
                raise ValueError('Tactical extension members differ')
            for m in members:
                p=PurePosixPath(m.name)
                if p.is_absolute()or '..'in p.parts:raise ValueError('Unsafe extension')
            stream.extractall(project,filter='data')
    else:
        original=(inputs/'extensions').resolve()
        for name in config['extensions']:
            member=original/name
            if member.is_symlink()or not member.resolve().is_relative_to(original):raise ValueError('Unsafe expanded extension')
            target=project/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(member,target)
    for name,digest in config['extensions'].items():
        if sha(project/name)!=digest:raise ValueError('Migrated training source differs')
    sys.path.insert(0,str(project))
    import torch
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices required')
    output=Path('/kaggle/working/card_balance_study');output.mkdir(exist_ok=False)
    print(json.dumps(dict(event='CUDA_READY',gpu=[torch.cuda.get_device_name(i)for i in range(2)],
        init_sha256=config['source_pins']['parent.pt'],source_migration=config['extensions'])),flush=True)
    print(json.dumps(dict(event='INPUTS_VERIFIED',source_sha256=manifest['archive_sha256'],protocol_sha256=config['input_pins']['protocol.json'])),flush=True)
    baseline=subprocess.Popen([sys.executable,'-c',BASELINE,str(source/'parent.pt'),str(output/'baseline_validation.json')],cwd=project)
    command=[sys.executable,'-c',"import torch,runpy; torch.set_num_threads(1); runpy.run_module('sim.train_attack_ppo',run_name='__main__')",
        '--init',str(source/'parent.pt'),'--output',str(output/'train'),'--device','cuda:0','--task-mode','joint',
        '--curriculum','tactical_v1','--geometry','native_anchors_candidate_v1','--coordinate-mode','physical_live_cells_v2',
        '--initial-hand-mode','shuffle','--deadline-scale','3','--visible-fraction','.5',
        '--reward-profile','data/ppo_tuning/profiles/economy_soft.json',
        '--updates','100','--target-environment-steps','102400','--envs','8','--rollout','128',
        '--checkpoint-every','10','--eval-every','25','--eval-episodes','16','--wallclock-seconds','18000']
    def run_trials(trials,device,wait_baseline=False):
        if wait_baseline and baseline.wait():raise RuntimeError('Original validation failed')
        results=[]
        for trial in trials:
            args=list(command)
            args[args.index('--output')+1]=str(output/trial['id'])
            args[args.index('--device')+1]=device
            args+=['--card-balance-weight',str(trial['weight']),'--card-balance-target',trial['target']]
            process=subprocess.Popen(args,cwd=project,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                print(line,end='',flush=True)
                try:row=json.loads(line)
                except ValueError:continue
                if isinstance(row,dict)and 'update'in row and 'step'in row and 'actual_cuda'in row:
                    print(json.dumps(dict(event='TRIAL_PROGRESS',trial_id=trial['id'],
                        update=row['update'],step=row['step'],percent=row['percent'])),flush=True)
            code=process.wait();results.append(dict(trial=trial,returncode=code))
            if code:raise RuntimeError('Card balance trial failed: '+trial['id'])
        return results
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2)as pool:
        first=pool.submit(run_trials,[config['trials'][0],config['trials'][2]],'cuda:0')
        second=pool.submit(run_trials,[config['trials'][1]],'cuda:1',True)
        results=first.result()+second.result()
    baseline_code=baseline.wait()
    result=dict(schema='card-balance-ppo-study-result-v1',trials=results,baseline_returncode=baseline_code,
        status='complete'if baseline_code==0 else 'failed',steps_per_trial=102400,total_target_steps=307200,
        init_sha256=config['source_pins']['parent.pt'],protocol_sha256=config['input_pins']['protocol.json'],
        fresh_optimizer=True,test_used_for_selection=False,quality_accepted=False,live_checkpoint_changed=False)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if baseline_code:raise RuntimeError('Original tactical validation failed')


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dataset=output/'dataset';dataset.mkdir();job=output/'job';job.mkdir()
    frozen=Path('tmp/kaggle/fireball_balanced_20261025/dataset')
    manifest=json.loads((frozen/'manifest.json').read_text())
    names=('sim/train_attack_ppo.py','sim/tactical_strategy.py','sim/card_balance.py')
    extensions={name:sha(name)for name in names}
    # All engine, physics, catalogs and encoders remain the corrected source;
    # only the trainer opts in to the separately declared new environment.
    replaced={name:manifest['files'][name]for name in names if name in manifest['files']}
    if set(replaced)!={'sim/train_attack_ppo.py'}:raise ValueError('Unexpected trainer migration')
    with tarfile.open(dataset/'extensions.tar.gz','w:gz')as stream:
        for name in names:stream.add(name,arcname=name,recursive=False)
    trials=[dict(id='baseline',weight=0.,target='all_cards'),dict(id='balance_all',weight=.02,target='all_cards'),dict(id='balance_units',weight=.02,target='units')]
    protocol=dict(schema='tactical-card-balance-ppo-study-v1',target_steps_per_trial=102400,total_target_steps=307200,trials=trials,init_sha256=sha(frozen/'parent.pt'),
        fresh_optimizer=True,observation_contract='joint_public_presence_tactical_v1',scalar_count=309,
        task_rule='public_enemy_presence_v1',geometry='native_anchors_candidate_v1',fireball_speed_mt_per_sec=12000,
        tactical_cases=324,split_cases=dict(train=216,validation=54,test=54),
        family_split_rule='four_opponent_families_train_one_validation_one_test_per_goal',
        initial_hand_rule='required_pair_plus_random_spares_slots_and_cycle_for_new_introductory_tasks',
        additional_rewards=dict(useful_musketeer_unit_damage=.003,own_tower_hp_lost=-.01),
        selection_metric='tactical_min_macro_and_defense_no_damage_v1',
        full_training_gate=dict(minimum_each_tactical_validation_success=.1,maximum_defense_no_damage_drop=.05),
        test_used_for_selection=False,quality_accepted=False,live_checkpoint_changed=False)
    (dataset/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    owner='andriashadrin';slug='hastycr-card-balance-study-inputs-20261025'
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(id=owner+'/'+slug,title=slug,licenses=[dict(name='other')],isPrivate=True),indent=2)+'\n')
    config=dict(owner=owner,slug=slug,source_slug='hastycr-fireball-balanced-inputs-20261025',
        source_pins={n:sha(frozen/n)for n in ('manifest.json','parent.pt')},
        input_pins={n:sha(dataset/n)for n in ('protocol.json','extensions.tar.gz')},extensions=extensions,replaced=replaced,trials=trials)
    code='import hashlib,json,shutil,tarfile\nfrom pathlib import Path,PurePosixPath\n'+inspect.getsource(sha)+'\n'+inspect.getsource(prepare_source)+'\nBASELINE='+repr(BASELINE)+'\n'+inspect.getsource(execute)+'\nexecute('+repr(config)+')\n'
    compile(code,'train.py','exec');(job/'train.py').write_text(code)
    kernel=owner+'/hastycr-card-balance-study-20261025'
    (job/'kernel-metadata.json').write_text(json.dumps(dict(id=kernel,title='hastycr card balance study 20261025',code_file='train.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,dataset_sources=[owner+'/'+slug,owner+'/'+config['source_slug']],competition_sources=[],kernel_sources=[]),indent=2)+'\n')
    preparation=dict(schema='tactical-ppo-preparation-v1',kernel=kernel,config=config,bootstrap_sha256=sha(job/'train.py'),submitted=False)
    (output/'preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
    return dict(kernel=kernel,trials=3,steps_per_trial=102400,submitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(prepare(parser.parse_args().output)))
