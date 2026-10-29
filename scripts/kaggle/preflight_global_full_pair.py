"""Local CPU smoke/resume test using commands captured from frozen full run().

The controller prefix is a dry run: it never pretends to complete full steps.
Actual CPU training uses16/32 steps and at most5 supervised updates. No API.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace


def set_flag(args, flag, value):
    result=list(args)
    index=result.index(flag)
    result[index+1]=str(value)
    return result


def smoke_args(args, output, steps, resume=None):
    result=list(args)
    for flag,value in {'--device':'cpu','--output':output,'--envs':1,'--rollout':16,
        '--updates':steps//16,'--target-environment-steps':steps,'--minibatch-size':16,
        '--eval-every':0,'--eval-episodes':6,'--checkpoint-every':1,
        '--value-warmup-steps':0,'--wallclock-seconds':600}.items():
        result=set_flag(result,flag,value)
    if resume is not None:
        index=result.index('--init');result[index:index+2]=['--resume',str(resume)]
    return result


def equal_state(a,b):
    import numpy as np
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor)and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray)and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict)and a.keys()==b.keys()and all(equal_state(a[k],b[k])for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b)and len(a)==len(b)and all(equal_state(x,y)for x,y in zip(a,b))
    return a==b


def child(spec_path):
    import torch
    torch.set_num_threads(1)
    spec=json.loads(Path(spec_path).read_text())
    root=Path(spec['root']);inputs=Path(spec['inputs']);project=Path.cwd()
    trial=spec['trial'];config=spec['config']
    from sim.train_attack_ppo import evaluate,initialize_attack_policy
    from sim.continuous_push import CONTRACT
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    output=root/'controller_output';output.mkdir()
    air_data=project/'preflight_demonstrations'
    if 'air_examples.npz'in config['pins']:
        air_data.mkdir();shutil.copyfile(inputs/'air_examples.npz',air_data/'examples.npz')
        shutil.copyfile(inputs/'air_manifest.json',air_data/'manifest.json')
    initial=next(m for m in config['models']if m['id']==spec['initial_selection']['selected'])
    warm_updates=min(5,trial.get('warmstart_updates',0))
    if warm_updates:
        from scripts.attack.train_air_position_warmstart import train
        train(air_data,inputs/initial['file'],output/f"full_{trial['id']}_s64_warmstart",
              'cpu',warm_updates,trial['warmstart_kl_weight'],64)
    tree=ast.parse(Path(spec['bootstrap']).read_text())
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute')
    run=next(n for n in execute.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    captured=[];warm_commands=[]
    class Captured(Exception):pass
    def popen(args,**kwargs):captured.append(list(args));raise Captured()
    def warm_dry_run(args,**kwargs):
        if '--updates'not in args or int(args[args.index('--updates')+1])!=trial['warmstart_updates']:
            raise ValueError('Unexpected original warm command')
        warm_commands.append(list(args));return SimpleNamespace(returncode=0)
    namespace=dict(Path=Path,json=json,sys=sys,sha=sha,output=output,inputs=inputs,
        project=project,initial_model=initial,air_data=air_data,config=config,
        print=lambda *a,**kw:None,
        subprocess=SimpleNamespace(run=warm_dry_run,Popen=popen,PIPE=subprocess.PIPE,STDOUT=subprocess.STDOUT))
    exec(compile(ast.Module(body=[run],type_ignores=[]),'exact_frozen_full_run.py','exec'),namespace)
    try:namespace['run'](trial,'cpu',64,True)
    except Captured:pass
    if len(captured)!=1:raise ValueError('Exactly one original PPO command required')
    args=captured[0]
    if args[args.index('--target-environment-steps')+1]!='4096000' or args[args.index('--updates')+1]!='4000':
        raise ValueError('Captured controller is not the full-budget branch')
    # These are real shortened optimizer runs, with the actual frozen data.
    first=smoke_args(args,root/'first',16);whole=smoke_args(args,root/'whole',32)
    resumed=smoke_args(args,root/'resumed',32,root/'first/update_000001.pt')
    for command in(first,whole,resumed):subprocess.run(command,cwd=project,check=True)
    a=torch.load(root/'whole/update_000002.pt',map_location='cpu',weights_only=False)
    b=torch.load(root/'resumed/update_000002.pt',map_location='cpu',weights_only=False)
    if not all(equal_state(a[key],b[key])for key in('state_dict','optimiser','rng')):
        raise ValueError('Exact resume changed weights, optimizer or random state')
    net,_,_=initialize_attack_policy(a,device='cpu',num_scalars=309,observation_contract=CONTRACT)
    profile=json.loads((output/f"full_{trial['id']}_s64_profile.json").read_text())
    validation=evaluate(net,'cpu',episodes=6,seed=900100,task_mode='joint',env_options=dict(
        curriculum='continuous_v1',validation_selection='worst_family',continuous_profile=profile,
        geometry='native_anchors_candidate_v1',coordinate_mode='physical_live_cells_v2',
        action_delay_ms=trial.get('validation_action_delay_ms',trial.get('action_delay_ms',225)),max_ms=310000))
    if len(validation['records'])!=12:raise ValueError('Full-game preflight coverage missing')
    result=dict(frozen_run_ast_sha256=hashlib.sha256(ast.dump(run).encode()).hexdigest(),
        original_full_command=args,original_warm_commands=warm_commands,actual_supervised_updates=warm_updates,
        actual_smoke_target_steps=32,exact_resume_16_to_32=True,optimizer_and_rng_equal=True,
        validation_games=12,actual_cuda=False,quality_accepted=False,
        technical_fixture=spec.get('technical_fixture',False))
    (root/'child_preflight.json').write_text(json.dumps(result,indent=2)+'\n')


def unpack(inputs,config,project):
    from scripts.kaggle.run_five_round_robin import prepare_source,sha
    for name,digest in config['pins'].items():
        if sha(inputs/name)!=digest:raise ValueError('Preflight input changed')
    prepare_source(inputs,project,config['source_manifest_sha256'])
    for name,digest in config['replaced'].items():
        if sha(project/name)!=digest:raise ValueError('Unexpected original source')
    with tarfile.open(inputs/'extensions.tar.gz')as archive:
        members=archive.getmembers()
        if {m.name for m in members}!=set(config['extensions'])or any(not m.isfile()or Path(m.name).is_absolute()or '..'in Path(m.name).parts for m in members):
            raise ValueError('Unsafe preflight extension archive')
        archive.extractall(project,filter='data')
    for name,digest in config['extensions'].items():
        if sha(project/name)!=digest:raise ValueError('Preflight extension changed')


def preflight(prepared,output):
    import os
    from scripts.attack.select_pilot_pool import select
    from scripts.kaggle.run_five_round_robin import sha
    prepared,output=Path(prepared).absolute(),Path(output).absolute()
    prep=json.loads((prepared/'preparation.json').read_text())
    ranking=json.loads((prepared/'dataset/global_selection.json').read_text())
    if select([r['prepared']for r in ranking['studies']],limit=4)!=ranking or not ranking['gate_met']:
        raise ValueError('Actual positive global selection required')
    if sha(prepared/'job/train.py')!=prep['bootstrap_sha256']:raise ValueError('Full driver changed')
    for name,digest in prep['config']['pins'].items():
        if sha(prepared/'dataset'/name)!=digest:raise ValueError('Full bundle input changed')
    pair_index=prep['pair_index']
    expected=ranking['selected'][2*pair_index:2*pair_index+2]
    if pair_index not in (0,1)or len(expected)!=2 or [c['selected']for c in prep['children']]!=expected:
        raise ValueError('Prepared candidates differ from verified global pair')
    for index,entry in enumerate(prep['children']):
        inputs=prepared/'dataset'/f'bundle_{index}'
        tree=ast.parse((inputs/'train.py').read_text())
        if (entry['index']!=index or sha(inputs/'train.py')!=entry['bootstrap_sha256']or
                ast.literal_eval(tree.body[-1].value.args[0])!=entry['config']):
            raise ValueError('Child controller configuration changed')
    from scripts.kaggle.prepare_global_full_pair import transport_inputs
    upload=prepared/prep['upload_directory']
    for name,digest in prep['upload_pins'].items():
        if sha(upload/name)!=digest:raise ValueError('Full transport changed before preflight')
    output.mkdir(parents=True,exist_ok=False);results=[]
    decoded=transport_inputs(upload,output/'decoded_inputs',prep['config'])
    for index,entry in enumerate(prep['children']):
        root=output/f'child_{index}';root.mkdir()
        inputs=decoded/f'bundle_{index}';project=root/'project'
        unpack(inputs,entry['config'],project)
        spec=dict(root=str(root),inputs=str(inputs),trial=entry['selected']['trial'],config=entry['config'],
            initial_selection=json.loads((inputs/'initial_selection.json').read_text()),bootstrap=str(inputs/'train.py'))
        path=root/'spec.json';path.write_text(json.dumps(spec,indent=2)+'\n')
        child_script=Path(__file__).absolute()
        variant=json.loads((inputs/'protocol.json').read_text()).get('study_variant')
        if variant=='bank_counterfactual_v1':
            child_script=child_script.with_name('preflight_bank_teacher.py')
        elif variant=='mana_decision_v1':
            child_script=child_script.with_name('preflight_mana_campaign.py')
        with(root/'run.log').open('x')as log:
            subprocess.run([sys.executable,str(child_script),'--child',str(path)],cwd=project,
                env=dict(os.environ,PYTHONPATH=str(project)),stdout=log,stderr=subprocess.STDOUT,check=True)
        results.append(json.loads((root/'child_preflight.json').read_text()))
    result=dict(frozen_sources_verified=True,exact_resume_16_to_32=True,children=results,
        validation_games=24,actual_cuda=False,quality_accepted=False,
        preparation_sha256=sha(prepared/'preparation.json'),bootstrap_sha256=prep['bootstrap_sha256'],
        scope='Frozen full commands; explicitly reduced CPU budgets; CUDA routing checked statically, enforced at launch')
    with(prepared/'preflight.json').open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result


def enable_submission(prepared):
    from scripts.kaggle.run_five_round_robin import sha
    from scripts.attack.select_pilot_pool import select
    prepared=Path(prepared)
    prep=json.loads((prepared/'preparation.json').read_text())
    report=json.loads((prepared/'preflight.json').read_text())
    if (not report.get('frozen_sources_verified')or not report.get('exact_resume_16_to_32')or
            report.get('preparation_sha256')!=sha(prepared/'preparation.json')or
            report.get('bootstrap_sha256')!=sha(prepared/'job/train.py')or
            len(report.get('children',[]))!=2 or report.get('validation_games')!=24 or
            any(c.get('technical_fixture')or not c.get('optimizer_and_rng_equal')for c in report['children'])):
        raise ValueError('Verified selected-pair preflight required; technical fixtures cannot submit')
    ranking=json.loads((prepared/'dataset/global_selection.json').read_text())
    if select([r['prepared']for r in ranking['studies']],limit=4)!=ranking or not ranking['gate_met']:
        raise ValueError('Global selection changed before submission')
    for name,digest in prep['config']['pins'].items():
        if sha(prepared/'dataset'/name)!=digest:raise ValueError('Full input changed after preflight')
    metadata=dict(id=prep['kernel'],title=prep['kernel'].split('/')[-1],code_file='train.py',
        language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,
        dataset_sources=[prep['config']['owner']+'/'+prep['config']['slug']],competition_sources=[],kernel_sources=[])
    with(prepared/'job/kernel-metadata.json').open('x')as stream:
        json.dump(metadata,stream,indent=2);stream.write('\n')
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--child');parser.add_argument('--prepared');parser.add_argument('--output');parser.add_argument('--enable-submission',action='store_true')
    args=parser.parse_args()
    if args.child:child(args.child)
    elif args.prepared and args.output:
        preflight(args.prepared,args.output)
        if args.enable_submission:enable_submission(args.prepared)
    else:parser.error('Provide --child or --prepared with --output')
