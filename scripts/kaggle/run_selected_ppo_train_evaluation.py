"""Descriptive train-split evaluation of fixed final weights; never trains/selects."""
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

from scripts.kaggle.run_strategy_evaluation import prepare_source,sha
from scripts.kaggle.run_five_round_robin import skill_policy,skill_environment
from scripts.kaggle.run_selected_ppo_evaluation import verify_selection

LAUNCH_CONFIG=None
OUTPUT=Path('/kaggle/working/train_evaluation')


def mount(reference):
    owner,slug=reference.split('/')
    paths=[p for p in (Path('/kaggle/input')/slug,Path('/kaggle/input/datasets')/owner/slug)if p.is_dir()]
    if len(paths)!=1:raise ValueError('Expected one exact train-evaluation input mount')
    return paths[0]


def schedule(scenarios):
    if not scenarios or any(s.split!='train'for s in scenarios):raise ValueError('Train-only catalog required')
    names=sorted(s.scenario_id for s in scenarios)
    if len(names)!=len(set(names)):raise ValueError('Duplicate train scenario')
    return [dict(scenario_id=name,seed=seed)for name in names for seed in (42,43,44)]


def worker(config,index):
    if index not in (0,1):raise ValueError('Expected one of two evaluation workers')
    project=OUTPUT/'project';sys.path[:0]=[str(project),str(project/'scripts')]
    import torch
    from sim.eval_strategy import load_selected
    from sim.eval_defense_duel import load_network
    from sim.eval_defense import evaluate_episode
    from sim.strategy_env import StrategyEnv
    from sim.evaluation_terminal_damage import observe_terminal_damage
    torch.set_num_threads(2);device='cuda:'+str(index)
    manifest=json.loads((OUTPUT/'evaluation_manifest.json').read_text());weights=mount(config['selected_dataset'])
    selection=json.loads((weights/'selection.json').read_text());mode=verify_selection(selection,config,manifest)
    protocol=json.loads((project/'data/strategy/evaluation_protocol.json').read_text())
    joint=mode=='joint';policy=weights/'policy.pt'
    network=load_selected(policy,device,protocol['environment'],original_training_files=manifest.get('original_training_files'))if joint else load_network(policy,device,'physical_live')
    network=skill_policy(network,joint,torch).eval().requires_grad_(False)
    env=skill_environment(StrategyEnv,joint,split='train',**protocol['environment'])
    contexts=schedule(env.scenarios)[index::2];records=[]
    with (OUTPUT/('worker_'+str(index)+'.jsonl')).open('x')as stream:
        for ordinal,context in enumerate(contexts):
            with observe_terminal_damage(env)as observer:
                record=observer.enrich(evaluate_episode(env,network,device,**context))
            if record.get('split')!='train' or record.get('strategy_scenario_id')!=context['scenario_id'] or record.get('seed')!=context['seed']:
                raise ValueError('Train evaluator returned an off-protocol episode')
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();records.append(record)
            if (ordinal+1)%64==0 or ordinal+1==len(contexts):
                print(json.dumps(dict(event='TRAIN_EVALUATION_PROGRESS',worker=index,done=ordinal+1,total=len(contexts))),flush=True)


def execute(config):
    OUTPUT.mkdir(parents=True,exist_ok=False)
    source=mount(config['evaluation_dataset']);weights=mount(config['selected_dataset'])
    if sha(weights/'selection.json')!=config['selection_sha256']or sha(weights/'policy.pt')!=config['policy_sha256']:
        raise ValueError('Fixed final policy SHA differs')
    project=OUTPUT/'project';manifest=prepare_source(source,project,config['manifest_sha256'])
    (OUTPUT/'evaluation_manifest.json').write_text(json.dumps(manifest)+'\n')
    sys.path[:0]=[str(project),str(project/'scripts')]
    import torch
    from sim.strategy_env import StrategyEnv
    from sim.eval_defense import aggregate
    from sim.train_attack_ppo import summarize_attack
    from sim.eval_strategy import summarize_terminal_attribution
    if not torch.cuda.is_available()or torch.cuda.device_count()!=2:raise RuntimeError('Two CUDA GPUs required')
    selection=json.loads((weights/'selection.json').read_text());mode=verify_selection(selection,config,manifest)
    protocol=json.loads((project/'data/strategy/evaluation_protocol.json').read_text())
    env=skill_environment(StrategyEnv,mode=='joint',split='train',**protocol['environment']);contexts=schedule(env.scenarios)
    if len(contexts)!=4452:raise ValueError('Expected all1484train scenarios on three fixed seeds')
    context_sha=hashlib.sha256(json.dumps(contexts,sort_keys=True).encode()).hexdigest()
    report=dict(schema='selected-ppo-train-evaluation-v1',status='evaluating',split='train',training_steps=2048000,
        selected_sha256=config['policy_sha256'],selection_sha256=config['selection_sha256'],mode=mode,profile=selection['profile'],
        source_sha256=manifest['archive_sha256'],context_schedule_sha256=context_sha,seeds=[42,43,44],expected_episodes=4452,
        gpu_names=[torch.cuda.get_device_name(i)for i in range(2)],training_performed=False,test_used_for_selection=False,quality_accepted=False)
    def save():(OUTPUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save();print(json.dumps(report),flush=True)
    def run(index):
        with (OUTPUT/('worker_'+str(index)+'.log')).open('x')as stream:
            process=subprocess.Popen([sys.executable,__file__,'--worker',str(index)],env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
            if process.wait():raise RuntimeError('Train evaluation worker failed: '+str(index))
    try:
        with ThreadPoolExecutor(2)as pool:
            futures=[pool.submit(run,i)for i in range(2)]
            for future in futures:future.result()
        records=[json.loads(line)for i in range(2)for line in (OUTPUT/('worker_'+str(i)+'.jsonl')).read_text().splitlines()]
        actual=[(r['strategy_scenario_id'],r['seed'])for r in records];expected={(r['scenario_id'],r['seed'])for r in contexts}
        if len(actual)!=4452 or set(actual)!=expected:raise ValueError('Incomplete/duplicate train evaluation schedule')
        records.sort(key=lambda r:(r['strategy_scenario_id'],r['seed']))
        defense=[r for r in records if r['task']=='defense'];attack=[r for r in records if r['task']=='attack']
        result=dict(schema='selected-ppo-train-skill-result-v1',split='train',episodes=4452,records=records,
            checkpoint_sha256=config['policy_sha256'],source_sha256=manifest['archive_sha256'],context_schedule_sha256=context_sha,
            input_mode='joint_309_cells_v2'if mode=='joint'else'physical_live_306_timer_task_ignored',terminal_attribution_enabled=True,
            defense=aggregate(defense),attack=dict(overall=summarize_attack(attack),terminal_attribution_v2=summarize_terminal_attribution(attack)),
            training_performed=False,checkpoint_selection_performed=False,quality_accepted=False)
        (OUTPUT/'train.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        report.update(status='completed',episodes=4452,train_sha256=sha(OUTPUT/'train.json'))
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise
    save();print(json.dumps(report),flush=True);return report


if __name__=='__main__':
    if LAUNCH_CONFIG is None:raise RuntimeError('Pinned final policy required')
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(LAUNCH_CONFIG,int(sys.argv[2]))
    else:execute(LAUNCH_CONFIG)
