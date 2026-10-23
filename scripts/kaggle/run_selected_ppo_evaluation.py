"""Heldout common skills and fifteen opponents for one fixed-budget full PPO."""
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

LAUNCH_CONFIG=None


def verify_selection(selection,config,manifest):
    if (selection.get('schema')!='selected-full-ppo-evaluation-input-v1' or
        selection.get('test_used_for_selection') is not False or selection.get('training_steps')!=2048000 or
        selection.get('training_update')!=2000 or selection.get('selected_sha256')!=config['policy_sha256']):
        raise ValueError('Completed predetermined final PPO required')
    mode=selection.get('mode')
    if mode not in ('defense','joint') or selection.get('num_scalars')!=(306 if mode=='defense' else 309):
        raise ValueError('Unexpected final policy input contract')
    if selection.get('source_sha256')!=manifest.get('training_source_archive_sha256'):
        raise ValueError('Evaluation source does not descend from selected training source')
    if selection.get('policy_optimizer_finite') is not True:raise ValueError('Final PPO audit missing')
    return mode


def duel_command(project,policy,output,mode):
    return [sys.executable,'-m','sim.eval_defense_duel','--candidate',str(policy),
            '--candidate-mode','joint_enemy_presence_v1' if mode=='joint' else 'physical_live',
            '--baseline-manifest',str(project/'data/defense/baselines.json'),'--matches','20',
            '--seed','710001','--delay-ms','150','--max-ms','400000','--device','cuda:1','--output',str(output/'duels.json')]


def execute(config,input_root=Path('/kaggle/input'),output=Path('/kaggle/working/strategy_evaluation')):
    def mount(reference):
        owner,slug=reference.split('/')
        found=[p for p in (input_root/slug,input_root/'datasets'/owner/slug)if p.is_dir()]
        if len(found)!=1:raise ValueError('Expected one exact selected-PPO evaluation mount')
        return found[0]
    source=mount(config['evaluation_dataset']);weights=mount(config['selected_dataset']);policy=weights/'policy.pt'
    if sha(weights/'selection.json')!=config['selection_sha256'] or sha(policy)!=config['policy_sha256']:
        raise ValueError('Pinned final PPO input SHA mismatch')
    selection=json.loads((weights/'selection.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='selected-ppo-final-evaluation-v1',status='verifying',parts={},quality_accepted=False,
                test_used_for_selection=False,selected_sha256=config['policy_sha256'],selection_sha256=config['selection_sha256'])
    def save():(output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        project=output/'project';manifest=prepare_source(source,project,config['manifest_sha256'])
        mode=verify_selection(selection,config,manifest)
        protocol_path=project/'data/strategy/evaluation_protocol.json'
        if sha(protocol_path)!=config['protocol_sha256']:raise ValueError('Heldout protocol SHA mismatch')
        protocol=json.loads(protocol_path.read_text());sys.path[:0]=[str(project),str(project/'scripts')]
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA GPUs required')
        torch.set_num_threads(2)
        from sim.eval_strategy import load_selected,evaluate
        from sim.eval_defense_duel import load_network
        from sim.strategy_env import StrategyEnv
        from sim.defense_protocol import EvaluationProtocol,acceptance_gate
        joint=mode=='joint'
        network=(load_selected(policy,'cuda:0',protocol['environment']) if joint else load_network(policy,'cuda:0','physical_live'))
        network=skill_policy(network,joint,torch).eval().requires_grad_(False)
        def factory(**options):return skill_environment(StrategyEnv,joint,**options)
        report.update(status='evaluating',mode=mode,profile=selection['profile'],training_steps=2048000,
                      source_sha256=manifest['archive_sha256'],training_source_sha256=selection['source_sha256'],
                      input_mode='joint_309_cells_v2' if joint else 'physical_live_306_timer_task_ignored',
                      gpu_names=[torch.cuda.get_device_name(i)for i in range(2)])
        save();print(json.dumps(report),flush=True)
        environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        def skills():
            for split in ('validation','test'):
                result=evaluate(network,'cuda:0',protocol,split,env_factory=factory,terminal_attribution=True)
                result.update(checkpoint_sha256=config['policy_sha256'],protocol_sha256=config['protocol_sha256'],input_mode=report['input_mode'])
                with (output/(split+'.json')).open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
                (output/(split+'.log')).write_text('Common heldout skills completed; no training or selection.\n')
                print(json.dumps(dict(event='part_completed',part=split,episodes=result['episodes'])),flush=True)
        def duels():
            with (output/'duels.log').open('x')as stream:
                subprocess.run(duel_command(project,policy,output,mode),cwd=project,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
            print(json.dumps(dict(event='part_completed',part='duels',games=300)),flush=True)
        with ThreadPoolExecutor(2)as pool:
            futures=[pool.submit(skills),pool.submit(duels)]
            for future in futures:future.result()
        test=json.loads((output/'test.json').read_text());matches=json.loads((output/'duels.json').read_text())
        if test.get('checkpoint_sha256')!=config['policy_sha256'] or matches.get('candidate_sha256')!=config['policy_sha256']:
            raise ValueError('Evaluation reports a different policy')
        frozen=EvaluationProtocol.from_dict(json.loads((project/'data/defense/evaluation_protocol.json').read_text()))
        defense=[r for r in test['records']if r['task']=='defense']
        attack=[r for r in test['records']if r['task']=='attack']
        expected={(sid,seed)for sid in frozen.test_scenario_ids for seed in frozen.defense_seeds}
        actual=[(r['scenario_id'],r['seed'])for r in defense]
        if len(actual)!=708 or len(actual)!=len(expected) or set(actual)!=expected or len(attack)!=96:raise ValueError('Incomplete common804-case test')
        pairs=matches['head_to_head'];actual_pairs=[(r['baseline_id'],r['seed'],r['candidate_side'])for r in pairs]
        expected_pairs={(b,s,seat)for b in frozen.baseline_ids for s in frozen.paired_seeds for seat in (1,-1)}
        if len(actual_pairs)!=300 or len(actual_pairs)!=len(expected_pairs) or set(actual_pairs)!=expected_pairs:raise ValueError('Incomplete historical opponent schedule')
        coverage=test['defense']['coverage']
        complete=all(coverage.get(k)is True for k in ('metrics_complete','tower_metrics_complete','scenario_catalog_known','all_expected_scenarios_seen')) and matches.get('coverage_complete_for_acceptance')is True
        for name in ('validation','test','duels'):report['parts'][name]=dict(file=name+'.json',sha256=sha(output/(name+'.json')))
        report.update(status='completed',test_episodes=804,defense_test_episodes=708,attack_test_episodes=96,duel_games=300,
                      defense=test['defense']['overall'],attack=test['attack']['overall'],terminal_attribution_v2=test['attack']['terminal_attribution_v2'],
                      baselines=[dict(id=b.get('id'),summary=b['summary'])for b in matches['baselines']],
                      acceptance=acceptance_gate(frozen,defense,pairs,coverage_complete=complete),
                      quality_note='Simulator/live mechanics parity remains unverified; no model promotion.')
        save();print(json.dumps(report),flush=True);return report
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise


if __name__=='__main__':
    if LAUNCH_CONFIG is None:raise RuntimeError('Prepared fixed-budget final PPO evaluation config required')
    execute(LAUNCH_CONFIG)
