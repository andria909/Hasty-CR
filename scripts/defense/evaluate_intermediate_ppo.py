"""Explicit user-requested interim PPO test and paired v25 comparison.

This is not final acceptance, resume, training, or selection from test.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from scripts.defense.evaluate_stage2 import frozen_members, relative_path, sha

LAUNCH_CONFIG=None


def execute(config,input_root=Path('/kaggle/input'),output=Path('/kaggle/working/intermediate')):
    def mount(reference):
        owner,slug=reference.split('/')
        paths=[p for p in (input_root/slug,input_root/'datasets'/owner/slug)if p.is_dir()]
        if len(paths)!=1:raise ValueError('Expected one explicit evaluation mount')
        return paths[0]
    evaluation=mount(config['evaluation_dataset']);weights=mount(config['checkpoint_dataset'])/'ppo.pt'
    if sha((evaluation/'manifest.json').read_bytes())!=config['manifest_sha256']:
        raise ValueError('Pinned evaluation source manifest mismatch')
    if sha(weights.read_bytes())!=config['checkpoint_sha256']:
        raise ValueError('Pinned intermediate checkpoint mismatch')
    members,manifest,protocol=frozen_members(evaluation,extracted_root=evaluation/'source')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='intermediate-ppo-evaluation-v1',status='verified_inputs',
        checkpoint_sha256=config['checkpoint_sha256'],quality_accepted=False,
        scope='User-requested intermediate validation-best PPO; not final training-target acceptance',
        evaluation_source_sha256=manifest['archive_sha256'],test_used_for_selection=False,
        expected_defense_episodes=len(protocol['test_scenario_ids'])*len(protocol['defense_seeds']),
        expected_v25_matches=2*len(protocol['paired_seeds']))
    def save():(output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        project=output/'project';project.mkdir()
        for name,raw in members.items():
            target=project.joinpath(*relative_path(name).parts);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        sys.path[:0]=[str(project),str(project/'scripts')]
        import torch
        if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA GPUs required')
        blob=torch.load(weights,map_location='cpu',weights_only=False)
        if blob.get('update')!=300 or blob.get('step')!=307200:
            raise ValueError('Expected exact historical PPO300 checkpoint')
        report.update(status='evaluating',update=blob['update'],training_steps=blob['step'],
            gpu_names=[torch.cuda.get_device_name(i)for i in range(2)])
        save();print(json.dumps(report),flush=True)
        from sim.defense_protocol import load_baseline_allowlist
        allowed=load_baseline_allowlist(project/'data/defense/baselines.json',project_root=project)
        baseline=allowed['v25']
        defense=[sys.executable,'-m','sim.eval_defense','--init',str(weights),'--device','cuda:0',
            '--split','test','--all-scenarios','--seeds',','.join(map(str,protocol['defense_seeds'])),
            '--max-ms','240000','--output',str(output/'defense.json')]
        duel=[sys.executable,'-m','sim.eval_defense_duel','--candidate',str(weights),
            '--candidate-mode','physical_live','--baseline',baseline['path'],
            '--baseline-mode','legacy_training','--baseline-sha256',baseline['sha256'],
            '--matches',str(report['expected_v25_matches']),'--seed',str(protocol['paired_seeds'][0]),
            '--device','cuda:1','--output',str(output/'v25.json')]
        environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        def run(item):
            name,command=item
            with (output/(name+'.log')).open('x')as log:
                subprocess.run(command,cwd=project,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
            partial=json.loads((output/(name+'.json')).read_text())
            summary=(partial['overall'] if name=='defense' else partial['baselines'][0]['summary'])
            print(json.dumps(dict(event='evaluation_finished',part=name,summary=summary,
                checkpoint_sha256=config['checkpoint_sha256'])),flush=True)
        with ThreadPoolExecutor(2)as pool:list(pool.map(run,[('defense',defense),('v25',duel)]))
        test=json.loads((output/'defense.json').read_text());duels=json.loads((output/'v25.json').read_text())
        if test['checkpoint_sha256']!=config['checkpoint_sha256']or duels['candidate_sha256']!=config['checkpoint_sha256']:
            raise ValueError('Evaluation reported a different checkpoint')
        expected={(sid,seed)for sid in protocol['test_scenario_ids']for seed in protocol['defense_seeds']}
        actual=[(r['scenario_id'],r['seed'])for r in test['records']]
        if len(actual)!=len(expected)or set(actual)!=expected:raise ValueError('Incomplete/duplicate test schedule')
        rows=duels['baselines'][0]['matches']
        expected_pairs={(seed,side)for seed in protocol['paired_seeds']for side in (1,-1)}
        if len(rows)!=len(expected_pairs)or {(r['seed'],r['candidate_side'])for r in rows}!=expected_pairs:
            raise ValueError('Incomplete/duplicate paired v25 schedule')
        report.update(status='completed',defense_summary=test['overall'],
            by_difficulty={key:row for key,row in test['by']['difficulty'].items()},
            v25_summary=duels['baselines'][0]['summary'],
            defense_sha256=sha((output/'defense.json').read_bytes()),v25_sha256=sha((output/'v25.json').read_bytes()))
        save();print(json.dumps(report),flush=True)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise
    return report


if __name__=='__main__':
    if LAUNCH_CONFIG is None:raise RuntimeError('Prepared pinned launch configuration required')
    execute(LAUNCH_CONFIG)
