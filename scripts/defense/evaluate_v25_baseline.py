"""User-requested v25 baseline on the exact frozen PPO300 defense test proxy."""
import json
import os
from pathlib import Path
import subprocess
import sys
from scripts.defense.evaluate_stage2 import frozen_members, relative_path, sha

LAUNCH_CONFIG=None


def execute(config,input_root=Path('/kaggle/input'),output=Path('/kaggle/working/baseline')):
    owner,slug=config['evaluation_dataset'].split('/')
    paths=[p for p in (input_root/slug,input_root/'datasets'/owner/slug) if p.is_dir()]
    if len(paths)!=1:raise ValueError('Expected one explicit evaluation mount')
    evaluation=paths[0]
    if sha((evaluation/'manifest.json').read_bytes())!=config['manifest_sha256']:
        raise ValueError('Evaluation manifest SHA mismatch')
    members,manifest,protocol=frozen_members(evaluation,extracted_root=evaluation/'source')
    if manifest['archive_sha256']!=config['source_sha256']:raise ValueError('Source SHA mismatch')
    checkpoint=config['checkpoint_path']
    if checkpoint not in members or sha(members[checkpoint])!=config['checkpoint_sha256']:
        raise ValueError('Canonical v25 SHA mismatch')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='v25-frozen-defense-baseline-v1',status='verified_inputs',
        checkpoint_sha256=config['checkpoint_sha256'],source_sha256=manifest['archive_sha256'],
        expected_episodes=len(protocol['test_scenario_ids'])*len(protocol['defense_seeds']),
        scope='Same frozen physical_live defense proxy as PPO300; not legacy-training v25 deployment',
        quality_accepted=False,test_used_for_selection=False)
    def save():(output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        project=output/'project';project.mkdir()
        for name,raw in members.items():
            target=project.joinpath(*relative_path(name).parts)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        sys.path[:0]=[str(project),str(project/'scripts')]
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        report.update(status='evaluating',gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        save();print(json.dumps(report),flush=True)
        command=[sys.executable,'-m','sim.eval_defense','--init',str(project/checkpoint),
            '--device','cuda:0','--split','test','--all-scenarios','--seeds',','.join(map(str,protocol['defense_seeds'])),
            '--action-delay-ms','150','--initial-elixir','10','--max-ms','240000','--output',str(output/'defense.json')]
        environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        with (output/'defense.log').open('x')as log:
            subprocess.run(command,cwd=project,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        raw=(output/'defense.json').read_bytes();data=json.loads(raw)
        expected={(sid,seed)for sid in protocol['test_scenario_ids']for seed in protocol['defense_seeds']}
        actual=[(r['scenario_id'],r['seed'])for r in data['records']]
        if len(actual)!=708 or len(expected)!=708 or set(actual)!=expected:
            raise ValueError('Incomplete/duplicate test schedule')
        if data['checkpoint_sha256']!=config['checkpoint_sha256']:raise ValueError('Wrong evaluated checkpoint')
        report.update(status='completed',episodes=len(actual),defense_sha256=sha(raw),
            defense_summary=data['overall'],by_difficulty=data['by']['difficulty'])
        save();print(json.dumps(report),flush=True)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise
    return report


if __name__=='__main__':
    if LAUNCH_CONFIG is None:raise RuntimeError('Prepared pinned launch configuration required')
    execute(LAUNCH_CONFIG)
