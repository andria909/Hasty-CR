"""Pinned joint-policy validation/test and all15 historical paired opponents.

One selected policy, no training or selection from test, no live games.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

LAUNCH_CONFIG=None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_source(source,output,expected):
    source=Path(source)
    if sha(source/'manifest.json')!=expected:raise ValueError('Evaluation manifest SHA mismatch')
    manifest=json.loads((source/'manifest.json').read_text())
    files=manifest['files']
    for name in files:
        p=PurePosixPath(name)
        if p.is_absolute()or '..'in p.parts or '\\'in name or str(p)!=name:
            raise ValueError('Unsafe evaluation source path')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    archive=source/'source.tar.gz'
    if archive.exists():
        if sha(archive)!=manifest['archive_sha256']:raise ValueError('Evaluation archive SHA mismatch')
        with tarfile.open(archive)as stream:
            entries=stream.getmembers()
            if ({m.name for m in entries}!=set(files)or len(entries)!=len(files)
                    or any(not m.isfile()for m in entries)):
                raise ValueError('Unsafe/duplicate evaluation archive members')
            stream.extractall(output,filter='data')
    else:
        extracted=(source/'source').resolve()
        for name in files:
            original=extracted/name
            if original.is_symlink()or not original.resolve().is_relative_to(extracted):
                raise ValueError('Unsafe extracted evaluation member')
            target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
    for name,digest in files.items():
        if sha(output/name)!=digest:raise ValueError('Evaluation source member SHA mismatch')
    return manifest


def commands(project,policy,output,protocol_sha):
    protocol=project/'data/strategy/evaluation_protocol.json'
    shared=[sys.executable,'-m','sim.eval_strategy','--checkpoint',str(policy),
        '--checkpoint-sha256',sha(policy),'--protocol',str(protocol),'--protocol-sha256',protocol_sha,'--device','cuda:0']
    return {'skills':[(name,[*shared,'--split',name,'--output',str(output/(name+'.json'))])for name in ('validation','test')],
        'duels':[('duels',[sys.executable,'-m','sim.eval_defense_duel','--candidate',str(policy),
            '--candidate-mode','joint_enemy_presence_v1','--baseline-manifest',str(project/'data/defense/baselines.json'),
            '--matches','20','--seed','710001','--device','cuda:1','--output',str(output/'duels.json')])]}


def execute(config,input_root=Path('/kaggle/input'),output=Path('/kaggle/working/strategy_evaluation')):
    def mount(reference):
        owner,slug=reference.split('/')
        found=[p for p in (input_root/slug,input_root/'datasets'/owner/slug)if p.is_dir()]
        if len(found)!=1:raise ValueError('Expected one exact joint evaluation mount')
        return found[0]
    source=mount(config['evaluation_dataset']);weights=mount(config['selected_dataset'])
    if sha(weights/'selection.json')!=config['selection_sha256']or sha(weights/'policy.pt')!=config['policy_sha256']:
        raise ValueError('Pinned validation selection/policy SHA mismatch')
    selection=json.loads((weights/'selection.json').read_text());audit=selection['terminal_training_audit']
    if (selection.get('schema')!='joint-strategy-selection-v1'or selection.get('test_used_for_selection')is not False
            or selection.get('selection_split')!='validation'or audit.get('target_met')is not True
            or selection.get('selected_sha256')!=config['policy_sha256']):
        raise ValueError('Completed training and pre-test validation selection required')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='joint-final-evaluation-v1',status='verified_selection',algorithm=selection['algorithm'],
        selected_sha256=config['policy_sha256'],selection_sha256=config['selection_sha256'],
        terminal_training_steps=audit['steps'],quality_accepted=False,test_used_for_selection=False,parts={})
    def save():(output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        project=output/'project'
        manifest=prepare_source(source,project,config['manifest_sha256'])
        if manifest['training_source_archive_sha256']!=audit['source_archive_sha256']:
            raise ValueError('Evaluation source does not descend from selected training source')
        protocol=project/'data/strategy/evaluation_protocol.json'
        if sha(protocol)!=config['protocol_sha256']:raise ValueError('Joint protocol SHA mismatch')
        report.update(source_sha256=manifest['archive_sha256'],training_source_sha256=audit['source_archive_sha256'])
        sys.path[:0]=[str(project),str(project/'scripts')]
        import torch
        if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA GPUs required')
        torch.set_num_threads(2)
        from sim.eval_strategy import load_selected
        # Source/observation contracts also verify before any test is executed.
        load_selected(weights/'policy.pt','cpu',json.loads(protocol.read_text())['environment'])
        report.update(status='evaluating',gpu_names=[torch.cuda.get_device_name(i)for i in range(2)])
        save();print(json.dumps(report),flush=True)
        environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        def run_chain(chain):
            for name,args in chain:
                with (output/(name+'.log')).open('x')as log:
                    subprocess.run(args,cwd=project,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
                part=json.loads((output/(name+'.json')).read_text())
                brief=(dict(baselines=[dict(id=b.get('id'),summary=b['summary'])for b in part['baselines']])if name=='duels'else
                       dict(episodes=part['episodes'],attack=part['attack']['overall'],defense=part['defense']['overall']))
                print(json.dumps(dict(event='part_completed',part=name,summary=brief)),flush=True)
            return True
        plan=commands(project,weights/'policy.pt',output,config['protocol_sha256'])
        with ThreadPoolExecutor(2)as pool:list(pool.map(run_chain,plan.values()))
        test=json.loads((output/'test.json').read_text());duels=json.loads((output/'duels.json').read_text())
        if test.get('checkpoint_sha256')!=config['policy_sha256']or duels.get('candidate_sha256')!=config['policy_sha256']:
            raise ValueError('Evaluation reports a different selected policy')
        from sim.defense_protocol import EvaluationProtocol, acceptance_gate
        protocol_data=json.loads((project/'data/defense/evaluation_protocol.json').read_text())
        frozen=EvaluationProtocol.from_dict(protocol_data)
        defense=[r for r in test['records']if r['task']=='defense']
        expected={(sid,seed)for sid in frozen.test_scenario_ids for seed in frozen.defense_seeds}
        actual=[(r['scenario_id'],r['seed'])for r in defense]
        if len(actual)!=len(expected)or set(actual)!=expected:raise ValueError('Incomplete/duplicate full defense test')
        pairs=duels['head_to_head'];expected_pairs={(b,s,seat)for b in frozen.baseline_ids for s in frozen.paired_seeds for seat in (1,-1)}
        actual_pairs=[(r['baseline_id'],r['seed'],r['candidate_side'])for r in pairs]
        if len(actual_pairs)!=len(expected_pairs)or set(actual_pairs)!=expected_pairs:
            raise ValueError('Incomplete/duplicate historical opponent schedule')
        for name in ('validation','test','duels'):
            report['parts'][name]=dict(file=name+'.json',sha256=sha(output/(name+'.json')))
        coverage=test['defense']['coverage']
        complete=all(coverage.get(k)is True for k in ('metrics_complete','tower_metrics_complete',
            'scenario_catalog_known','all_expected_scenarios_seen'))and duels.get('coverage_complete_for_acceptance')is True
        report.update(status='completed',test_episodes=test['episodes'],defense_test_episodes=len(defense),
            duel_games=len(pairs),defense=test['defense']['overall'],attack=test['attack']['overall'],
            baselines=[dict(id=b.get('id'),summary=b['summary'])for b in duels['baselines']],
            acceptance=acceptance_gate(frozen,defense,pairs,coverage_complete=complete),
            quality_note='Unvalidated simulator mechanics and explicit public heuristic task switching remain unresolved')
        save();print(json.dumps(report),flush=True)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise
    return report


if __name__=='__main__':
    if LAUNCH_CONFIG is None:raise RuntimeError('Prepared pinned joint evaluation config required')
    execute(LAUNCH_CONFIG)
