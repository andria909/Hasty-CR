"""Standalone, opt-in Kaggle controller for frozen post-stage2 evaluation.

Requires explicit mounted input roots and a pinned original-stage2 or recovery result SHA. Never
trains, uploads, picks a model from test results, or searches ambiguous mounts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tarfile

EXPECTED_PROTOCOL='c03b49396e95de983419211e476d732e1841ad97bb8d6a79c9b74ed97397437f'


def sha(data):return hashlib.sha256(data).hexdigest()


def relative_path(value):
    if not isinstance(value,str)or not value or '\\'in value:
        raise ValueError('Invalid relative resource path')
    path=PurePosixPath(value)
    if not path.parts or path.is_absolute()or '..'in path.parts or str(path)!=value:
        raise ValueError('Unsafe relative resource path')
    return path


def exact_file(root,name):
    relative=relative_path(name)
    root=Path(root).resolve();path=root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root)or path.is_symlink()or not path.is_file():
        raise ValueError(f'Missing or unsafe explicit input file: {name}')
    return path


def frozen_members(evaluation_input,*,extracted_root=None,expected_protocol=EXPECTED_PROTOCOL):
    evaluation_input=Path(evaluation_input)
    manifest=json.loads(exact_file(evaluation_input,'manifest.json').read_bytes())
    if manifest.get('evaluation_protocol_fingerprint')!=expected_protocol:
        raise ValueError('Evaluation protocol fingerprint mismatch')
    files=manifest['files']
    if not isinstance(files,dict)or not files:raise ValueError('Empty frozen file manifest')
    for name,digest in files.items():
        relative_path(name)
        if not isinstance(digest,str)or len(digest)!=64:raise ValueError('Invalid file SHA')
    archive=evaluation_input/'source.tar.gz'
    members={}
    if archive.exists():
        archive_bytes=exact_file(evaluation_input,'source.tar.gz').read_bytes()
        if sha(archive_bytes)!=manifest['archive_sha256']:raise ValueError('Frozen archive hash mismatch')
        with tarfile.open(fileobj=io.BytesIO(archive_bytes),mode='r:gz')as stream:
            for member in stream:
                name=str(relative_path(member.name))
                if not member.isfile()or name in members or name not in files:
                    raise ValueError('Unsafe, duplicate or unlisted frozen archive member')
                members[name]=stream.extractfile(member).read()
    else:
        source=Path(extracted_root)if extracted_root is not None else evaluation_input
        members={name:exact_file(source,name).read_bytes()for name in files}
    if set(members)!=set(files):raise ValueError('Missing frozen archive members')
    for name,data in members.items():
        if sha(data)!=files[name]:raise ValueError(f'Frozen member SHA mismatch: {name}')
    for name,digest in manifest.get('external_files',{}).items():
        if sha(exact_file(evaluation_input,name).read_bytes())!=digest:
            raise ValueError('External frozen file hash mismatch')
    protocol=json.loads(members['data/defense/evaluation_protocol.json'])
    if sha(members['data/defense/evaluation_protocol.json'])!=manifest['evaluation_protocol_sha256']:
        raise ValueError('Protocol byte SHA mismatch')
    fingerprint=sha(json.dumps(protocol,sort_keys=True).encode())
    if fingerprint!=expected_protocol:raise ValueError('Protocol contents mismatch')
    if protocol.get('medium_threshold_confirmed')is not False:
        raise ValueError('Unconfirmed medium threshold must remain explicit')
    return members,manifest,protocol


def selected_candidates(stage2_input,result_sha,manifest):
    result_bytes=exact_file(stage2_input,'stage2_result.json').read_bytes()
    if sha(result_bytes)!=result_sha:raise ValueError('Pinned stage2 result SHA mismatch')
    result=json.loads(result_bytes)
    if result.get('status')!='passed':raise ValueError('Stage2 result is not successfully completed')
    if result.get('input_checkpoint_sha256')!=manifest['checkpoint_sha256']:
        raise ValueError('Stage2 v25 provenance mismatch')
    if result.get('warmstarts')!=manifest.get('warmstarts'):
        raise ValueError('Stage2 warmstart provenance mismatch')
    required_steps={'ppo':2048000,'grpo':1000000}
    if result.get('targets')!={name+'_steps':steps for name,steps in required_steps.items()}:
        raise ValueError('Stage2 training targets differ from the requested long run')
    selected={}
    for name in ('ppo','grpo'):
        row=result[name]
        if not isinstance(row.get('step'),int) or row['step']<required_steps[name]:
            raise ValueError('Stage2 training step target has not been reached; resume training before final test')
        if row.get('returncode')!=0 or result.get(name+'_validation',{}).get('returncode')!=0:
            raise ValueError('Candidate training/validation command failed')
        relative=relative_path(row['selected_checkpoint'])
        if relative.parts[0]!=name or not relative.name.startswith('best_validation')or relative.suffix!='.pt':
            raise ValueError('Only explicitly validation-selected best checkpoints are admissible; no final/latest fallback')
        path=exact_file(stage2_input,str(relative))
        if sha(path.read_bytes())!=row['selected_sha256']:
            raise ValueError('Selected validation checkpoint SHA mismatch')
        selected[name]=dict(path=path,sha256=row['selected_sha256'],source=dict(row))
    return selected,result


def selected_recovery_candidates(stage2_input,result_sha,manifest):
    raw=exact_file(stage2_input,'recovery_result.json').read_bytes()
    if sha(raw)!=result_sha:
        raise ValueError('Pinned recovery result SHA mismatch')
    report=json.loads(raw)
    if (report.get('schema')!='defense-semantic-recovery-v1' or
            report.get('status')!='training_targets_completed_validation_selected' or
            report.get('not_exact_resume')is not True or
            report.get('original_stage2_status_not_overridden')is not True):
        raise ValueError('Recovery schema/status/nonexact provenance is incomplete')
    source_hash=manifest.get('recovery_source_archive_sha256')
    patch=manifest.get('evaluation_environment_patch',{})
    repaired_env_hash=patch.get('to_sha256')
    if (not isinstance(source_hash,str)or len(source_hash)!=64 or
            any(character not in '0123456789abcdef'for character in source_hash)or
            not isinstance(repaired_env_hash,str)or len(repaired_env_hash)!=64 or
            report.get('source_provenance',{}).get('archive_sha256')!=source_hash or
            patch.get('to_sha256')!=manifest.get('files',{}).get('sim/defense_env.py') or
            manifest.get('evaluation_repaired_files')!=['sim/defense_env.py']):
        raise ValueError('Recovery source does not match the repaired evaluation environment provenance')
    required={'ppo':2048000,'grpo':1000000}
    if report.get('targets')!={name+'_steps':target for name,target in required.items()}:
        raise ValueError('Recovery training targets differ from the requested long run')
    selected={}
    for name,target in required.items():
        counters=report.get('training_counters',{}).get(name,{})
        steps=counters.get('steps')
        if (not isinstance(steps,int)or isinstance(steps,bool)or steps<target or
                counters.get('target_met')is not True or counters.get('not_exact_resume')is not True):
            raise ValueError('Recovery training step target is incomplete')
        selection=report.get('validation_selection',{}).get(name,{})
        if (selection.get('split')!='validation' or selection.get('episodes')!=32 or
                selection.get('seed')!=900000):
            raise ValueError('Recovery selection must use the explicit fresh validation protocol')
        choice=selection.get('selected',{})
        candidates=selection.get('candidates')
        if (not isinstance(choice,dict)or not isinstance(candidates,list)or not candidates or
                any(not isinstance(row,dict)for row in candidates)or choice not in candidates):
            raise ValueError('Recovery selected checkpoint is absent from validation candidates')
        scores=[row.get('mean_reward')for row in candidates]
        if any(not isinstance(score,(int,float))or isinstance(score,bool)or not math.isfinite(score)for score in scores):
            raise ValueError('Recovery validation scores must be finite')
        if choice.get('mean_reward')!=max(scores):
            raise ValueError('Recovery checkpoint is not the explicit validation-best choice')
        artifact=report.get('selected_artifacts',{}).get(name,{})
        relative=relative_path(artifact.get('path'))
        if str(relative)!=f'selected/{name}.pt' or len(relative.parts)!=2:
            raise ValueError('Recovery selected artifact must have the exact self-contained relative path')
        digest=artifact.get('sha256')
        if (not isinstance(digest,str)or len(digest)!=64 or
                any(character not in '0123456789abcdef'for character in digest) or digest!=choice.get('sha256')):
            raise ValueError('Recovery selected artifact SHA does not match validation selection')
        path=exact_file(stage2_input,str(relative))
        if sha(path.read_bytes())!=digest:
            raise ValueError('Recovery selected artifact content SHA mismatch')
        selected[name]=dict(path=path,sha256=digest,source=dict(choice),selection_mode='semantic_recovery_fresh_validation')
    return selected,report


def commands(project,output,selected,protocol):
    seeds=','.join(map(str,protocol['defense_seeds']))
    paired=list(protocol['paired_seeds'])
    if paired!=list(range(710001,710011)):
        raise ValueError('This frozen duel CLI requires the exact contiguous paired seed schedule')
    result={}
    for gpu,name in enumerate(('ppo','grpo')):
        checkpoint=str(selected[name]['path'])
        result[name]=[
            [sys.executable,'-m','sim.eval_defense','--init',checkpoint,'--device',f'cuda:{gpu}',
             '--split','test','--all-scenarios','--seeds',seeds,'--max-ms','240000',
             '--output',str(output/f'{name}_defense.json')],
            [sys.executable,'-m','sim.eval_defense_duel','--candidate',checkpoint,
             '--candidate-mode','physical_live','--baseline-manifest',str(project/'data/defense/baselines.json'),
             '--device',f'cuda:{gpu}','--matches','20','--seed','710001',
             '--output',str(output/f'{name}_duels.json')]]
    return result


def aggregate_candidate(protocol,defense,duels,candidate_sha,gate):
    if defense.get('checkpoint_sha256')!=candidate_sha or duels.get('candidate_sha256')!=candidate_sha:
        raise ValueError('Evaluation output does not match the pinned selected checkpoint')
    records=defense['records'];rows=duels['head_to_head']
    expected={(sid,seed)for sid in protocol.test_scenario_ids for seed in protocol.defense_seeds}
    observed=[(r.get('scenario_id'),r.get('seed'))for r in records]
    if len(observed)!=len(expected)or set(observed)!=expected:
        raise ValueError('Incomplete, duplicate or off-protocol defense evaluation')
    planned={(baseline,seed,seat)for baseline in protocol.baseline_ids
             for seed in protocol.paired_seeds for seat in (-1,1)}
    actual=[(r.get('baseline_id'),r.get('seed'),r.get('candidate_side'))for r in rows]
    if len(actual)!=len(planned)or set(actual)!=planned:
        raise ValueError('Incomplete, duplicate or off-protocol duel evaluation')
    coverage=defense.get('coverage',{})
    complete=all(coverage.get(key)is True for key in ('metrics_complete','tower_metrics_complete',
        'scenario_catalog_known','all_expected_scenarios_seen'))and duels.get('coverage_complete_for_acceptance')is True
    verdict=gate(protocol,records,rows,coverage_complete=complete)
    return dict(selected_sha256=candidate_sha,defense_episodes=len(records),duel_games=len(rows),
        acceptance=verdict,observed_duel_summaries={r['label']:r['summary']for r in duels['baselines']},
        selection='fixed stage2 validation-selected checkpoint; no test-based model selection')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-input',type=Path,required=True)
    p.add_argument('--stage2-input',type=Path,required=True)
    provenance=p.add_mutually_exclusive_group(required=True)
    provenance.add_argument('--stage2-result-sha256')
    provenance.add_argument('--recovery-result-sha256')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extracted-root',type=Path)
    args=p.parse_args(argv)
    args.output=args.output.resolve()
    if args.output.exists():raise ValueError('Use a new output directory')
    members,manifest,protocol_data=frozen_members(args.evaluation_input,extracted_root=args.extracted_root)
    recovery_mode=args.recovery_result_sha256 is not None
    if recovery_mode:
        selected,source_result=selected_recovery_candidates(args.stage2_input,args.recovery_result_sha256,manifest)
    else:
        selected,source_result=selected_candidates(args.stage2_input,args.stage2_result_sha256,manifest)
    # Do not import any bundled project code until all frozen bytes and selections validate.
    args.output.mkdir(parents=True,exist_ok=False)
    project=args.output/'project';project.mkdir()
    for name,data in members.items():
        path=project.joinpath(*relative_path(name).parts);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    sys.path.insert(0,str(project));sys.path.insert(1,str(project/'scripts'))
    from sim.defense_protocol import EvaluationProtocol,acceptance_gate,load_baseline_allowlist
    protocol=EvaluationProtocol.from_dict(protocol_data)
    if protocol.fingerprint!=EXPECTED_PROTOCOL:raise ValueError('Frozen core protocol disagrees with verified protocol')
    load_baseline_allowlist(project/'data/defense/baselines.json',project_root=project)
    import torch
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:
        raise RuntimeError('Two real CUDA GPUs are required; no CPU fallback')
    for row in selected.values():
        blob=torch.load(row['path'],map_location='cpu',weights_only=False)
        if not all(torch.isfinite(value).all().item()for value in blob['state_dict'].values()):
            raise ValueError('Selected checkpoint contains nonfinite policy weights')
        if not recovery_mode and not math.isfinite(float(blob['best_validation'])):
            raise ValueError('Selected checkpoint lacks a finite validation best score')
    command_sets=commands(project,args.output,selected,protocol_data)
    report=dict(status='running',protocol_fingerprint=protocol.fingerprint,
        source_archive_sha256=manifest['source_archive_sha256'],evaluation_archive_sha256=manifest['archive_sha256'],
        stage2_result_sha256=args.stage2_result_sha256,recovery_result_sha256=args.recovery_result_sha256,
        input_result_mode='semantic_recovery'if recovery_mode else 'original_stage2',
        original_stage2_status_not_overridden=source_result.get('original_stage2_status_not_overridden')if recovery_mode else None,
        not_exact_resume=True if recovery_mode else None,
        recovery_source_archive_sha256=manifest.get('recovery_source_archive_sha256'),
        recovery_training_counters=source_result.get('training_counters')if recovery_mode else None,
        recovery_validation_selection=source_result.get('validation_selection')if recovery_mode else None,gpu_names=[torch.cuda.get_device_name(i)for i in range(2)],
        selected={name:dict(path=str(row['path']),sha256=row['sha256'])for name,row in selected.items()},
        source_result_provenance_limitations=([
            'semantic recovery changes environment semantics; this is not an exact trajectory continuation',
            'pinned recovery report and source hashes are provenance evidence, not a signed source attestation']if recovery_mode else [
            'stage2 result lacks source archive hash; pinned result plus warmstart/v25 hashes do not prove its full source provenance']),
        candidates={},quality_passed=None)
    (args.output/'evaluation_result.json').write_text(json.dumps(report,indent=2))
    environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    def run(name):
        for index,command in enumerate(command_sets[name]):
            with (args.output/f'{name}_{index}.log').open('x')as stream:
                subprocess.run(command,cwd=project,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
        defense=json.loads((args.output/f'{name}_defense.json').read_text())
        duels=json.loads((args.output/f'{name}_duels.json').read_text())
        return name,aggregate_candidate(protocol,defense,duels,selected[name]['sha256'],acceptance_gate)
    try:
        with ThreadPoolExecutor(max_workers=2)as pool:
            futures=[pool.submit(run,name)for name in ('ppo','grpo')]
            for future in futures:
                name,result=future.result();report['candidates'][name]=result
                (args.output/'evaluation_result.json').write_text(json.dumps(report,indent=2))
        report['status']='completed'
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        (args.output/'evaluation_result.json').write_text(json.dumps(report,indent=2))
    # Never select a winning algorithm or substitute another checkpoint using these TEST results.


if __name__=='__main__':main()
