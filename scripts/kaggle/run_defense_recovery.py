"""Future GPU recovery controller; explicit verified inputs, never uploads jobs.

Invoking this controller runs training. Importing it or --help does not. Both
algorithms continue with repaired environment semantics, not an exact resume.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile

PPO_UPDATES = 2000
PPO_STEPS = 2048000
GRPO_STEPS = 1000000
BASELINE_SHA = 'd65ab55893592c0e4655a1882db459c4a7fb1ebd9e10afaf2daa4d2b369fc972'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or str(p) != name:
        raise ValueError('Unsafe input path')
    return p


def verify_recovered(root, receipt_sha):
    root = Path(root)
    receipt_path = root/'recovery_receipt.json'
    if sha(receipt_path) != receipt_sha:
        raise ValueError('Recovery receipt SHA mismatch')
    receipt = json.loads(receipt_path.read_text())
    if not receipt.get('complete') or receipt.get('remote_status') not in ('COMPLETE','ERROR','FAILED','CANCELLED','CANCELED'):
        raise ValueError('Recovery receipt is not complete and terminal')
    patterns = {'ppo_latest':r'ppo/update_\d{6}\.pt',
                'ppo_best_validation':r'ppo/best_validation_\d{6}\.pt',
                'grpo_boundary':r'grpo/boundary_u\d{6}_s\d{6}\.pt',
                'grpo_best_validation':r'grpo/best_validation_u\d{6}_s\d{6}\.pt'}
    selected = {}
    for role, name in receipt['selected_roles'].items():
        if role not in patterns or not re.fullmatch(patterns[role], name):
            raise ValueError('Unexpected recovery checkpoint role/path')
        safe_relative(name)
        path = root/name
        if sha(path) != receipt['files'][name]['sha256']:
            raise ValueError('Recovered weight SHA mismatch')
        selected[role] = path
    if not {'ppo_latest','grpo_boundary'} <= selected.keys():
        raise ValueError('Missing required completed-boundary checkpoints')
    return receipt, selected


def prepare_source(source, destination, *, extracted_root=None):
    source, destination = Path(source), Path(destination)
    manifest_path = source/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['checkpoint_sha256'] != BASELINE_SHA or manifest.get('not_exact_resume') is not True:
        raise ValueError('Unexpected baseline or recovery semantics')
    destination.mkdir(exist_ok=False)
    archive = source/'source.tar.gz'
    if archive.exists():
        if sha(archive) != manifest['archive_sha256']:
            raise ValueError('Source archive SHA mismatch')
        with tarfile.open(archive) as stream:
            entries = stream.getmembers()
            if {m.name for m in entries} != set(manifest['files']) or len(entries) != len(manifest['files']):
                raise ValueError('Source archive member manifest mismatch')
            for entry in entries:
                safe_relative(entry.name)
                if not entry.isfile():
                    raise ValueError('Only regular source archive members allowed')
            stream.extractall(destination, filter='data')
    else:
        # Exactly this known extracted root, never an ambiguous recursive search.
        extracted = Path(extracted_root).resolve() if extracted_root is not None else source.resolve()
        for relative in manifest['files']:
            safe_relative(relative)
            original = extracted/relative
            if not original.resolve().is_relative_to(extracted) or original.is_symlink():
                raise ValueError('Unsafe extracted source member')
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
    for relative, digest in manifest['files'].items():
        safe_relative(relative)
        if sha(destination/relative) != digest:
            raise ValueError(f'Source file SHA mismatch: {relative}')
    migration = destination/'data/defense/grpo_code_migration.json'
    if sha(migration) != manifest['recovery_migration_sha256']:
        raise ValueError('Pinned migration SHA mismatch')
    baseline_verified = False
    if (source/'v25.pt').exists():
        if sha(source/'v25.pt') != BASELINE_SHA:
            raise ValueError('Baseline SHA mismatch')
        baseline_verified = True
    return dict(manifest_sha256=sha(manifest_path), archive_sha256=manifest['archive_sha256'],
        source_file_count=len(manifest['files']), original_source_manifest_sha256=manifest.get('source_manifest_sha256'),
        original_source_archive_sha256=manifest.get('source_archive_sha256'),
        changed_files=manifest.get('recovery_changed_files'), baseline_sha256=BASELINE_SHA,
        baseline_file_verified=baseline_verified, migration_sha256=sha(migration))


def ppo_remaining(blob):
    update, step = blob['update'], blob['step']
    if not isinstance(update,int) or not isinstance(step,int) or not 0 <= update <= PPO_UPDATES:
        raise ValueError('Invalid PPO progress counters')
    if step != update*8*128:
        raise ValueError('PPO update/step counters disagree with frozen rollout configuration')
    return PPO_UPDATES-update


def commands(selected, ppo_blob, output, project, python=sys.executable):
    output, project = Path(output), Path(project)
    shared = ['--max-ms','240000','--wallclock-seconds','21600','--checkpoint-every','50','--seed','42']
    remaining = ppo_remaining(ppo_blob)
    result = {}
    if remaining:
        result['ppo'] = [python,'-m','sim.train_defense_ppo','--init',str(selected['ppo_latest']),
            '--resume',str(selected['ppo_latest']),'--output',str(output/'ppo'),*shared,
            '--device','cuda:0','--updates',str(remaining),'--envs','8','--rollout','128',
            '--epochs','3','--lr','0.00005','--eval-every','100','--eval-episodes','32']
    result['grpo'] = [python,'-m','sim.train_defense_grpo','--resume',str(selected['grpo_boundary']),
        '--resume-code-migration',str(project/'data/defense/grpo_code_migration.json'),
        '--output',str(output/'grpo'),*shared,'--device','cuda:1','--updates','5000',
        '--target-environment-steps','1000000','--group-size','8','--epochs','2','--lr','0.00003',
        '--validation-every','100','--validation-episodes','32']
    return result


def load_checkpoint(path):
    import torch
    blob = torch.load(path, map_location='cpu', weights_only=False)
    def finite(value):
        if isinstance(value,torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite(v)for v in value.values())
        if isinstance(value,(list,tuple)):return all(finite(v)for v in value)
        return True
    if not all(finite(blob.get(key,{}))for key in
               ('state_dict','reference_state_dict','optimizer','optimiser')):
        raise ValueError(f'Nonfinite checkpoint weights/optimizer: {path.name}')
    if 'state_dict' not in blob:
        raise ValueError('Missing checkpoint weights')
    return blob


def validate_resume_states(ppo, grpo):
    shared = dict(seed=42,max_ms=240000,initial_elixir=10,action_delay_ms=150,difficulty=None)
    for blob, expected in ((ppo,dict(shared,envs=8,rollout=128,epochs=3,lr=5e-5)),
                           (grpo,dict(shared,group_size=8,epochs=2,lr=3e-5))):
        config=blob.get('config',{})
        for key,value in expected.items():
            if key not in config or config[key]!=value:
                raise ValueError(f'Frozen recovery configuration mismatch: {key}')
        if not isinstance(config.get('scenario_ids'),list) or not config['scenario_ids']:
            raise ValueError('Missing original scenario curriculum IDs')
    if ppo['config']['scenario_ids']!=grpo['config']['scenario_ids']:
        raise ValueError('Recovered algorithms have different curriculum IDs')
    ppo_required=('optimiser','sampler','rng','environment_states','observations','infos','episode_counters')
    if not all(key in ppo for key in ppo_required) or ppo['environment_states'] is None:
        raise ValueError('PPO checkpoint lacks full environment/optimizer/RNG state')
    grpo_required=('reference_state_dict','optimizer','sampler','torch_rng','cuda_rng',
                   'action_rng','numpy_rng','python_rng','config')
    if not all(key in grpo for key in grpo_required) or grpo.get('checkpoint_phase')!='boundary' or not grpo.get('resume_supported'):
        raise ValueError('GRPO checkpoint lacks full completed-boundary resume state')
    if not isinstance(grpo.get('environment_steps'),int) or grpo['environment_steps']<0:
        raise ValueError('Invalid GRPO completed-step counter')
    ppo_remaining(ppo)


def live_summary(line,child=None):
    """Keep scalar progress before large sampler/episode arrays in live stdout."""
    try:
        record=json.loads(line)
    except (ValueError,TypeError):
        return line.rstrip()[:1000]
    if not isinstance(record,dict):return line.rstrip()[:1000]
    keys=('event','update','updates','step','environment_steps','elapsed_seconds',
          'actual_cuda','device','cuda_available','loss','gradient_norm')
    summary={key:record[key]for key in keys if key in record}
    target={'ppo':PPO_STEPS,'grpo':GRPO_STEPS}.get(child)
    completed=record.get('step'if child=='ppo'else 'environment_steps')
    if target and isinstance(completed,int) and not isinstance(completed,bool) and completed>=0:
        summary.update(training_percent=round(min(100.,100*completed/target),2),target_steps=target)
    if isinstance(record.get('validation'),dict):
        summary['validation']={key:record['validation'][key]for key in
                               ('mean_reward','mean_return')if key in record['validation']}
    return json.dumps(summary) if summary else line.rstrip()[:1000]


def run_logged(command, project, log):
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with Path(log).open('x') as stream:
        name=Path(log).stem
        process = subprocess.Popen(command, cwd=project, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True,bufsize=1)
        print(json.dumps(dict(event='child_started',child=name,pid=process.pid)),flush=True)
        for line in process.stdout:
            stream.write(line);stream.flush()
            print(name+': '+live_summary(line,child=name),flush=True)
        code = process.wait()
        print(json.dumps(dict(event='child_completed',child=name,pid=process.pid,returncode=code)),flush=True)
    if code:
        raise RuntimeError(f'Child failed ({code}); inspect {log}')


VALIDATE = '''import json,sys,torch
from sim.train_defense_ppo import evaluate
from sim.train_ppo import build_network,checkpoint_architecture,load_policy_state
from sim.env import NUM_PLANES,NUM_SCALARS,ACTIONS
blob=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
net=build_network(NUM_PLANES,NUM_SCALARS,ACTIONS,checkpoint_architecture(blob)).to(sys.argv[3])
load_policy_state(net,blob['state_dict'])
report=evaluate(net,torch.device(sys.argv[3]),episodes=32,seed=900000,env_options=dict(action_delay_ms=150,max_ms=240000,initial_elixir=10,difficulty=None))
with open(sys.argv[2],'x') as out:json.dump(report,out,allow_nan=False)
'''


def choose_fresh_validation(candidates, algorithm, output, project):
    rows = []
    for i, path in enumerate(sorted(set(map(Path,candidates)))):
        destination = output/f'{algorithm}_selection_{i}.json'
        run_logged([sys.executable,'-c',VALIDATE,str(path),str(destination),
                    'cuda:0' if algorithm=='ppo' else 'cuda:1'], project,
                   output/f'{algorithm}_selection_{i}.log')
        value = json.loads(destination.read_text())['mean_reward']
        import math
        if not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Missing/nonfinite fresh validation reward')
        rows.append(dict(checkpoint=str(path),sha256=sha(path),mean_reward=value,
                         validation_report=str(destination),validation_sha256=sha(destination)))
    return dict(selected=max(rows,key=lambda r:r['mean_reward']),candidates=rows,
                split='validation',episodes=32,seed=900000,quality_accepted=False)


def materialize_selected_artifact(selection, algorithm, output):
    if algorithm not in ('ppo','grpo'):
        raise ValueError('Unknown selected algorithm')
    selected=selection['selected']
    source=Path(selected['checkpoint'])
    if sha(source)!=selected['sha256']:
        raise ValueError('Selected weights changed after validation')
    target=Path(output)/'selected'/f'{algorithm}.pt'
    target.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as incoming,target.open('xb') as outgoing:
        shutil.copyfileobj(incoming,outgoing)
    if sha(target)!=selected['sha256']:
        raise ValueError('Selected artifact copy SHA mismatch')
    return dict(path=target.relative_to(output).as_posix(),sha256=selected['sha256'])


def execute(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    report = dict(schema='defense-semantic-recovery-v1',status='started',not_exact_resume=True,
                  quality_accepted=False,original_stage2_status_not_overridden=True,
                  targets=dict(ppo_steps=PPO_STEPS,grpo_steps=GRPO_STEPS),children={})
    def save():
        (output/'recovery_result.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        provenance = prepare_source(Path(args.source_input).resolve(),output/'project',
                                    extracted_root=getattr(args,'source_extracted_root',None))
        report['source_provenance'] = provenance
        receipt, selected = verify_recovered(Path(args.recovered_input).resolve(),args.recovery_receipt_sha256)
        report.update(recovery_receipt_sha256=args.recovery_receipt_sha256,
                      remote_hash_verified=all(receipt['files'][str(p.relative_to(Path(args.recovered_input).resolve()))].get('remote_hash_verified',False)for p in selected.values()))
        # Resolve pickle classes from the verified patched source, not an
        # incidental checkout or another mounted dataset.
        sys.path[:0]=[str(output/'project'),str(output/'project/scripts')]
        blobs = {role:load_checkpoint(path) for role,path in selected.items()}
        grpo = blobs['grpo_boundary']
        validate_resume_states(blobs['ppo_latest'],grpo)
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()<2:
            raise RuntimeError('Recovery requires two CUDA devices; CPU fallback forbidden')
        plan = commands(selected,blobs['ppo_latest'],output,output/'project')
        report['commands'] = plan
        save()
        errors = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(run_logged,command,output/'project',output/f'{name}.log'):name for name,command in plan.items()}
            for future in as_completed(futures):
                name=futures[future]
                try:
                    future.result()
                    report['children'][name]=dict(status='completed',not_exact_resume=True)
                except Exception as error:
                    report['children'][name]=dict(status='failed',error=str(error),not_exact_resume=True)
                    errors.append(name)
                save()
        if errors:
            raise RuntimeError('Recovery child failed: '+','.join(errors))
        ppo_files=sorted((output/'ppo').glob('update_*.pt'))
        grpo_files=sorted((output/'grpo').glob('final_u*_s*.pt'))
        finals={'ppo':ppo_files[-1] if ppo_files else selected['ppo_latest'],
                'grpo':grpo_files[-1] if grpo_files else None}
        if finals['grpo'] is None:
            raise ValueError('Missing new GRPO final checkpoint')
        report['training_counters']={}
        for algorithm,path in finals.items():
            blob=load_checkpoint(path)
            steps=blob['step'] if algorithm=='ppo' else blob['environment_steps']
            updates=blob['update'] if algorithm=='ppo' else blob['updates']
            report['training_counters'][algorithm]=dict(steps=steps,updates=updates,
                checkpoint=str(path),sha256=sha(path),target_met=steps >= (PPO_STEPS if algorithm=='ppo' else GRPO_STEPS),
                not_exact_resume=True)
        save()
        if not all(row['target_met']for row in report['training_counters'].values()):
            raise ValueError('Wallclock ended before requested training targets; recovery incomplete')
        report['validation_selection']={}
        report['selected_artifacts']={}
        for algorithm in ('ppo','grpo'):
            candidates=[finals[algorithm]]+sorted((output/algorithm).glob('best_validation*.pt'))[-1:]
            if algorithm=='ppo' and 'ppo_best_validation' in selected:
                candidates.append(selected['ppo_best_validation'])
            report['validation_selection'][algorithm]=choose_fresh_validation(candidates,algorithm,output,output/'project')
            report['selected_artifacts'][algorithm]=materialize_selected_artifact(
                report['validation_selection'][algorithm],algorithm,output)
            save()
        report['status']='training_targets_completed_validation_selected'
        save()
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        save()
        raise
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-input',type=Path,required=True)
    parser.add_argument('--source-extracted-root',type=Path,help='Explicit Kaggle source/ directory if archives are automatically extracted')
    parser.add_argument('--recovered-input',type=Path,required=True)
    parser.add_argument('--recovery-receipt-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    execute(parser.parse_args(argv))


if __name__=='__main__':main()
