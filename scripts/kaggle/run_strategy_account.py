"""One shared attack/defense policy per account; pinned GPU-only runner."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import tarfile

LAUNCH_CONFIG = None


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress_line(line, algorithm, target):
    """Normalize trainer counters without inferring progress from validation."""
    try:
        row=json.loads(line)
    except (ValueError,TypeError):
        return line.rstrip()[:1000]
    if not isinstance(row,dict):return line.rstrip()[:1000]
    step=next((row[k]for k in ('environment_steps','step','steps')if k in row),None)
    keys=('event','update','actual_cuda','device','cuda_available','loss','elapsed_seconds')
    summary={k:row[k]for k in keys if k in row}
    if isinstance(step,int)and not isinstance(step,bool)and step>=0:
        summary.update(environment_steps=step,target_steps=target,
                       training_percent=round(min(100.,100*step/target),2))
    return json.dumps(summary)if summary else line.rstrip()[:1000]


def commands(config,project,input_root,output):
    algorithm=config['algorithm']
    args=[sys.executable,'-m','sim.train_attack_'+algorithm,'--init',str(input_root/'v25.pt'),
        '--output',str(output/algorithm),'--task-mode','joint','--device','cuda:0',
        '--target-environment-steps',str(config['target_steps']),'--updates','100000',
        '--initial-hand-mode','shuffle','--deadline-scale','3','--seed','42',
        '--geometry','legacy_v1','--coordinate-mode','physical_live_cells_v2',
        '--max-ms','240000','--wallclock-seconds','21600','--checkpoint-every','50',
        '--eval-every','100','--eval-episodes','32']
    if algorithm=='ppo':args+=['--envs','8','--rollout','128','--lr','0.00005']
    elif algorithm=='grpo':args+=['--group-size','8','--epochs','2','--lr','0.00003']
    else:raise ValueError('Unsupported strategy algorithm')
    return args


def execute(config,input_root=Path('/kaggle/input'),output=Path('/kaggle/working/strategy')):
    owner,slug=config['dataset'].split('/')
    found=[p for p in (input_root/slug,input_root/'datasets'/owner/slug)if p.is_dir()]
    if len(found)!=1:raise ValueError('Expected one explicit joint strategy mount')
    source=found[0]
    for name,expected in config['pins'].items():
        if name not in ('manifest.json','v25.pt','telemetry.py')or digest(source/name)!=expected:
            raise ValueError('Pinned strategy input SHA mismatch')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='joint-strategy-account-v1',status='started',owner=owner,
        algorithm=config['algorithm'],target_steps=config['target_steps'],quality_accepted=False,
        scope='One shared policy trained on balanced attack and defense skill episodes; full-match acceptance pending')
    def save():(output/'strategy_result.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        manifest=json.loads((source/'manifest.json').read_text())
        project=output/'project';project.mkdir()
        for relative in manifest['files']:
            path=PurePosixPath(relative)
            if path.is_absolute()or '..'in path.parts or '\\'in relative:
                raise ValueError('Unsafe source member')
        archive=source/'source.tar.gz'
        if archive.exists():
            if digest(archive)!=manifest['archive_sha256']:raise ValueError('Archive SHA mismatch')
            with tarfile.open(archive)as stream:
                members=stream.getmembers()
                if len(members)!=len(manifest['files'])or {m.name for m in members}!=set(manifest['files']):
                    raise ValueError('Archive members differ from manifest')
                if any(not m.isfile()for m in members):raise ValueError('Only regular source files allowed')
                stream.extractall(project,filter='data')
        else:
            for relative in manifest['files']:
                original=source/'source'/relative
                if original.is_symlink():raise ValueError('Source symlink rejected')
                target=project/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(original,target)
        for relative,expected in manifest['files'].items():
            if digest(project/relative)!=expected:raise ValueError('Extracted source SHA mismatch')
        report['source_archive_sha256']=manifest['archive_sha256'];save()
        sys.path[:0]=[str(project),str(project/'scripts')]
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU fallback forbidden')
        report.update(gpu_names=[torch.cuda.get_device_name(i)for i in range(torch.cuda.device_count())],
            torch_version=str(torch.__version__),status='training')
        save();print(json.dumps(report),flush=True)
        spec=importlib.util.spec_from_file_location('strategy_telemetry',source/'telemetry.py')
        telemetry=importlib.util.module_from_spec(spec);spec.loader.exec_module(telemetry)
        telemetry.live_summary=lambda line,child=None:progress_line(line,config['algorithm'],config['target_steps'])
        args=commands(config,project,source,output);report['command']=args;save()
        # Same telemetry as tested recovery, now with distinct account/task scope.
        telemetry.run_logged(args,project,output/(config['algorithm']+'.log'))
        report.update(status='trainer_completed',target_status_requires_training_report=True)
        save();print(json.dumps(report),flush=True)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise
    return report


if __name__=='__main__':
    if LAUNCH_CONFIG is None:raise RuntimeError('Use a prepared self-contained job')
    execute(LAUNCH_CONFIG)
