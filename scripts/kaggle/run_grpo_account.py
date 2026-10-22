"""Standalone pinned single-trainer GRPO continuation on another account."""
import argparse
import importlib.util
import json
from pathlib import Path
import hashlib

LAUNCH_CONFIG = None


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(project,checkpoint,output,python):
    return [python,'-m','sim.train_defense_grpo','--resume',str(checkpoint),
        '--resume-code-migration',str(project/'data/defense/grpo_code_migration.json'),
        '--output',str(output/'grpo'),'--device','cuda:1','--updates','5000',
        '--target-environment-steps','1000000','--group-size','8','--epochs','2',
        '--lr','0.00003','--max-ms','240000','--wallclock-seconds','21600',
        '--checkpoint-every','50','--seed','42','--validation-every','100',
        '--validation-episodes','32']


def execute(config,input_root=Path('/kaggle/input'),output=Path('/kaggle/working/account2')):
    owner,slug=config['dataset'].split('/')
    if not owner or not slug or any(part in ('.','..')for part in (owner,slug)):
        raise ValueError('Invalid explicit dataset reference')
    mounts=[input_root/slug,input_root/'datasets'/owner/slug]
    found=[p for p in mounts if p.is_dir()]
    if len(found)!=1:raise ValueError('Expected one explicit account2 input mount')
    source=found[0]
    for name in ('manifest.json','controller.py','grpo_resume.pt'):
        if digest(source/name)!=config['pins'][name]:raise ValueError('Pinned input SHA mismatch: '+name)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='defense-grpo-account2-v1',status='started',owner=owner,
        target_steps=1000000,quality_accepted=False,not_exact_resume=True,
        checkpoint_sha256=config['pins']['grpo_resume.pt'])
    def save():(output/'account2_result.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        spec=importlib.util.spec_from_file_location('pinned_grpo_controller',source/'controller.py')
        controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
        project=output/'project'
        report['source_provenance']=controller.prepare_source(source,project,extracted_root=source/'source')
        import sys
        sys.path[:0]=[str(project),str(project/'scripts')]
        blob=controller.load_checkpoint(source/'grpo_resume.pt')
        if blob.get('checkpoint_phase')!='boundary' or blob.get('resume_supported')is not True:
            raise ValueError('Completed-boundary resume checkpoint required')
        report['restored_steps']=blob['environment_steps']
        report['restored_update']=blob['updates']
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()<2:
            raise RuntimeError('Two CUDA devices required for the preserved cuda:1 resume contract')
        report.update(status='training',torch_version=str(torch.__version__),
            gpu_names=[torch.cuda.get_device_name(i)for i in range(torch.cuda.device_count())])
        save();print(json.dumps(report),flush=True)
        args=command(project,source/'grpo_resume.pt',output,sys.executable)
        report['command']=args;save()
        controller.run_logged(args,project,output/'grpo.log')
        finals=sorted((output/'grpo').glob('final_u*_s*.pt'))
        if not finals:raise ValueError('No GRPO final checkpoint')
        final=controller.load_checkpoint(finals[-1])
        report['steps']=final['environment_steps'];report['updates']=final['updates']
        if report['steps']<1000000:raise ValueError('Training ended before target steps')
        candidates=[finals[-1]]+sorted((output/'grpo').glob('best_validation*.pt'))[-1:]
        report['validation_selection']=controller.choose_fresh_validation(candidates,'grpo',output,project)
        report['selected_artifact']=controller.materialize_selected_artifact(report['validation_selection'],'grpo',output)
        report['status']='grpo_target_completed_validation_selected';save()
        print(json.dumps(report),flush=True)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())if args.config else LAUNCH_CONFIG
    if config is None:parser.error('--config required for unbound runner')
    execute(config)


if __name__=='__main__':main()
