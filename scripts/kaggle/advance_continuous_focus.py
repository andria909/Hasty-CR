"""Bounded, idempotent handoff from verified v2 training to frozen focus study.

No remote calls while prerequisites are absent. Never re-push a kernel after
an ambiguous submission; observations/collectors may reconnect independently.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def project_python():
    # Resolving the executable symlink bypasses pyvenv.cfg and loses torch.
    return str(Path('.venv-live/bin/python').absolute())


def bind_verified_inputs(template,training_report,output,owner):
    env=dict(os.environ)
    env['PYTHONPATH']=str(Path.cwd())+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([project_python(),str((Path(template)/'bind.py').resolve()),
        '--template',str(template),'--training-report',str(training_report),'--output',str(output),
        '--owner',owner],env=env,check=True)


def prerequisites(training,evaluation):
    if not Path(training).exists()or not Path(evaluation).exists():return False
    try:t=json.loads(Path(training).read_text());e=json.loads(Path(evaluation).read_text())
    except(ValueError,OSError):return False
    return(t.get('schema')=='verified-continuous-ppo-training-v1'and t.get('all_training_budgets_verified')is True
        and t.get('kernel')=='andreiperelaiko/hastycr-continuous-slots-20261026'
        and e.get('schema')=='verified-continuous-frozen-pool-v1'and e.get('total_games')==600)


def http_status(error):
    return getattr(getattr(error,'response',None),'status_code',None)or getattr(error,'status',None)


def kernel_exists(api,owner,kernel):
    """Use authenticated ownership listing; SDK masks some absent slugs as403."""
    for page in range(1,101):
        rows=api.kernels_list(mine=True,page=page,page_size=100)
        if rows is None:raise ValueError('Owned kernel listing returned no response')
        for row in rows:
            ref=getattr(row,'ref',None)
            if not isinstance(ref,str)or not ref.startswith(owner+'/'):raise ValueError('Owned kernel identity differs')
            if ref==kernel:return True
        if len(rows)<100:return False
    raise ValueError('Owned kernel listing exceeded bounded search')


def dataset_exists(api,owner,slug):
    """Disambiguate SDK403 for a missing slug using authenticated own listing."""
    for page in range(1,101):
        rows=api.dataset_list(mine=True,search=slug.split('/')[-1],page=page)
        if rows is None:raise ValueError('Owned dataset listing returned no response')
        for row in rows:
            ref=getattr(row,'ref',None)
            if not isinstance(ref,str)or not ref.startswith(owner+'/'):raise ValueError('Owned dataset identity differs')
            if ref==slug:return True
        if not rows:return False
    raise ValueError('Owned dataset listing exceeded bounded search')


def recover_submission(api,kernel,output):
    """Read remote code only; recover a lost receipt iff exact code matches."""
    output=Path(output);prepared=json.loads((output/'preparation.json').read_text())
    recovery=output/('submission_recovery_'+str(time.time_ns()));recovery.mkdir()
    api.kernels_pull(kernel,str(recovery),metadata=True,quiet=True)
    candidates=[p for p in recovery.rglob('*.py')if not p.is_symlink()and p.resolve().is_relative_to(recovery.resolve())]
    if not any(sha(p)==prepared['bootstrap_sha256']for p in candidates):return False
    (output/'launch_receipt.json').write_text(json.dumps(dict(kernel=kernel,recovered_from_exact_remote_code=True,
        bootstrap_sha256=prepared['bootstrap_sha256']),indent=2)+'\n')
    return True


def run(args,api_factory=None):
    state_dir=Path(args.state);state_dir.mkdir(parents=True,exist_ok=True)
    lock=(state_dir/'coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=state_dir/'state.json'
    state=json.loads(state_path.read_text())if state_path.exists()else dict(phase='waiting_verified_prerequisites',
        template=str(args.template),output=str(args.output),owner=args.owner,kernel=args.owner+'/hastycr-continuous-focus-20261026')
    def save(**values):
        old=state.get('phase');state.update(values,checked_ns=time.time_ns())
        state_path.write_text(json.dumps(state,indent=2)+'\n')
        if old!=state.get('phase'):print(json.dumps(dict(phase=state['phase'],kernel=state['kernel'])),flush=True)
    template=Path(args.template);manifest=json.loads((template/'template_manifest.json').read_text())
    if any(sha(template/n)!=h for n,h in manifest['files'].items()):raise ValueError('Frozen handoff template changed')
    deadline=time.monotonic()+172800;next_remote=0;next_sync=0;api=None
    while time.monotonic()<deadline:
        if not prerequisites(args.training_report,args.evaluation_report):
            save(phase='waiting_verified_prerequisites');time.sleep(30);continue
        output=Path(args.output)
        if not(output/'preparation.json').exists():
            if output.exists():raise ValueError('Partial preparation preserved; inspect before rebinding')
            save(phase='binding_verified_weights')
            bind_verified_inputs(template,args.training_report,output,args.owner)
        preparation=json.loads((output/'preparation.json').read_text())
        if preparation['kernel']!=state['kernel']:raise ValueError('Prepared kernel changed')
        if api is None:
            from scripts.kaggle.collect_final_evaluation import make_api_client,configure_http_timeout
            if api_factory is not None:api=api_factory(args.owner)
            else:
                token=json.loads(Path('/Users/andriashadrin/.kaggle/access_tokens.json').read_text())['tokens'][args.token_index]
                os.environ['KAGGLE_API_TOKEN']=token if isinstance(token,str)else token['token']
                configure_http_timeout();api=make_api_client(args.owner)
        if time.monotonic()<next_remote:
            if time.monotonic()>=next_sync:
                from scripts.kaggle.watch_continuous_training import sync_status
                if sync_status(output,args.status_report):
                    subprocess.run([project_python(),'-m','scripts.defense.update_done'],check=True)
                next_sync=time.monotonic()+1200
            time.sleep(30);continue
        next_remote=time.monotonic()+1200
        slug=args.owner+'/hastycr-continuous-focus-inputs-20261026'
        try:
            if not state.get('dataset_ready'):
                try:status=api.dataset_status(slug)
                except Exception as exc:
                    if http_status(exc)==403:
                        if dataset_exists(api,args.owner,slug):raise
                    elif http_status(exc)!=404:raise
                    status=None
                if str(status).lower()in('not_found','notfound'):status=None
                if status is None:
                    if state.get('dataset_request_sent'):
                        save(phase='dataset_submission_uncertain_no_duplicate');continue
                    save(phase='uploading_frozen_focus_inputs',dataset_request_sent=True)
                    response=api.dataset_create_new(str(output/'dataset'),public=False,quiet=True,convert_to_csv=False,dir_mode='tar')
                    (output/'dataset_create.json').write_text(json.dumps(response.to_dict()if hasattr(response,'to_dict')else str(response),indent=2)+'\n')
                    continue
                if not state.get('dataset_request_sent'):
                    # Existing inputs are versioned explicitly; old versions are retained.
                    save(phase='versioning_frozen_focus_inputs',dataset_request_sent=True)
                    response=api.dataset_create_version(str(output/'dataset'),'Bind verified corrected-slot references and frozen focus protocol',
                        quiet=True,convert_to_csv=False,delete_old_versions=False,dir_mode='tar')
                    (output/'dataset_version.json').write_text(json.dumps(response.to_dict()if hasattr(response,'to_dict')else str(response),indent=2)+'\n')
                    continue
                if str(status).lower()!='ready':save(phase='waiting_dataset_ready');continue
                save(dataset_ready=True,phase='inputs_ready')
            if not state.get('push_requested'):
                if kernel_exists(api,args.owner,state['kernel']):
                    save(phase='existing_kernel_needs_provenance_check');return
                save(phase='submitting_focus_kernel',push_requested=True)
                response=api.kernels_push(str(output/'job'),acc='NvidiaTeslaT4')
                (output/'launch_receipt.json').write_text(json.dumps(response.to_dict()if hasattr(response,'to_dict')else str(response),indent=2)+'\n')
                save(push_receipt_saved=True)
            if not state.get('push_receipt_saved'):
                # Observe the same handle; do not submit another GPU job.
                remote=api.kernels_status(state['kernel'])
                if not recover_submission(api,state['kernel'],output):
                    save(phase='submission_uncertain_observation_only',remote_status=json.loads(str(remote)));continue
                save(phase='submission_receipt_recovered_from_exact_code',push_receipt_saved=True)
            if not state.get('monitors'):
                monitors={};sdk=Path('/Users/andriashadrin/.local/share/hastycr-kaggle-cli/bin/python')
                commands={
                    'stream':[str(sdk),'-m','scripts.kaggle.stream_training','--kernel',state['kernel'],'--output',str(output/'live_stream'),
                        '--api-user',args.owner,'--reconnect-seconds','1200','--reconnects','100'],
                    'collector':[str(sdk),'-m','scripts.kaggle.collect_final_evaluation','--kernel',state['kernel'],'--output',str(output/'collected'),
                        '--profile','continuous-focus','--api-user',args.owner,'--poll-seconds','1200']}
                for label,command in commands.items():
                    with(output/(label+'.log')).open('a')as log:
                        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    monitors[label]=child.pid
                save(phase='focus_kernel_submitted',monitors=monitors)
                report=dict(schema='continuous-focus-training-status-v1',kernel=state['kernel'],status='submitted_pending_cuda_proof',
                    progress_snapshot={},pilot_target_steps=819200,conditional_full_steps=8192000,quality_accepted=False,live_stopped=True)
                Path(args.status_report).write_text(json.dumps(report,indent=2)+'\n')
                status_path=Path('reports/defense_plan/realism_status.json');status=json.loads(status_path.read_text());status['stage']='continuous_focus_cohort'
                status['continuous_training_report']=str(args.status_report);status_path.write_text(json.dumps(status,indent=2)+'\n')
                subprocess.run([project_python(),'-m','scripts.defense.update_done'],check=True)
            remote=api.kernels_status(state['kernel']);state_remote=json.loads(str(remote))
            save(phase='focus_kernel_'+state_remote['status'].lower(),remote_status=state_remote)
            if state_remote['status']in('ERROR','CANCEL_ACKNOWLEDGED','CANCELLED'):
                save(phase='focus_remote_failed_outputs_preserved');return
        except Exception as exc:
            save(observation_error=type(exc).__name__,http_status=http_status(exc))
            continue
        for receipt in sorted(output.glob('collected*/collection_receipt.json')):
            try:complete=json.loads(receipt.read_text()).get('complete')
            except(ValueError,OSError):continue
            if not complete:continue
            subprocess.run([project_python(),'-m','scripts.attack.verify_continuous_focus',
                '--collected',str(receipt.parent),'--prepared',str(output),'--output',str(args.verified_report)],check=True)
            verified=json.loads(Path(args.verified_report).read_text())
            status=json.loads(Path(args.status_report).read_text());status['status']='cohort_verified'
            status['verified_report']=str(args.verified_report);Path(args.status_report).write_text(json.dumps(status,indent=2)+'\n')
            subprocess.run([project_python(),'-m','scripts.defense.update_done'],check=True)
            save(phase='focus_cohort_verified',full_gate_met=verified['full_gate_met']);return
        time.sleep(30)
    save(phase='handoff_deadline_reached_remote_job_untouched')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True,type=Path);p.add_argument('--state',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--training-report',default='reports/defense_plan/continuous_slots_verified_20261026.json',type=Path)
    p.add_argument('--evaluation-report',default='reports/defense_plan/continuous_control_pool_verified_20261026.json',type=Path)
    p.add_argument('--owner',default='vertel0');p.add_argument('--token-index',type=int,default=1)
    p.add_argument('--status-report',default='reports/defense_plan/continuous_focus_training_20261026.json',type=Path)
    p.add_argument('--verified-report',default='reports/defense_plan/continuous_focus_verified_20261026.json',type=Path)
    run(p.parse_args())
