"""Launch the precommitted four-candidate evaluation once both full pairs verify.

Reads local observer receipts only while waiting. No live play, model promotion,
training restart, or additional remote polling. Failures preserve every artifact.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.kaggle.local_artifact_transfer import file_sha


def check_config(config):
    version=config.get('schema')
    if version not in ('queued-full-candidate-evaluation-v1','queued-full-candidate-evaluation-v2'):
        raise ValueError('Versioned evaluation queue required')
    if len(config['full_pairs']) not in ((2,)if version.endswith('-v1')else(1,2)):
        raise ValueError('One or two complete full pairs required (v1 requires two)')
    if version.endswith('-v2'):
        if file_sha(config['blueprint'])!=config['blueprint_sha256']:
            raise ValueError('Queued evaluation blueprint changed')
        from scripts.attack.evaluation_execution import execution_for,HOG_CLIPPED
        if execution_for(json.loads(Path(config['blueprint']).read_text()))!=HOG_CLIPPED:
            raise ValueError('New evaluation queue requires current Hog physics')
    if config['owner']!='andriashadrin' or config['token_index']!=0:
        raise ValueError('Use the first full pair account after its completion')
    if file_sha(Path(config['source'])/'manifest.json')!=config['source_manifest_sha256']:
        raise ValueError('Queued evaluation source changed')
    for name,digest in config['code_pins'].items():
        if file_sha(name)!=digest:raise ValueError('Queued evaluation implementation changed; inspect before continuing')
    targets=[Path(config[n]).absolute() for n in ('binding','prepared','preflight_output')]
    if len(set(targets))!=3 or any(a.is_relative_to(b) for a in targets for b in targets if a!=b):
        raise ValueError('Independent nonoverlapping output paths required')


def ready(config):
    for entry in config['full_pairs']:
        root=Path(entry['prepared']); proof=Path(entry['verified_report'])
        receipt=root/'collected/collection_receipt.json'
        if not proof.exists() or not receipt.exists():return False
        try:p=json.loads(proof.read_text());r=json.loads(receipt.read_text())
        except json.JSONDecodeError:return False
        if (p.get('schema')!='verified-global-selected-full-pair-v1' or p.get('full_steps_verified')!=8192000
                or p.get('kernel')!=entry['kernel'] or r.get('kernel')!=entry['kernel']
                or r.get('remote_status')!='COMPLETE' or r.get('complete') is not True):
            raise ValueError('Full-pair proof/terminal receipt differs')
    return True


def execute(config, transition):
    check_config(config)
    if not ready(config):raise ValueError('Both actual full pairs must be complete')
    # Refuse automatic retries over partial staging or an uncertain submission.
    for name in ('binding','prepared','preflight_output'):
        if Path(config[name]).exists():raise ValueError('Existing evaluation artifacts require inspection; no duplicate launch')
    from scripts.attack.bind_full_candidate_evaluation import bind
    from scripts.kaggle.prepare_full_candidate_evaluation import prepare
    from scripts.kaggle.preflight_full_candidate_evaluation import preflight,enable_submission
    from scripts.kaggle.start_full_evaluation_observers import start
    transition('binding_actual_full_weights')
    extra=dict(blueprint_path=config['blueprint'])if config['schema'].endswith('-v2')else{}
    result=bind([e['prepared'] for e in config['full_pairs']],config['binding'],**extra)
    candidates=2*len(config['full_pairs']);budget={2:4200,4:6300}[candidates]
    if result['candidates']!=candidates or result['total_games']!=budget:raise ValueError('Precommitted candidate budget differs')
    transition('packaging')
    prepare(config['binding'],config['source'],config['prepared'],config['owner'],config['slug'])
    transition('exact_package_preflight')
    preflight(config['prepared'],config['preflight_output']);enable_submission(config['prepared'])
    transition('submitting_once')
    sdk='/Users/andriashadrin/.local/share/hastycr-kaggle-cli/bin/python'
    subprocess.run([sdk,'-m','scripts.kaggle.submit_prepared_pilot','--root',config['prepared'],
                    '--token-index',str(config['token_index']),'--evaluation'],check=True)
    transition('starting_observers')
    state=start(config['prepared'],config['binding'],config['report'],config['token_index'])
    transition('evaluation_submitted_observers_running',kernel=state['kernel'],pids=state['pids'])
    return state


def watch(config_path,state_path):
    config_path,state_path=Path(config_path),Path(state_path)
    config=json.loads(config_path.read_text());check_config(config)
    if state_path.exists():raise ValueError('Existing queue state requires inspection; no duplicate worker')
    state=dict(schema='full-candidate-evaluation-queue-status-v1',pid=os.getpid(),config_sha256=file_sha(config_path))
    def transition(stage,**extra):
        state.update(stage=stage,checked_host_ns=time.time_ns(),**extra)
        state_path.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    transition('waiting_for_verified_full_pairs',full_pairs=len(config['full_pairs']))
    deadline=time.monotonic()+172800
    try:
        while time.monotonic()<deadline:
            if file_sha(config_path)!=state['config_sha256']:raise ValueError('Queue configuration changed')
            if ready(config):return execute(config,transition)
            time.sleep(60)
        transition('local_wait_expired_remote_jobs_unchanged')
    except Exception as error:
        transition('requires_inspection_artifacts_preserved',error_type=type(error).__name__)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--state',required=True)
    args=parser.parse_args();watch(args.config,args.state)
