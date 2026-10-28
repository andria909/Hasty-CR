"""Submit a verified recovery once an observed same-account GPU job terminates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.kaggle.validate_evaluation_submission import validate_submission
from scripts.kaggle.start_full_evaluation_observers import start


def terminal_slot(status,kernel):
    return status.get('kernel')==kernel and status.get('remote_status')in ('COMPLETE','ERROR','CANCEL_ACK','CANCELED')


def watch(root,wait_for,binding,report,token_index):
    root,wait_for,binding,report=map(Path,(root,wait_for,binding,report))
    prep=json.loads((root/'preparation.json').read_text());validate_submission(root,prep)
    waited=json.loads((wait_for/'preparation.json').read_text())
    if waited['config']['owner']!=prep['config']['owner']:raise ValueError('GPU slot must belong to same account')
    if prep['config']['owner']!=('andriashadrin','vertel0','andreiperelaiko')[token_index]:raise ValueError('Wrong account')
    if (root/'kernel_push_requested.json').exists():raise ValueError('Existing submission requires inspection')
    path=root/'recovery_queue.json'
    state=dict(pid=os.getpid(),status='waiting_for_gpu_slot',wait_kernel=waited['kernel'],poll_seconds=1200)
    with path.open('x')as stream:json.dump(state,stream,indent=2)
    while True:
        status_path=wait_for/'collected.status.json'
        status=json.loads(status_path.read_text())if status_path.exists()else{}
        if terminal_slot(status,waited['kernel']):break
        state.update(last_local_check_ns=time.time_ns(),observed_remote_status=status.get('remote_status'))
        path.write_text(json.dumps(state,indent=2)+'\n');time.sleep(1200)
    state['status']='submitting_once';path.write_text(json.dumps(state,indent=2)+'\n')
    sdk='/Users/andriashadrin/.local/share/hastycr-kaggle-cli/bin/python'
    try:
        subprocess.run([sdk,'-m','scripts.kaggle.submit_prepared_pilot','--root',str(root),'--token-index',str(token_index),'--evaluation'],check=True)
        state['monitoring']=start(root,binding,report,token_index);state['status']='submitted_observers_started'
    except Exception as exc:
        state.update(status='submission_failed_inspect_no_retry',error_type=type(exc).__name__);raise
    finally:path.write_text(json.dumps(state,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('root','wait-for','binding','report'):p.add_argument('--'+arg,required=True)
    p.add_argument('--token-index',type=int,required=True);a=p.parse_args()
    watch(a.root,a.wait_for,a.binding,a.report,a.token_index)
