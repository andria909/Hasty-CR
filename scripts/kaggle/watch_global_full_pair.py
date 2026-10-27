"""Local progress/terminal verification for an isolated full pair; no API calls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.attack.verify_global_full_pair import verify


def watch(root, report, verified_report):
    root, report, verified_report=Path(root),Path(report),Path(verified_report)
    next_sync=0;deadline=time.monotonic()+172800
    while time.monotonic()<deadline:
        state=json.loads(report.read_text())
        if time.monotonic()>=next_sync:
            source=root/'live_stream/progress.json'
            if source.exists():
                try:snapshot=json.loads(source.read_text())
                except (ValueError,OSError):snapshot=None
                if snapshot and snapshot.get('kernel')==state['kernel']:
                    state['progress_snapshot']=snapshot
                    if snapshot.get('trials'):
                        state.update(status='full_training_steps_observed',actual_training_steps_confirmed=True)
            state['last_local_sync_ns']=time.time_ns();report.write_text(json.dumps(state,indent=2)+'\n')
            subprocess.run([sys.executable,'-m','scripts.defense.update_done'],check=True)
            next_sync=time.monotonic()+1200
        receipt=root/'collected/collection_receipt.json'
        if receipt.exists():
            try:collected=json.loads(receipt.read_text())
            except (ValueError,OSError):collected=None
            if collected and collected.get('complete'):
                result=verify(root/'collected',root)
                with verified_report.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
                state.update(status='full_complete_budgets_verified',verified_report=str(verified_report),
                    ready_for_independent_evaluation=result['ready_for_independent_evaluation'])
                report.write_text(json.dumps(state,indent=2)+'\n')
                subprocess.run([sys.executable,'-m','scripts.defense.update_done'],check=True)
                print('FULL_PAIR_BUDGETS_VERIFIED',flush=True);return
        status=root/'collected.status.json'
        if status.exists():
            try:remote=json.loads(status.read_text())
            except (ValueError,OSError):remote={}
            if remote.get('kernel')==state['kernel'] and remote.get('remote_status')in('ERROR','CANCELLED','CANCELED'):
                state.update(status='remote_terminal_error_requires_inspection',remote_status=remote['remote_status'])
                report.write_text(json.dumps(state,indent=2)+'\n')
                subprocess.run([sys.executable,'-m','scripts.defense.update_done'],check=True)
                return
        time.sleep(30)
    state=json.loads(report.read_text());state['status']='local_wait_expired_remote_job_unchanged'
    report.write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--report',required=True);parser.add_argument('--verified-report',required=True)
    args=parser.parse_args();watch(args.root,args.report,args.verified_report)
