"""Bounded local verification after the600-game collector finishes."""
import argparse
import json
from pathlib import Path
import time
import subprocess
import sys

from scripts.attack.verify_continuous_pool import verify


def watch(root,output,status_report=None):
    root=Path(root);deadline=time.monotonic()+50000;next_sync=0
    while time.monotonic()<deadline:
        if status_report and time.monotonic()>=next_sync:
            path=Path(status_report);state=json.loads(path.read_text());source=root/'live_stream/progress.json'
            if source.exists():
                try:r=json.loads(source.read_text())
                except(ValueError,OSError):r=None
                if r and r.get('kernel')==state['kernel']:state['progress_snapshot']=r
            path.write_text(json.dumps(state,indent=2)+'\n')
            subprocess.run([sys.executable,'-m','scripts.defense.update_done'],check=True)
            next_sync=time.monotonic()+1200
        for receipt in sorted(root.glob('collected*/collection_receipt.json')):
            try:complete=json.loads(receipt.read_text()).get('complete')
            except(ValueError,OSError):continue
            if not complete:continue
            report=verify(receipt.parent,root);Path(output).write_text(json.dumps(report,indent=2)+'\n')
            if status_report:
                path=Path(status_report);state=json.loads(path.read_text());state.update(status='complete_verified',verified_report=str(output))
                if state.get('planned_markdown_report'):
                    from scripts.attack.summarize_continuous_pool import render
                    Path(state['planned_markdown_report']).write_text(render(output,receipt.parent))
                path.write_text(json.dumps(state,indent=2)+'\n')
                subprocess.run([sys.executable,'-m','scripts.defense.update_done'],check=True)
            print(json.dumps(dict(event='POOL600_VERIFIED',standings=report['standings'])),flush=True)
            return
        time.sleep(30)
    raise TimeoutError('Local verification deadline; remote job unchanged')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--status-report')
    a=parser.parse_args();watch(a.root,a.output,a.status_report)
