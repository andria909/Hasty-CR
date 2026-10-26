"""Finite local postprocessor; collector owns remote checks every20minutes."""
import argparse
import json
from pathlib import Path
import time

from scripts.attack.verify_continuous_training import verify


def watch(root,output,seconds=50000,status_report='reports/defense_plan/continuous_training_20261026.json'):
    root=Path(root);until=time.monotonic()+seconds
    while time.monotonic()<until:
        for receipt in sorted(root.glob('collected*/collection_receipt.json')):
            try:complete=json.loads(receipt.read_text()).get('complete')
            except (ValueError,OSError):continue
            if not complete:continue
            report=verify(receipt.parent,root)
            Path(output).write_text(json.dumps(report,indent=2)+'\n')
            status=Path(status_report)
            if status.exists():
                state=json.loads(status.read_text());state['status']='training_complete_budgets_verified'
                state['verified_report']=str(output)
                state['progress_snapshot']={'trials':{r['id']:{'percent':100}for r in report['models']}}
                status.write_text(json.dumps(state,indent=2)+'\n')
                import subprocess,sys
                subprocess.run([sys.executable,'-m','scripts.defense.update_done'],check=True)
            print(json.dumps(dict(event='CONTINUOUS_BUDGETS_VERIFIED',models=len(report['models']),quality_accepted=False)),flush=True)
            return
        time.sleep(30)
    raise TimeoutError('Bounded local postprocessor expired; remote job untouched')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--status-report',default='reports/defense_plan/continuous_training_20261026.json')
    a=p.parse_args();watch(a.root,a.output,status_report=a.status_report)
