"""Finish a finite local report when the existing collector saves its receipt.

No API polling, training, notebook restarts or game input. Preserve partial
downloads; accept only a complete collection for the prepared notebook.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=24)
    args=parser.parse_args()
    if not 0<args.hours<=48:parser.error('Bounded reporting horizon required')
    deadline=time.monotonic()+args.hours*3600
    status_path=Path('reports/defense_plan/realism_status.json')
    while time.monotonic()<deadline:
        candidates=[]
        for folder in args.run.glob('collected*'):
            receipt=folder/'collection_receipt.json'
            if not folder.is_dir()or not receipt.is_file():continue
            try:value=json.loads(receipt.read_text())
            except ValueError:continue
            if value.get('complete')is True and value.get('remote_status')=='COMPLETE':candidates.append(folder)
        if candidates:
            from scripts.attack.summarize_card_usage import verify,markdown
            folder=sorted(candidates)[-1]
            try:
                report=verify(folder,args.run,Path('reports/defense_plan/live_card_usage_20261025.json'))
                output=Path('reports/defense_plan/card_usage_comparison_20261025.json')
                page=Path('reports/card_usage_20261025.md')
                output.write_text(json.dumps(report,indent=2)+'\n');page.write_text(markdown(report))
                status=json.loads(status_path.read_text())if status_path.exists()else {}
                status.update(card_usage_status='complete_verified',card_usage_report=str(output),card_usage_markdown=str(page),
                    simulation_models_verified=len(report['simulation_models']),live_games_started=False)
                status_path.write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
                subprocess.run([sys.executable,'-m','scripts.defense.update_done'],check=True)
                print(json.dumps(dict(status='complete_verified',models=len(report['simulation_models']),report=str(output),markdown=str(page))),flush=True)
                return
            except Exception as error:
                (args.run/'postprocess_error.json').write_text(json.dumps(dict(error_type=type(error).__name__,message=str(error)),indent=2)+'\n')
                raise
        time.sleep(30)
    (args.run/'postprocess_timeout.json').write_text(json.dumps(dict(status='observation_horizon_expired',remote_job_changed=False))+'\n')


if __name__=='__main__':main()
