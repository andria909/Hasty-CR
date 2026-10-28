"""Local-only progress and terminal verification for a frozen focus-style study."""
import argparse
import json
from pathlib import Path
import time
import subprocess
import sys

from scripts.attack.verify_continuous_focus import verify


def watch(root,status_report,verified_report,summary_report=None,markdown_report=None):
    root=Path(root);status_report=Path(status_report);deadline=time.monotonic()+86400;next_sync=0
    while time.monotonic()<deadline:
        if time.monotonic()>=next_sync:
            state=json.loads(status_report.read_text())
            source=root/'live_stream/progress.json'
            if source.exists():
                try:r=json.loads(source.read_text())
                except (ValueError,OSError):r=None
                if r and r.get('kernel')==state['kernel']:
                    state['progress_snapshot']=r
                    if r.get('trials'):
                        state['status']='training_steps_observed';state['actual_training_steps_confirmed']=True
            state['last_local_sync_ns']=time.time_ns()
            status_report.write_text(json.dumps(state,indent=2)+'\n')
            subprocess.run([sys.executable,'-m','scripts.defense.update_done'],check=True)
            next_sync=time.monotonic()+1200
        for receipt in root.glob('collected*/collection_receipt.json'):
            try:r=json.loads(receipt.read_text())
            except (ValueError,OSError):continue
            if not r.get('complete'):continue
            prep=json.loads((root/'preparation.json').read_text())
            if prep.get('study_variant')=='late_air_sequential_teacher_v1':
                from scripts.attack.verify_sequence_pilot import verify_sequence
                report=verify_sequence(receipt.parent,root)
            elif prep.get('study_variant') in ('own_pending_inputs_v1','pending_sequence_retention_v1','pending_diversity_slots_v1'):
                from scripts.attack.verify_frozen_pending import verify as verify_pending
                report=verify_pending(receipt.parent,root)
            elif prep.get('study_variant')=='native_fireball_slot_v1':
                from scripts.attack.verify_native_fireball_slot import verify_native_fireball
                report=verify_native_fireball(receipt.parent,root)
            elif prep.get('study_variant')=='tower_support_reward_v1':
                from scripts.attack.verify_tower_support_pilot import verify_tower_support
                report=verify_tower_support(receipt.parent,root)
            elif prep.get('study_variant')=='spell_waste_reward_v1':
                from scripts.attack.verify_spell_waste_pilot import verify_spell_waste
                report=verify_spell_waste(receipt.parent,root)
            else:
                report=verify(receipt.parent,root)
            Path(verified_report).write_text(json.dumps(report,indent=2)+'\n')
            if summary_report or markdown_report:
                if receipt.parent!=root/'collected':raise ValueError('Summary requires canonical collected directory')
                from scripts.attack.summarize_clock_pilots import summarize,render
                summary=summarize(root,verified=report)
                if summary_report:Path(summary_report).write_text(json.dumps(summary,indent=2)+'\n')
                if markdown_report:Path(markdown_report).write_text(render(summary))
            state=json.loads(status_report.read_text());state.update(status='pilot_complete_budgets_verified',verified_report=str(verified_report))
            status_report.write_text(json.dumps(state,indent=2)+'\n')
            subprocess.run([sys.executable,'-m','scripts.defense.update_done'],check=True)
            print('PILOT_BUDGETS_VERIFIED',flush=True);return
        time.sleep(30)
    raise TimeoutError('Local focus watcher expired; remote training unchanged')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--status-report',required=True)
    p.add_argument('--verified-report',required=True);p.add_argument('--summary-report');p.add_argument('--markdown-report')
    a=p.parse_args();watch(a.root,a.status_report,a.verified_report,a.summary_report,a.markdown_report)
