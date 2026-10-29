"""Local verified selection and conditional full continuation of the BC/PPO study."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.attack.watch_global_pilot_selection import ready_studies
from scripts.attack.select_pilot_pool import select
from scripts.kaggle.watch_mana_campaign import continue_full


def watch(config_path,state_file):
    config=json.loads(Path(config_path).read_text());state_file=Path(state_file)
    if state_file.exists():raise FileExistsError('Inspect existing selection process')
    state=dict(status='waiting_for_verified_pilot',full_submitted=False,live_games=False)
    def save():
        state_file.write_text(json.dumps(state,indent=2)+'\n')
        subprocess.run([sys.executable,'-m','scripts.defense.update_done'],check=True)
    save();deadline=time.monotonic()+172800
    try:
        while time.monotonic()<deadline:
            if ready_studies(config):
                ranking=select([config['studies'][0]['prepared']],limit=4)
                with Path(config['ranking_report']).open('x')as out:
                    json.dump(ranking,out,indent=2);out.write('\n')
                state.update(status='ranked',eligible_profiles=ranking['eligible_profiles'],ranking_report=config['ranking_report']);save()
                state.update(continue_full(config,ranking));save();return
            time.sleep(1200)
        state['status']='local_wait_expired';save()
    except Exception as exc:
        state.update(status='inspection_required',error_type=type(exc).__name__,message=str(exc));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--state-file',required=True)
    a=p.parse_args();watch(a.config,a.state_file)
