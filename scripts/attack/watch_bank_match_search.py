"""Finish verification, replay and dataset assembly after an existing CPU search."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.attack.build_match_correction_dataset import assemble,replayed_example
from scripts.attack.replay_bank_match_context import replay
from scripts.attack.verify_bank_match_search import verify
from scripts.defense.current_dashboard import render


def watch(state_file,base,verified,replays,dataset,reuse_replay=None):
    state_file=Path(state_file);state=json.loads(state_file.read_text())
    replays=Path(replays)
    if state.get('postprocess_pid'):
        check=subprocess.run(['ps','-p',str(state['postprocess_pid']),'-o','command='],capture_output=True,text=True)
        if 'scripts.attack.watch_bank_match_search'in check.stdout:raise RuntimeError('Observer already running')
    def save(phase,**extra):
        state.update(postprocess_pid=os.getpid(),postprocess_phase=phase,postprocess_updated_ns=time.time_ns(),**extra)
        temp=state_file.with_name(state_file.name+'.writing')
        temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_file)
        print(json.dumps(dict(phase=phase,**extra)),flush=True)
    try:
        save('waiting_for_search',verified_file=str(verified),replays=str(replays),dataset=str(dataset))
        while not all((Path(root)/'result.json').is_file()for root in state['shards']):
            process=subprocess.run(['ps','-p',str(state['pid']),'-o','command='],capture_output=True,text=True)
            if ('scripts.attack.search_bank_match_actions'not in process.stdout or state['root']not in process.stdout):
                raise RuntimeError('Existing search process is absent and complete results are missing')
            time.sleep(30)
        save('verifying_search')
        proof=verify(state['shards'],verified)
        replays.mkdir(parents=True,exist_ok=False)
        if reuse_replay:
            old=json.loads(Path(reuse_replay).read_text())
            matching=next((r for r in proof['accepted']if r['index']==old['index']),None)
            if matching:
                replayed_example(proof['root'],matching,reuse_replay)
                shutil.copyfile(reuse_replay,replays/f"context_{old['index']:04d}.json")
        save('replaying_improvements',accepted_contexts=len(proof['accepted']),replayed_contexts=0)
        for count,row in enumerate(proof['accepted'],1):
            receipt=replays/f"context_{row['index']:04d}.json"
            if not receipt.exists():replay(row['context_path'],receipt)
            replayed_example(proof['root'],row,receipt)
            save('replaying_improvements',replayed_contexts=count)
        save('assembling_dataset')
        result=assemble(base,verified,replays,dataset)
        save('complete',status='COMPLETE; search, replay and mixed dataset verified',
            dataset_examples=result['examples'],correction_examples=result['correction_examples'],
            dataset_sha256=result['data_sha256'],full_training_accepted=False)
        Path('done.md').write_text(render(Path('reports/defense_plan')))
    except BaseException as exc:
        save('error',postprocess_error=type(exc).__name__+': '+str(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('state-file','base','verified','replays','dataset'):p.add_argument('--'+key,required=True)
    p.add_argument('--reuse-replay');a=p.parse_args()
    watch(a.state_file,a.base,a.verified,a.replays,a.dataset,a.reuse_replay)
