"""Wait for collected paired CUDA outputs, then run full simulator validation."""
import argparse
import json
from pathlib import Path
import time

from scripts.attack.verify_bank_correction_pair import verify
from scripts.attack.validate_bank_full_games import run as full_validation
from scripts.attack.summarize_correction_full_games import summarize
from scripts.defense.current_dashboard import render


def read_if_complete(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return None


def watch(root,status,verified,summary,markdown,config):
    root,status=Path(root),Path(status);next_sync=0;deadline=time.monotonic()+86400
    def save(**extra):
        state=json.loads(status.read_text());state.update(**extra,local_updated_ns=time.time_ns())
        temporary=status.with_name(status.name+'.writing');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(status)
        Path('done.md').write_text(render(Path('reports/defense_plan')))
    try:
        while time.monotonic()<deadline:
            if time.monotonic()>=next_sync:
                stream=root/'live_stream/progress.json';state=json.loads(status.read_text())
                progress=read_if_complete(stream)
                if progress and progress.get('kernel')==state['kernel']:save(progress_snapshot=progress)
                next_sync=time.monotonic()+1200
            receipt=root/'collected/collection_receipt.json'
            collected=read_if_complete(receipt)
            if collected and collected.get('complete'):
                save(status='verifying_cuda_outputs')
                proof=verify(root);Path(verified).write_text(json.dumps(proof,indent=2)+'\n')
                save(status='full_validation_running',verified_report=str(verified),validation_games_target=648)
                weights=[str(root/'dataset/parent.pt')]+[r['checkpoint']for r in proof['runs']]
                full_validation(weights,config,root/'full_validation')
                result=summarize(verified,root/'full_validation/result.json',summary,markdown)
                save(status='full_validation_verified',full_validation_games_verified=result['games_verified'],
                    summary_report=str(summary),full_training_accepted=False)
                return
            time.sleep(30)
        raise TimeoutError('Local paired observer expired; remote job unchanged')
    except BaseException as exc:
        save(status='inspection_required',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','status','verified','summary','markdown','config'):p.add_argument('--'+key,required=True)
    a=p.parse_args();watch(a.root,a.status,a.verified,a.summary,a.markdown,a.config)
