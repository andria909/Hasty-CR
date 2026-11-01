"""Evaluate every mixed fitting candidate after independently verified collection.

No Kaggle calls, emulator actions, training or isolated-policy promotion.
The 800/8000 comparison is fixed before observing its label scores.
"""
import argparse
import json
from pathlib import Path
import time

from scripts.kaggle.local_artifact_transfer import file_sha
from scripts.kaggle.watch_bank_correction_pair import read_if_complete
from scripts.attack.validate_bank_full_games import run as full_validation
from scripts.attack.summarize_correction_full_games import summarize
from scripts.attack.summarize_match_decisions import summarize as behavior


def weights_for_comparison(root, proof):
    expected={f'{mode}{budget}_s{seed}'for mode in ('mixed','isolated')for budget in (800,8000)for seed in (402,403)}
    if (proof.get('schema')!='verified-bank-fit-depth-v1'or proof.get('actual_cuda_verified')is not True
            or proof.get('supervised_updates_verified')!=35200 or len(proof.get('runs',[]))!=8
            or {r['id']for r in proof['runs']}!=expected):
        raise ValueError('All eight independently verified CUDA runs required')
    parent=Path(root)/'dataset/parent.pt'
    if file_sha(parent)!=proof['initial_sha256']:raise ValueError('Initial policy changed')
    candidates=[r for r in proof['runs']if r['case']['dataset']=='control']
    if {r['id']for r in candidates}!={f'mixed{b}_s{s}'for b in (800,8000)for s in (402,403)}:
        raise ValueError('Mixed candidate identities differ')
    for r in candidates:
        if file_sha(r['checkpoint'])!=r['checkpoint_sha256']:raise ValueError('Candidate changed')
    return [str(parent)]+[r['checkpoint']for r in candidates]


def watch(root,verified,status,config,summary,markdown,behavior_json,behavior_md):
    root,status=Path(root),Path(status)
    def save(**fields):
        state=read_if_complete(status)or{}
        state.update(fields,local_updated_ns=time.time_ns(),live_games=False,full_training_accepted=False)
        temp=status.with_suffix('.writing');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(status)
    save(status='waiting_for_verified_weights',root=str(root),verified=str(verified),validation_games_target=360)
    try:
        deadline=time.monotonic()+86400
        while time.monotonic()<deadline:
            proof=read_if_complete(verified)
            if proof is not None:break
            time.sleep(30)
        else:raise TimeoutError('Verified artifacts not received; remote job unchanged')
        weights=weights_for_comparison(root,proof)
        save(status='full_validation_running',verified_sha256=file_sha(verified))
        output=root/'mixed_full_validation'
        full_validation(weights,config,output)
        result=summarize(verified,output/'result.json',summary,markdown)
        behavior(output/'result.json',behavior_json,behavior_md)
        save(status='full_validation_verified',games_verified=result['games_verified'],summary=str(summary),behavior=str(behavior_json))
        from scripts.defense.current_dashboard import render
        Path('done.md').write_text(render(Path('reports/defense_plan')))
    except BaseException as exc:
        save(status='inspection_required',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','verified','status','config','summary','markdown','behavior_json','behavior_md'):
        p.add_argument('--'+name.replace('_','-'),required=True)
    a=p.parse_args();watch(**vars(a))
