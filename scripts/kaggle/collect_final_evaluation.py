"""Collect bounded final-evaluation outputs only after authoritative terminal state."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.kaggle.recover_defense_outputs import TERMINAL,list_outputs,terminal_state


def configure_http_timeout():
    """Bound HTTP in this local observer, without changing SDK files or jobs."""
    import requests
    original_send=requests.Session.send
    if getattr(original_send,'_hasty_bounded',False):return
    def bounded_send(session,request,**kwargs):
        if kwargs.get('timeout')is None:kwargs['timeout']=(30,120)
        return original_send(session,request,**kwargs)
    bounded_send._hasty_bounded=True
    requests.Session.send=bounded_send


def collect(api,kernel,destination,profile='defense'):
    state=terminal_state(api.kernels_status(kernel))
    if state not in TERMINAL:raise ValueError('Evaluation is not terminal')
    files,logs=list_outputs(api,kernel)
    if terminal_state(api.kernels_status(kernel))!=state:
        raise ValueError('Remote evaluation state changed during listing')
    if profile=='strategy':
        prefix='strategy_evaluation/'
        result_file='result.json'
        allowed={result_file,'validation.json','test.json','duels.json','validation.log','test.log','duels.log'}
    elif profile=='train':
        prefix='train_evaluation/'
        result_file='result.json'
        allowed={result_file,'train.json','worker_0.jsonl','worker_1.jsonl','worker_0.log','worker_1.log'}
    elif profile=='fireball-evaluation':
        prefix='fireball_evaluation/'
        result_file='result.json'
        allowed={result_file,'fireball_test.json','regression_test.json','duels.json','duels_raw.jsonl'}
    elif profile=='fireball':
        prefix='fireball_refinement/'
        result_file='result.json'
        allowed={result_file,'candidate.pt','metrics.jsonl','baseline_validation.json','demonstration_audit.json'}
    elif profile=='defense':
        prefix='evaluation/'
        result_file='evaluation_result.json'
        allowed={result_file}
        for algorithm in ('ppo','grpo'):
            allowed.update({algorithm+'_defense.json',algorithm+'_duels.json',algorithm+'_0.log',algorithm+'_1.log'})
    else:raise ValueError('Unknown evaluation profile')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    import requests
    receipt=dict(kernel=kernel,remote_status=state,files={},remote_job_changed=False,complete=False,
        remote_hash_note='No independent metadata SHA; downloaded artifacts checked against internal reports separately')
    try:
        for name in sorted(allowed):
            remote=prefix+name
            if remote not in files:continue
            response=requests.get(files[remote].url,timeout=(30,300));response.raise_for_status()
            raw=response.content;(destination/name).write_bytes(raw)
            receipt['files'][name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
        if result_file not in receipt['files']:raise ValueError('Terminal evaluation result absent')
        (destination/'kernel_log_receipt.json').write_text(json.dumps(logs)+'\n')
        receipt['complete']=True
    finally:
        (destination/'collection_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--poll-seconds',type=int,default=1200)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--profile',choices=('defense','strategy','train','fireball','fireball-evaluation'),default='defense')
    args=parser.parse_args()
    if args.poll_seconds<1:parser.error('poll-seconds must be positive')
    # The installed SDK sends requests without a default timeout. Bound only
    # this observer process; active jobs and the installed package are untouched.
    configure_http_timeout()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    while True:
        try:
            state=terminal_state(api.kernels_status(args.kernel))
        except Exception as error:
            print(json.dumps(dict(observation_error=type(error).__name__,remote_job_changed=False)),flush=True)
            if args.once:raise
            time.sleep(args.poll_seconds)
            continue
        status=dict(kernel=args.kernel,remote_status=state,checked_host_ns=time.time_ns(),remote_job_changed=False)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.with_suffix('.status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
        if state in TERMINAL:
            print(json.dumps(collect(api,args.kernel,args.output,profile=args.profile)),flush=True);break
        if args.once:break
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
