"""Terminal-only bounded round-robin collector; observes every twenty minutes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.kaggle.recover_defense_outputs import TERMINAL,list_outputs,terminal_state
from scripts.kaggle.collect_final_evaluation import configure_http_timeout

NAMES=('ppo1','grpo2','joint_ppo3','joint_grpo4','v25')
REQUIRED={'result.json',*[n+'_common_skill_test.json'for n in NAMES]}


def collect(api,kernel,output):
    state=terminal_state(api.kernels_status(kernel))
    if state not in TERMINAL:raise ValueError('Round robin is not terminal')
    files,logs=list_outputs(api,kernel)
    if terminal_state(api.kernels_status(kernel))!=state:raise ValueError('Remote state changed during listing')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    import requests
    receipt=dict(kernel=kernel,status=state,files={},complete=False,remote_job_changed=False,
                 remote_hash_note='No independent remote SHA; verify internal report consistency separately')
    try:
        for name in sorted(REQUIRED|{'worker_0.jsonl','worker_1.jsonl'}):
            key='round_robin/'+name
            if key not in files:continue
            response=requests.get(files[key].url,timeout=(30,300));response.raise_for_status();raw=response.content
            (output/name).write_bytes(raw);receipt['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        (output/'kernel_log_receipt.json').write_text(json.dumps(logs)+'\n')
        receipt['missing_reports']=sorted(REQUIRED-set(receipt['files']))
        receipt['complete']=not receipt['missing_reports']
        if state=='COMPLETE'and not receipt['complete']:raise ValueError('Completed job is missing required round-robin/skill reports')
    finally:
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--poll-seconds',type=int,default=1200);p.add_argument('--once',action='store_true')
    a=p.parse_args()
    if a.poll_seconds<1:p.error('poll-seconds must be positive')
    configure_http_timeout()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    while True:
        try:state=terminal_state(api.kernels_status(a.kernel))
        except Exception as error:
            print(json.dumps(dict(observation_error=type(error).__name__,remote_job_changed=False)),flush=True)
            if a.once:raise
            time.sleep(a.poll_seconds);continue
        status=dict(kernel=a.kernel,status=state,checked_host_ns=time.time_ns(),remote_job_changed=False)
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.with_suffix('.status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
        if state in TERMINAL:
            print(json.dumps(collect(api,a.kernel,a.output)),flush=True);break
        if a.once:break
        time.sleep(a.poll_seconds)


if __name__=='__main__':main()
