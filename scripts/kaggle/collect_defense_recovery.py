"""Collect terminal semantic-recovery artifacts; never submits or restarts jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from scripts.kaggle.recover_defense_outputs import TERMINAL, list_outputs, terminal_state


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(api,kernel,output,fetch=None,profile='recovery'):
    layouts={'recovery':('recovery','recovery_result.json','defense-semantic-recovery-v1'),
             'grpo-account2':('account2','account2_result.json','defense-grpo-account2-v1')}
    prefix,result_file,schema=layouts[profile]
    prefix+='/'
    state=terminal_state(api.kernels_status(kernel))
    if state not in TERMINAL:raise ValueError('Recovery kernel is not terminal')
    items,logs=list_outputs(api,kernel)
    if terminal_state(api.kernels_status(kernel))!=state:
        raise ValueError('Remote state changed during listing')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if fetch is None:
        import requests
        def fetch(url,path):
            with requests.get(url,stream=True,timeout=(30,300))as response:
                response.raise_for_status()
                with path.open('xb')as stream:
                    for chunk in response.iter_content(1024*1024):
                        if chunk:stream.write(chunk)
    receipt=dict(kernel=kernel,remote_status=state,complete=False,files={},
                 quality_accepted=False,remote_job_changed=False,
                 remote_hash_note='SDK metadata SHA unavailable; reported selected SHA checks internal consistency only')
    def download(remote,local,expected=None):
        if remote not in items:raise ValueError('Declared recovery artifact absent: '+remote)
        path=output/local;path.parent.mkdir(parents=True,exist_ok=True)
        fetch(items[remote].url,path)
        actual=sha(path)
        receipt['files'][local]=dict(remote_path=remote,sha256=actual,
            reported_sha256=expected,reported_hash_verified=expected is not None and expected==actual)
        if expected is not None and expected!=actual:raise ValueError('Reported artifact SHA mismatch')
    try:
        result_name=prefix+result_file
        result={}
        if result_name in items:
            download(result_name,result_file)
            result=json.loads((output/result_file).read_text())
            if result.get('schema')!=schema:
                raise ValueError('Unexpected recovery result schema')
        receipt['result_status']=result.get('status')
        receipt['result_available']=bool(result)
        planned={}
        artifacts=result.get('selected_artifacts',{})
        if profile=='grpo-account2' and result.get('selected_artifact'):
            artifacts={'grpo':result['selected_artifact']}
        for algorithm,row in artifacts.items():
            if algorithm not in ('ppo','grpo') or row.get('path')!=f'selected/{algorithm}.pt':
                raise ValueError('Unsafe selected recovery artifact path')
            digest=row.get('sha256','')
            if not re.fullmatch('[a-f0-9]{64}',digest):raise ValueError('Missing selected SHA')
            planned[row['path']]=digest
        # Preserve safe resumable endpoints if this run fails before selection.
        patterns=(r'ppo/update_(\d{6})\.pt',r'ppo/best_validation_(\d{6})\.pt',
            r'grpo/boundary_u(\d{6})_s(\d{6})\.pt',r'grpo/best_validation_u(\d{6})_s(\d{6})\.pt',
            r'grpo/final_u(\d{6})_s(\d{6})\.pt')
        for pattern in patterns:
            found=[]
            for remote in items:
                local=remote.removeprefix(prefix)
                match=re.fullmatch(pattern,local)if remote.startswith(prefix)else None
                if match:found.append((tuple(map(int,match.groups())),local))
            if found:planned.setdefault(max(found)[1],None)
        for local in ('ppo/metrics.jsonl','grpo/updates.jsonl','ppo/config.json','grpo/config.json',
                      'ppo.log','grpo.log'):
            if prefix+local in items:planned[local]=None
        for local,expected in sorted(planned.items()):download(prefix+local,local,expected)
        (output/'kernel_log_receipt.json').write_text(json.dumps(logs)+'\n')
        receipt['complete']=True
    finally:
        with (output/'collection_receipt.json').open('x')as stream:
            json.dump(receipt,stream,indent=2);stream.write('\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--watch-seconds',type=int,default=28800)
    parser.add_argument('--profile',choices=('recovery','grpo-account2'),default='recovery')
    args=parser.parse_args()
    if args.watch_seconds<0:parser.error('Negative observation window')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+args.watch_seconds
    while True:
        try:
            state=terminal_state(api.kernels_status(args.kernel))
            args.output.with_suffix('.status.json').write_text(json.dumps(dict(
                kernel=args.kernel,remote_status=state,checked_host_ns=time.time_ns(),remote_job_changed=False))+'\n')
            if state in TERMINAL:break
        except Exception as error:
            print(json.dumps(dict(observation_error=type(error).__name__,remote_job_changed=False)),flush=True)
        if time.monotonic()>=deadline:raise TimeoutError('Observation ended; remote job unchanged')
        time.sleep(30)
    print(json.dumps(collect(api,args.kernel,args.output,profile=args.profile),indent=2),flush=True)


if __name__=='__main__':main()
