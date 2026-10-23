"""Preserve terminal full-PPO outputs, including incomplete-budget recovery weights."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from scripts.kaggle.collect_final_evaluation import configure_http_timeout
from scripts.kaggle.recover_defense_outputs import TERMINAL,list_outputs,terminal_state


def collect(api,kernel,output,fetch=None):
    state=terminal_state(api.kernels_status(kernel))
    if state not in TERMINAL:raise ValueError('Full training is not confirmed terminal')
    files,logs=list_outputs(api,kernel)
    if terminal_state(api.kernels_status(kernel))!=state:raise ValueError('Remote state changed during listing')
    prefix='selected_full/';planned={'result.json','train.log','validation.log','validation.json','train/config.json','train/metrics.jsonl','train/training_result.json'}
    candidates=[key[len(prefix):] for key in files if re.fullmatch(r'selected_full/train/update_\d{6}\.pt',key)]
    if candidates:planned.add(max(candidates))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    receipt=dict(kernel=kernel,status=state,complete=False,files={},remote_job_changed=False,quality_accepted=False)
    if fetch is None:
        import requests
        def fetch(url,path):
            with requests.get(url,stream=True,timeout=(30,300)) as response:
                response.raise_for_status()
                with path.open('xb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        if chunk:stream.write(chunk)
    try:
        for name in sorted(planned):
            key=prefix+name
            if key not in files:continue
            path=output/name;path.parent.mkdir(parents=True,exist_ok=True);fetch(files[key].url,path)
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            receipt['files'][name]=dict(sha256=digest,bytes=path.stat().st_size)
        (output/'kernel_log_receipt.json').write_text(json.dumps(logs)+'\n')
        if 'result.json' in receipt['files']:
            result=json.loads((output/'result.json').read_text());receipt['training_status']=result.get('status')
            for name in ('checkpoint','validation'):
                digest=result.get(name+'_sha256')
                if digest is None:continue
                path=result.get('checkpoint') if name=='checkpoint' else 'validation.json'
                if path not in planned or receipt['files'].get(path,{}).get('sha256')!=digest:raise ValueError('Declared full-training artifact missing or changed')
            if result.get('status')=='completed':
                if result.get('step')!=2048000 or result.get('update')!=2000:raise ValueError('Completed result has incomplete step budget')
                if result.get('test_used_for_selection') is not False:raise ValueError('Test leakage in full-training result')
                if not all(k in result for k in ('checkpoint_sha256','validation_sha256')):raise ValueError('Completed result missing output pins')
                validation=json.loads((output/'validation.json').read_text())
                if validation.get('training_steps')!=2048000 or validation.get('checkpoint_sha256')!=result['checkpoint_sha256']:
                    raise ValueError('Validation uses different checkpoint or budget')
                receipt['complete']=state=='COMPLETE'
        if state=='COMPLETE' and receipt.get('training_status')=='completed' and not receipt['complete']:
            raise ValueError('Unable to verify declared completed training')
    finally:
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--poll-seconds',type=int,default=1200);p.add_argument('--once',action='store_true')
    args=p.parse_args()
    if args.poll_seconds<1:p.error('Positive observation interval required')
    configure_http_timeout()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    while True:
        try:state=terminal_state(api.kernels_status(args.kernel))
        except Exception as error:
            print(json.dumps(dict(observation_error=type(error).__name__)),flush=True)
            if args.once:raise
            time.sleep(args.poll_seconds);continue
        args.output.parent.mkdir(parents=True,exist_ok=True)
        record=dict(kernel=args.kernel,status=state,checked_host_ns=time.time_ns(),remote_job_changed=False)
        args.output.with_suffix('.status.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
        if state in TERMINAL:
            print(json.dumps(collect(api,args.kernel,args.output)),flush=True);break
        if args.once:break
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
