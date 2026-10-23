"""Collect a pinned coefficient study only after Kaggle reports terminal state."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from scripts.kaggle.collect_final_evaluation import configure_http_timeout
from scripts.kaggle.recover_defense_outputs import TERMINAL, list_outputs, terminal_state


def trial_ids(config):
    profiles=json.loads(Path('data/ppo_tuning/protocol.json').read_text())['profiles']
    if isinstance(profiles,dict):profiles=list(profiles)
    result=[]
    for worker in config['workers']:
        for row in worker:
            mode,profile,seed=row['mode'],row['profile'],row.get('seed',42)
            if mode not in ('defense','joint') or profile not in profiles or not isinstance(seed,int):
                raise ValueError('Unexpected study trial identity')
            result.append(f'{mode}_{profile}_s{seed}')
    if not result or len(result)!=len(set(result)):raise ValueError('Duplicate or empty trial schedule')
    return result


def collect(api,kernel,output,config,fetch=None):
    state=terminal_state(api.kernels_status(kernel))
    if state not in TERMINAL:raise ValueError('Study is not confirmed terminal')
    files,logs=list_outputs(api,kernel)
    if terminal_state(api.kernels_status(kernel))!=state:raise ValueError('Study state changed during listing')
    ids=trial_ids(config);planned={'study_result.json'}
    for name in ids:
        planned.update(name+'/'+suffix for suffix in ('trial_result.json','validation.json','train.log','validation.log','train/metrics.jsonl','train/config.json'))
        # Preserve any completed checkpoint, including an interrupted trial's last update.
        for key in files:
            prefix='coefficient_study/'+name+'/train/'
            if key.startswith(prefix) and re.fullmatch(r'update_\d{6}\.pt',key[len(prefix):]):planned.add(key[len('coefficient_study/'):])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    receipt=dict(kernel=kernel,status=state,complete=False,files={},trial_ids=ids,remote_job_changed=False,quality_accepted=False)
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
            key='coefficient_study/'+name
            if key not in files:continue
            path=output/name;path.parent.mkdir(parents=True,exist_ok=True);fetch(files[key].url,path)
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            receipt['files'][name]={'sha256':digest,'bytes':path.stat().st_size}
        (output/'kernel_log_receipt.json').write_text(json.dumps(logs)+'\n')
        report=json.loads((output/'study_result.json').read_text()) if (output/'study_result.json').exists() else None
        if report and report.get('status')=='completed':
            rows=report['trials']
            if len(rows)!=len(ids) or {r['trial_id'] for r in rows}!=set(ids):raise ValueError('Study result schedule mismatch')
            for row in rows:
                name=row['trial_id'];validation=name+'/validation.json';checkpoint=name+'/train/update_000100.pt'
                if row['status']!='completed' or row['training_steps']!=102400:raise ValueError('Incomplete trial budget')
                if row['checkpoint']!=checkpoint or row['validation']!=validation:raise ValueError('Unexpected output paths')
                for path,digest in ((checkpoint,row['checkpoint_sha256']),(validation,row['validation_sha256'])):
                    if receipt['files'].get(path,{}).get('sha256')!=digest:raise ValueError('Trial artifact SHA mismatch')
                saved=json.loads((output/name/'trial_result.json').read_text())
                if saved!=row:raise ValueError('Trial result differs from study manifest')
            receipt['complete']=state=='COMPLETE'
        if state=='COMPLETE' and not receipt['complete']:raise ValueError('Terminal study contains incomplete or failed trials')
    finally:
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel',required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--poll-seconds',type=int,default=1200);p.add_argument('--once',action='store_true')
    args=p.parse_args()
    if args.poll_seconds<1:p.error('Positive observation interval required')
    config=json.loads(args.config.read_text());trial_ids(config);configure_http_timeout()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    while True:
        try:state=terminal_state(api.kernels_status(args.kernel))
        except Exception as error:
            print(json.dumps({'observation_error':type(error).__name__}),flush=True)
            if args.once:raise
            time.sleep(args.poll_seconds);continue
        record=dict(kernel=args.kernel,status=state,checked_host_ns=time.time_ns(),remote_job_changed=False)
        args.output.with_suffix('.status.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
        if state in TERMINAL:
            print(json.dumps(collect(api,args.kernel,args.output,config)),flush=True);break
        if args.once:break
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
