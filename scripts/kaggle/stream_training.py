"""Record live Kaggle training progress without exposing authentication data."""
import argparse
import json
from pathlib import Path
import re
import time
from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reconnects',type=int,default=20)
    args=parser.parse_args()
    if args.reconnects<0:parser.error('reconnects must be nonnegative')
    args.output.mkdir(parents=True,exist_ok=False)
    api=KaggleApi();api.authenticate()
    progress={'kernel':args.kernel,'source':'kaggle_live_log_stream','trainers':{}}
    with (args.output/'events.jsonl').open('x') as stream:
        for attempt in range(args.reconnects+1):
            try:
                progress['stream_state']='connected'
                progress.pop('error_type',None);progress.pop('http_status',None)
                for event in api.kernels_logs_stream(args.kernel):
                    stream.write(json.dumps(event)+'\n');stream.flush()
                    line=event.get('data','')
                    for name in ('ppo','grpo'):
                        match=re.search(r'\b'+name+r':\s*\{[^\n]*?"update":\s*(\d+)',line)
                        if not match:continue
                        update=int(match.group(1))
                        step=re.search(r'"(?:step|environment_steps)":\s*(\d+)',line)
                        previous=progress['trainers'].get(name,{}).get('update',-1)
                        if update<=previous:continue
                        progress['trainers'][name]={'update':update,'steps':int(step.group(1))if step else None,
                                                   'remote_log_seconds':event.get('time')}
                        progress['checked_host_ns']=time.time_ns()
                        (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                        if update==1 or update%25==0:
                            print(json.dumps(progress),flush=True)
                progress['stream_state']='ended'
                break
            except Exception as error:
                progress['stream_state']='interrupted'
                progress['error_type']=type(error).__name__
                response=getattr(error,'response',None)
                progress['http_status']=getattr(response,'status_code',None)
                progress['reconnect_attempt']=attempt
                progress['checked_host_ns']=time.time_ns()
                (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                print(json.dumps(progress),flush=True)
                if type(error).__name__ not in ('SSLError','ConnectionError','ReadTimeout','Timeout') or attempt>=args.reconnects:
                    break
                time.sleep(2)
        progress['checked_host_ns']=time.time_ns()
        (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(progress),flush=True)

if __name__=='__main__':main()
