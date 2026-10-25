"""Record live Kaggle training progress without exposing authentication data."""
import argparse
import json
from pathlib import Path
import re
import time
import sys



def record_trial_event(progress,line,remote_time):
    try:row=json.loads(line)
    except (ValueError,TypeError):return False
    if not isinstance(row,dict):return False
    event=row.get('event')
    if event not in ('TRIAL_PROGRESS','FULL_TRAINING_PROGRESS'):return False
    step,update=row.get('step'),row.get('update')
    target=102400 if event=='TRIAL_PROGRESS' else 2048000
    identity=row.get('trial_id') if event=='TRIAL_PROGRESS' else 'full_'+str(row.get('mode'))+'_'+str(row.get('profile'))
    if not isinstance(identity,str) or not identity or type(step)is not int or type(update)is not int or not 0<=step<=target or update<0:return False
    trials=progress.setdefault('trials',{});previous=trials.get(identity)
    if previous and (update<=previous['update'] or step<previous['steps']):return False
    trials[identity]=dict(update=update,steps=step,target_steps=target,percent=100*step/target,remote_log_seconds=remote_time)
    return True


def reconnect_delay(interval,started):
    return max(0,interval-int(time.monotonic()-started))


def record_evaluation_event(progress,line,remote_time):
    """Track completed games only; this percentage is not quality acceptance."""
    try:row=json.loads(line)
    except (ValueError,TypeError):return False
    if not isinstance(row,dict):return False
    event=row.get('event')
    if event=='CUDA_READY'and row.get('role')in ('candidate','parent'):
        evaluation=progress.setdefault('evaluation',{})
        if evaluation.get('role')not in (None,row['role']):return False
        evaluation.update(role=row['role'],checkpoint_sha256=row.get('checkpoint_sha256'))
        return True
    if event=='heldout_game_progress':
        completed,total=row.get('completed'),row.get('total')
        if (type(completed)is not int or type(total)is not int or total<=0 or not 0<=completed<=total):return False
        evaluation=progress.setdefault('evaluation',{})
        previous=evaluation.get('games',{})
        if previous and (total!=previous['total']or completed<=previous['completed']):return False
        evaluation['games']=dict(completed=completed,total=total,percent=100*completed/total,
            remote_log_seconds=remote_time,quality_verified=False)
        return True
    if event=='heldout_part_complete'and row.get('part')in ('fixed_regression','fresh_seeds','king_finish','strategy_test','duels'):
        evaluation=progress.setdefault('evaluation',{})
        parts=evaluation.setdefault('completed_parts',[])
        if row['part']in parts:return False
        parts.append(row['part'])
        return True
    return False


def replay_evaluation_progress(path):
    """Read saved SSE events without reconnecting or touching the remote job."""
    progress={'source':'saved_kaggle_live_log_stream'}
    for line in Path(path).read_text().splitlines():
        event=json.loads(line)
        record_evaluation_event(progress,event.get('data',''),event.get('time'))
    return progress


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reconnects',type=int,default=20)
    parser.add_argument('--reconnect-seconds',type=int,default=1200)
    parser.add_argument('--api-user',help='Known owner; bearer token remains in environment')
    args=parser.parse_args()
    if args.reconnects<0:parser.error('reconnects must be nonnegative')
    if args.reconnect_seconds<1:parser.error('reconnect-seconds must be positive')
    if not __package__:sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from scripts.kaggle.collect_final_evaluation import configure_http_timeout,make_api_client
    configure_http_timeout()
    args.output.mkdir(parents=True,exist_ok=False)
    api=make_api_client(args.api_user)
    progress={'kernel':args.kernel,'source':'kaggle_live_log_stream','trainers':{}}
    with (args.output/'events.jsonl').open('x') as stream:
        for attempt in range(args.reconnects+1):
            connection_started=time.monotonic()
            try:
                progress['stream_state']='connected'
                progress.pop('error_type',None);progress.pop('http_status',None)
                for event in api.kernels_logs_stream(args.kernel):
                    stream.write(json.dumps(event)+'\n');stream.flush()
                    line=event.get('data','')
                    if (record_trial_event(progress,line,event.get('time'))or
                            record_evaluation_event(progress,line,event.get('time'))):
                        progress['checked_host_ns']=time.time_ns()
                        (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                    for name in ('ppo','grpo'):
                        match=re.search(r'\b'+name+r':\s*\{[^\n]*?"update":\s*(\d+)',line)
                        if not match:continue
                        update=int(match.group(1))
                        step=re.search(r'"(?:step|steps|environment_steps)":\s*(\d+)',line)
                        if not step:continue
                        previous=progress['trainers'].get(name,{}).get('update',-1)
                        if update<=previous:continue
                        progress['trainers'][name]={'update':update,'steps':int(step.group(1))if step else None,
                                                   'remote_log_seconds':event.get('time')}
                        progress['checked_host_ns']=time.time_ns()
                        (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                        if update==1 or update%25==0:
                            print(json.dumps(progress),flush=True)
                progress['stream_state']='ended'
                progress['checked_host_ns']=time.time_ns()
                (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                if attempt>=args.reconnects:break
                # A closed log connection does not prove the GPU job ended.
                time.sleep(reconnect_delay(args.reconnect_seconds,connection_started))
            except Exception as error:
                progress['stream_state']='interrupted'
                progress['error_type']=type(error).__name__
                response=getattr(error,'response',None)
                progress['http_status']=getattr(response,'status_code',None)
                progress['reconnect_attempt']=attempt
                progress['checked_host_ns']=time.time_ns()
                (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                print(json.dumps(progress),flush=True)
                transient=(type(error).__name__ in ('SSLError','ConnectionError','ReadTimeout','Timeout')
                           or progress['http_status'] in (404,429,500,502,503,504))
                if not transient or attempt>=args.reconnects:
                    break
                time.sleep(reconnect_delay(args.reconnect_seconds,connection_started))
        progress['checked_host_ns']=time.time_ns()
        (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(progress),flush=True)

if __name__=='__main__':main()
