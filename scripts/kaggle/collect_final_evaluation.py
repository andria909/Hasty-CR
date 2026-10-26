"""Collect bounded final-evaluation outputs only after authoritative terminal state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.kaggle.recover_defense_outputs import TERMINAL,list_outputs,terminal_state


def make_api_client(api_user=None,api_class=None):
    """Use explicit environment bearer credentials or the normal SDK login.

    The explicit path avoids an optional token-introspection round trip.
    Kaggle still authenticates and authorizes every resource API request.
    """
    if api_class is None:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api_class=KaggleApi
    api=api_class()
    if api_user is None:api.authenticate()
    else:
        token=os.environ.get('KAGGLE_API_TOKEN')
        if not token:raise ValueError('Explicit bearer credentials require KAGGLE_API_TOKEN')
        api.config_values[api.CONFIG_NAME_USER]=api_user
        api.config_values[api.CONFIG_NAME_TOKEN]=token
    return api


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
    prefetched={}
    if profile in ('continuous-focus','clock-fireball-pilot','history-pilot','board-potential-pilot'):
        import re
        import requests
        prefix='continuous_focus/';result_file='result.json'
        allowed={result_file,'selection.json','initial_selection.json','parent_reference.json'}
        run_pattern=(r'(?:pilot|full)_(?:uniform|air_golem)_(?:base|explore)_s6[234]'
            if profile=='continuous-focus' else r'pilot_history[12]_(?:uniform|air_golem)_s8[23]'
            if profile=='history-pilot' else r'pilot_board_(?:000|050|150|400)_s9[23]'
            if profile=='board-potential-pilot'else r'pilot_(?:current|seconds_preserved)_(?:control|miss)_s7[23]')
        for name in files:
            if not name.startswith(prefix):continue
            relative=name[len(prefix):]
            if re.fullmatch(r'corrected_full_(?:control|leak|balanced|strong)_s44_reference\.json',relative):allowed.add(relative)
            if re.fullmatch(run_pattern+r'(?:_summary\.json|_profile\.json|\.log)',relative):allowed.add(relative)
            if re.fullmatch(run_pattern+r'/(?:config\.json|metrics\.jsonl|training_result\.json)',relative):allowed.add(relative)
            if re.fullmatch(run_pattern+r'_summary\.json',relative):
                response=requests.get(files[name].url,timeout=(30,300));response.raise_for_status()
                raw=response.content;row=json.loads(raw);run=relative[:-len('_summary.json')]
                selected=row.get('checkpoint')
                if row.get('id')!=run or not isinstance(selected,str)or not re.fullmatch(re.escape(run)+r'/update_\d{6}\.pt',selected):
                    raise ValueError('Focus summary checkpoint path differs')
                if int(selected[-9:-3])>(100 if run.startswith('pilot_')else 4000):raise ValueError('Focus checkpoint exceeds declared budget')
                if prefix+selected not in files:raise ValueError('Focus selected checkpoint absent')
                prefetched[relative]=raw;allowed.add(selected)
        latest={}
        for name in files:
            match=re.fullmatch(re.escape(prefix)+'('+run_pattern+r')/update_(\d{6})\.pt',name)
            if match:
                key,update=match.group(1),int(match.group(2))
                if update>(100 if key.startswith('pilot_')else 4000):continue
                if key not in latest or update>latest[key][0]:latest[key]=(update,name[len(prefix):])
        allowed.update(v[1]for v in latest.values())
    elif profile=='continuous-pool':
        import itertools
        prefix='continuous_pool_evaluation/';result_file='result.json'
        models=('full_balanced_s44','full_strong_s44','parent','v25')
        allowed={result_file}|{name+'_test.json'for name in models}
        allowed.update(a+'__'+b+'_duels.json'for a,b in itertools.combinations(models,2))
    elif profile in ('continuous-ppo','continuous-ppo-selected'):
        import re
        prefix='continuous_ppo/';result_file='result.json'
        allowed={result_file,'selection.json'}
        selected_only=profile=='continuous-ppo-selected'
        if selected_only:
            import requests
            response=requests.get(files[prefix+result_file].url,timeout=(30,300));response.raise_for_status()
            raw=response.content;result=json.loads(raw)
            if result.get('status')!='complete'or len(result.get('full_runs',[]))!=2:raise ValueError('Complete two-model result required')
            for row in result['full_runs']:
                run=row['id'];checkpoint=row['checkpoint']
                if (not re.fullmatch(r'full_(?:control|leak|balanced|strong)_s44',run)or
                        not re.fullmatch(re.escape(run)+r'/(?:best_validation|update)_\d{6}\.pt',checkpoint)):
                    raise ValueError('Selected full checkpoint path differs')
                allowed.add(checkpoint)
            prefetched[result_file]=raw
        for name in files:
            if not name.startswith(prefix):continue
            relative=name[len(prefix):]
            if re.fullmatch(r'(?:pilot|full)_(?:control|leak|balanced|strong)_s4[234](?:_summary\.json|_profile\.json|\.log)',relative):
                allowed.add(relative)
            if re.fullmatch(r'(?:pilot|full)_(?:control|leak|balanced|strong)_s4[234]/(?:config\.json|metrics\.jsonl|training_result\.json|(?:best_validation|update)_\d{6}\.pt)',relative):
                # Keep latest resumable boundary and all validation candidates.
                if not relative.endswith('.pt')or(not selected_only and '/update_'not in relative):allowed.add(relative)
        latest={}
        for name in files:
            match=re.fullmatch(re.escape(prefix)+r'((?:pilot|full)_(?:control|leak|balanced|strong)_s4[234])/update_(\d{6})\.pt',name)
            if match:
                key,update=match.group(1),int(match.group(2))
                if key not in latest or update>latest[key][0]:latest[key]=(update,name[len(prefix):])
        if not selected_only:allowed.update(value[1]for value in latest.values())
    elif profile=='card-balance-study':
        import re
        prefix='card_balance_study/'
        result_file='result.json'
        allowed={result_file,'baseline_validation.json'}
        for trial in ('baseline','balance_all','balance_units'):
            allowed.update(trial+'/'+name for name in ('config.json','metrics.jsonl','training_result.json'))
            allowed.update(f'{trial}/best_validation_{update:06d}.pt'for update in (25,50,75,100))
            candidates=[]
            for name in files:
                match=re.fullmatch(re.escape(prefix+trial+'/update_')+r'(\d{6})\.pt',name)
                if match and 0<=int(match.group(1))<=100:candidates.append((int(match.group(1)),name[len(prefix):]))
            if candidates:allowed.add(max(candidates)[1])
    elif profile=='card-usage':
        prefix='card_usage/'
        result_file='result.json'
        ids=('v11','v14','v14_last','v18','v19','v20','v21','v24','v25','v26','v27','v28','v28b','v28b_last','v29',
             'full_defense_spell_precision','full_defense_economy_soft','full_joint_economy_soft','full_joint_tower_survival','fireball_balanced_epoch50')
        allowed={result_file}|{name+'.json'for name in ids}
    elif profile=='native-fireball-evaluation':
        prefix='native_fireball_evaluation/'
        result_file='result.json'
        allowed={result_file,'fixed_regression.json','fresh_seeds.json','king_finish.json',
                 'strategy_test.json','duels.json','duels_raw.jsonl'}
    elif profile=='strategy':
        prefix='strategy_evaluation/'
        result_file='result.json'
        allowed={result_file,'validation.json','test.json','duels.json','validation.log','test.log','duels.log'}
    elif profile=='train':
        prefix='train_evaluation/'
        result_file='result.json'
        allowed={result_file,'train.json','worker_0.jsonl','worker_1.jsonl','worker_0.log','worker_1.log'}
    elif profile=='balanced-match-validation':
        prefix='balanced_match_validation/'
        result_file='result.json'
        allowed={result_file,'teacher_v25.json','candidate_v25.json','candidate_parent.json'}
    elif profile=='fireball-balanced':
        prefix='fireball_balanced/'
        result_file='result.json'
        allowed={result_file,'candidate.pt','config.json','metrics.jsonl','baseline_skills.json',
                 'demonstration_audit.json','pair_contexts.json','rehearsal_contexts.json',
                 'shortlist.json','teacher_validation_games.json','validation_games.json'}
        allowed.update(f'epoch_{epoch}.pt'for epoch in range(10,81,10))
    elif profile=='fireball-selectivity':
        prefix='fireball_selectivity/'
        result_file='result.json'
        allowed={result_file,'candidate.pt','config.json','control_contexts.json','metrics.jsonl',
                 'baseline_validation.json','demonstration_audit.json'}
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
            if name in prefetched:raw=prefetched[name]
            else:
                response=requests.get(files[remote].url,timeout=(30,300));response.raise_for_status()
                raw=response.content
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
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
    parser.add_argument('--api-user',help='Known owner with bearer token supplied only through environment')
    parser.add_argument('--profile',choices=('defense','strategy','train','fireball','fireball-evaluation','fireball-selectivity','fireball-balanced','balanced-match-validation','native-fireball-evaluation','card-usage','card-balance-study','continuous-ppo','continuous-ppo-selected','continuous-pool','continuous-focus','clock-fireball-pilot','history-pilot','board-potential-pilot'),default='defense')
    args=parser.parse_args()
    if args.poll_seconds<1:parser.error('poll-seconds must be positive')
    # The installed SDK sends requests without a default timeout. Bound only
    # this observer process; active jobs and the installed package are untouched.
    configure_http_timeout()
    api=make_api_client(args.api_user)
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
            # Keep every partial attempt. A transient listing/download failure
            # after COMPLETE must not require restarting the remote notebook.
            destination=args.output
            if destination.exists():destination=destination.with_name(destination.name+'_attempt_'+str(time.time_ns()))
            try:
                print(json.dumps(collect(api,args.kernel,destination,profile=args.profile)),flush=True)
                break
            except Exception as error:
                print(json.dumps(dict(collection_error=type(error).__name__,destination=str(destination),remote_job_changed=False)),flush=True)
                response=getattr(error,'response',None)
                transient=(type(error).__name__ in ('SSLError','ConnectionError','ReadTimeout','Timeout')or
                    getattr(response,'status_code',None)in (404,429,500,502,503,504))
                if args.once or not transient:raise
        if args.once:break
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
