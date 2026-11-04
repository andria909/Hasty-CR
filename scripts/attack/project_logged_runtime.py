"""Project full PPO wall time from complete pilot log intervals and frozen full commands."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from scripts.kaggle.local_artifact_transfer import file_sha
from scripts.kaggle.stream_training import iter_log_objects


def project(start,points,*,pilot_updates=400,pilot_eval_every=50,full_updates=4000,full_eval_every=250):
    if any(not isinstance(v,(int,float))or not math.isfinite(v)for v in (start,*points.values())):
        raise ValueError('Finite remote timestamps required')
    if sorted(points)!=list(range(1,pilot_updates+1)):raise ValueError('Complete consecutive pilot timing required')
    if full_updates%full_eval_every:raise ValueError('Whole full evaluation periods required')
    intervals={i:points[i]-(start if i==1 else points[i-1])for i in points}
    if any(v<0 for v in intervals.values()):raise ValueError('Remote timestamps regressed')
    regular=[v for i,v in intervals.items()if i>1 and i%pilot_eval_every]
    evaluation=[v for i,v in intervals.items()if i%pilot_eval_every==0]
    if not regular or not evaluation or mean(regular)<=0:raise ValueError('Observed training/evaluation timing required')
    blocks=[]
    for end in range(pilot_eval_every,pilot_updates+1,pilot_eval_every):
        values=[intervals[i]for i in range(max(2,end-pilot_eval_every+1),end)]
        if values:blocks.append(mean(values))
    evaluations=full_updates//full_eval_every;normal=full_updates-evaluations-1
    return dict(observed_pilot_seconds=points[pilot_updates]-start,initial_interval_seconds=intervals[1],
        regular_update_mean_seconds=mean(regular),evaluation_update_mean_seconds=mean(evaluation),
        evaluation_update_seconds=evaluation,regular_block_means_seconds=blocks,
        full_updates=full_updates,full_evaluations=evaluations,
        estimated_full_seconds=intervals[1]+normal*mean(regular)+evaluations*mean(evaluation),
        slowest_observed_blocks_full_seconds=intervals[1]+normal*max(blocks)+evaluations*max(evaluation))


def audit(root,full_preflight):
    root=Path(root);prep=json.loads((root/'preparation.json').read_text());path=root/'live_stream/events.jsonl'
    if json.loads((root/'live_stream/progress.json').read_text())['kernel']!=prep['kernel']:
        raise ValueError('Log kernel differs from prepared study')
    raw=path.read_bytes();snapshot_dir=root/'runtime_log_snapshots';snapshot_dir.mkdir(exist_ok=True)
    snapshot=snapshot_dir/(hashlib.sha256(raw).hexdigest()+'.jsonl')
    if snapshot.exists():
        if snapshot.read_bytes()!=raw:raise ValueError('Snapshot content differs')
    else:
        with snapshot.open('xb')as file:file.write(raw)
    starts={};points={};completed={};inputs=False
    for line in raw.decode().splitlines():
        event=json.loads(line)
        for row in iter_log_objects(event.get('data','')):
            kind=row.get('event');name=row.get('trial_id');t=event.get('time')
            if kind=='INPUTS_VERIFIED':inputs|=row['protocol_sha256']==file_sha(root/'dataset/protocol.json')
            elif kind=='TRIAL_START':
                if name in starts and starts[name]!=t:raise ValueError('Repeated execution of one trial identity')
                starts[name]=t
            elif kind=='TRIAL_PROGRESS':
                if row['actual_cuda']!='Tesla T4' or row['step']!=row['update']*1024 or row['target_steps']!=409600:
                    raise ValueError('Actual expected T4 pilot counters required')
                values=points.setdefault(name,{})
                if row['update']in values and values[row['update']]!=t:raise ValueError('Conflicting replay timestamp')
                values[row['update']]=t
            elif kind=='FOCUS_TRIAL_COMPLETE':completed[name]=row
    if not inputs:raise ValueError('Matching frozen inputs not observed')
    preflight=json.loads(Path(full_preflight).read_text());runs=[]
    for check in preflight['checks']:
        if check['variant']!=prep['study_variant']or not check.get('full_run_ast_unchanged')or not check['proof'].get('long_economy_full_verified'):
            raise ValueError('Matching verified frozen full command required')
        command=check['proof']['original_full_command']
        value=lambda flag:int(command[command.index(flag)+1])
        for name,event in completed.items():
            if not name.startswith('pilot_'+check['trial']+'_s'):continue
            result=project(starts[name],points[name],full_updates=value('--updates'),full_eval_every=value('--eval-every'))
            result.update(id=name,full_wallclock_seconds=value('--wallclock-seconds'),
                logged_score=event['score'],logged_family_guard=event['family_guard'],
                complete_weights_verified=False)
            result['slow_scenario_headroom_seconds']=result['full_wallclock_seconds']-result['slowest_observed_blocks_full_seconds']
            runs.append(result)
    if not runs:raise ValueError('Completed matching pilot log required')
    return dict(schema='full-runtime-from-complete-pilot-log-v1',kernel=prep['kernel'],runs=runs,
        raw_log=str(snapshot),original_live_log=str(path),raw_log_sha256=file_sha(snapshot),full_preflight=str(full_preflight),full_preflight_sha256=file_sha(full_preflight),
        training_modified=False,quality_accepted=False,scope='Forecast from logged complete pilot counters; full weights not yet downloaded',
        limitations=['Remote logs may be buffered; intervals are not direct profiler measurements',
                    'Evaluation blocks also include checkpoint I/O and one PPO update',
                    'Future match complexity and hardware contention can change the observed rate',
                    'Slow-block scenario is a sensitivity estimate, not a bound or confidence interval',
                    'Kernel setup, final export and transfer are outside trainer forecast'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--full-preflight',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();target=Path(a.output)
    if target.exists():raise ValueError('Preserve previous forecast')
    target.write_text(json.dumps(audit(a.root,a.full_preflight),indent=2)+'\n')
