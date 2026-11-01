"""Separate movement states in recorded crossings without changing the simulator.

State names describe speed hypotheses, not native jump flags. All intervals,
including stops and impulses, enter the fit. Direction is fitted retrospectively
and must not be mistaken for a causal landing-point prediction.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SPEEDS = (0, 2400, 3200)


def decode(points, transition_cost=50):
    if len(points)<2:raise ValueError('At least two observations required')
    durations=[];distances=[]
    for a,b in zip(points,points[1:]):
        ticks=b['tick']-a['tick']
        if not 0<ticks<=5:raise ValueError('Non-contiguous native clock')
        durations.append(ticks/20)
        distances.append(math.hypot(b['x']-a['x'],b['y']-a['y']))
    emissions=np.abs(np.asarray(distances)[:,None]-np.asarray(durations)[:,None]*SPEEDS)
    cost=emissions[0].copy();back=[]
    switch=transition_cost*(1-np.eye(3))
    for row in emissions[1:]:
        candidate=cost[:,None]+switch
        previous=candidate.argmin(axis=0);back.append(previous)
        cost=candidate[previous,np.arange(3)]+row
    path=[int(cost.argmin())]
    for previous in reversed(back):path.append(int(previous[path[-1]]))
    return list(reversed(path)),durations


def segment(points,start,end):
    # Interval indices [start,end); both endpoint observations are retained.
    p=points[start:end+1]
    t=np.asarray([(r['tick']-p[0]['tick'])/20 for r in p])
    xy=np.asarray([[r['x'],r['y']]for r in p],dtype=float)
    design=np.column_stack((np.ones(len(t)),t))
    fit=np.linalg.lstsq(design,xy,rcond=None)[0]
    predicted=design@fit
    velocity=fit[1];speed=float(np.linalg.norm(velocity))
    direction=velocity/speed if speed else velocity
    errors={str(v):float(np.linalg.norm(xy-(xy[0]+t[:,None]*direction*v),axis=1).mean()/1000)
            for v in (2400,3200)}
    return dict(first_tick=p[0]['tick'],last_tick=p[-1]['tick'],duration_s=float(t[-1]),
        intervals=end-start,first_xy=xy[0].tolist(),last_xy=xy[-1].tolist(),
        fitted_speed_mt_per_sec=speed,velocity_mt_per_sec=velocity.tolist(),
        straight_line_rmse_tiles=float(np.sqrt(np.mean(np.sum((xy-predicted)**2,axis=1)))/1000),
        fixed_speed_mean_error_tiles=errors,
        turn_at_end_degrees=turn(points,end),
        left_censored=start==0,right_censored=end==len(points)-1)


def turn(points,index):
    if not 0<index<len(points)-1:return None
    a,b,c=points[index-1:index+2]
    u=np.array([b['x']-a['x'],b['y']-a['y']],float)
    v=np.array([c['x']-b['x'],c['y']-b['y']],float)
    norm=np.linalg.norm(u)*np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(u@v/norm,-1,1))))if norm else None


def distribution(values):
    return dict(n=len(values),p10=float(np.quantile(values,.1)),median=float(np.median(values)),
        p90=float(np.quantile(values,.9)))if values else dict(n=0)


def analyze(source,transition_cost=50):
    source=Path(source);raw=source.read_bytes();data=json.loads(raw)
    if data['schema']!='native-hog-river-crossings-v1':raise ValueError('Native crossing inventory required')
    rows=[];counts=Counter();seconds=Counter();segments=[]
    for index,case in enumerate(data['cases']):
        states,durations=decode(case['points'],transition_cost)
        for state,dt in zip(states,durations):counts[str(SPEEDS[state])]+=1;seconds[str(SPEEDS[state])]+=dt
        groups=[];start=0
        for end in range(1,len(states)+1):
            if end<len(states)and states[end]==states[start]:continue
            if states[start]==2 and end-start>=3 and sum(durations[start:end])>=.2:
                item=segment(case['points'],start,end)
                item.update(case=index,source=case['source'],world=case['world'],identity=case['identity'])
                groups.append(item);segments.append(item)
            start=end
        rows.append(dict(case=index,interval_states_mt_per_sec=[SPEEDS[s]for s in states],fast_segments=groups))
    keys={(s['source'],json.dumps(s['world']))for s in segments}
    summary=dict(crossings=len(rows),intervals=sum(counts.values()),interval_state_counts=dict(counts),
        duration_by_state_seconds=dict(seconds),fast_segments=len(segments),
        crossings_with_fast_segment=len({s['case']for s in segments}),source_world_groups=len(keys),
        fitted_fast_speed=distribution([s['fitted_speed_mt_per_sec']for s in segments]),
        straight_line_rmse_tiles=distribution([s['straight_line_rmse_tiles']for s in segments]),
        duration_s=distribution([s['duration_s']for s in segments]),
        retrospective_fixed_speed_mean_error_tiles={str(v):distribution([s['fixed_speed_mean_error_tiles'][str(v)]for s in segments])for v in (2400,3200)},
        turn_at_end_degrees=distribution([s['turn_at_end_degrees']for s in segments if s['turn_at_end_degrees']is not None]))
    return dict(schema='native-hog-movement-segments-v1',source=str(source),source_sha256=hashlib.sha256(raw).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol=dict(speed_hypotheses_mt_per_sec=SPEEDS,transition_cost_mt=transition_cost,
            loss='absolute displacement residual plus state switch cost',minimum_fast_intervals=3,
            minimum_fast_duration_s=.2,all_intervals_included=True,raw_sources_rehashed=False),
        summary=summary,cases=rows,accepted_physics_change=False,
        limitations=['Fast state is inferred, not a native jump flag; buffs/impulses are not observed.',
            'Fitted direction uses future positions and is not a causal route prediction.',
            'Fast speed summaries are conditional on state inference, not unbiased population estimates.',
            'Observation gaps of at most five ticks leave state boundaries uncertain.',
            'These previously inspected crossings are discovery data, not an untouched test.',
            'Do not change active training physics from segmentation alone.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--transition-cost',type=float,default=50)
    args=parser.parse_args();result=analyze(args.source,args.transition_cost)
    with Path(args.output).open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result['summary']))
