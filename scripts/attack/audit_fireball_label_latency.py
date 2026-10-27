"""Recheck existing TRAIN Fireball success labels at measured command delays.

Preserves original data and GPU packages. Outcomes are simulator labels only.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

from scripts.attack.evaluate_fireball_clusters import fixture_cases,cast_outcome
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.adapter import grid_to_point
from sim.env import ClashEnv
from sim.tactical_rewards import fireball_outcomes


def delayed_cast(match,action,delay_ms):
    if not isinstance(delay_ms,int) or not 0<=delay_ms<=10000:raise ValueError('Finite integer command delay required')
    trial=deepcopy(match);slot,x,y=ClashEnv.decode(int(action))
    if trial.players[1].hand[slot]!='fireball':raise ValueError('Fireball action required')
    requested=trial.elapsed_ms
    while trial.elapsed_ms<requested+delay_ms:trial.step(50)
    if not trial.play_card(1,'fireball',grid_to_point(x,y,1)):raise ValueError('Delayed cast illegal')
    cast=trial.battle.spell_cast_log[-1]
    while not cast['resolved'] and not trial.finished and trial.elapsed_ms<requested+delay_ms+10000:trial.step(50)
    if not cast['resolved']:raise ValueError('Delayed cast did not resolve')
    hits=[h for h in cast['hits']if h['target_side']==-1 and h['damage']>0]
    units=[h for h in hits if not h['is_tower']]
    return dict(action=int(action),cell=[x,y],**fireball_outcomes([cast]),damaged_units=len(units),
        unit_damage=sum(h['damage']for h in units),tower_damage=sum(h['damage']for h in hits if h['is_tower']),
        hit_uids=sorted(h['target_uid']for h in units),empty=not bool(cast['hits']),request_ms=requested,
        cast_ms=cast['cast_ms'],impact_ms=max((h['hit_ms']for h in cast['hits']),default=trial.elapsed_ms))


def succeeds(row):return row['damaged_units']>=3 and row['fireball_exchange_value']>0


def audit(data,output):
    data,output=Path(data),Path(output)
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest['split']!='train' or manifest['command_delay_ms']!=225 or file_sha(data/'examples.npz')!=manifest['data_sha256']:
        raise ValueError('Original225ms TRAIN data required')
    output.mkdir(parents=True,exist_ok=False)
    with np.load(data/'examples.npz',allow_pickle=False)as arrays:
        success_masks=arrays['success_masks'].copy()
    old={r['id']:r for r in manifest['records']if r.get('skill')=='fireball_group' and r.get('accepted')}
    rows=[]
    for label,match,_,_ in fixture_cases(seeds=manifest['seeds']):
        if label['id']not in old:continue
        record=old[label['id']];actions=np.flatnonzero(success_masks[record['array_index']])
        stored={r['action']:r for r in record['candidate_outcomes']}
        outcomes={225:[],1350:[],2025:[]}
        for action in actions:
            baseline=delayed_cast(match,int(action),225)
            if baseline!=stored[int(action)] or not succeeds(baseline):
                raise ValueError('Original labels no longer reproduce at225ms; inspect physics before latency comparison')
            outcomes[225].append(baseline)
            for delay in (1350,2025):outcomes[delay].append(delayed_cast(match,int(action),delay))
        rows.append(dict(label,original_actions=len(actions),by_delay={str(d):dict(
            successful_actions=sum(succeeds(r)for r in values),empty_actions=sum(r['empty']for r in values),outcomes=values)
            for d,values in outcomes.items()}))
        print(json.dumps(dict(completed=len(rows),total=len(old))),flush=True)
    if len(rows)!=len(old):raise ValueError('Original Fireball fixture coverage differs')
    summary={str(d):dict(actions=sum(r['original_actions']for r in rows),
        successful_actions=sum(r['by_delay'][str(d)]['successful_actions']for r in rows),
        empty_actions=sum(r['by_delay'][str(d)]['empty_actions']for r in rows),
        fixtures_with_any_success=sum(r['by_delay'][str(d)]['successful_actions']>0 for r in rows),fixtures=len(rows))for d in(225,1350,2025)}
    result=dict(schema='fireball-train-label-delay-audit-v1',summary=summary,rows=rows,source_data=str(data),
        source_manifest_sha256=file_sha(data/'manifest.json'),source_data_sha256=manifest['data_sha256'],
        original225ms_outcomes_reproduced=True,training_data_modified=False,quality_accepted=False,
        limitations=['Dependent action targets from only3seed groups, not independent games',
                     'Includes local heldout seed for diagnosing existing labels only; no new TRAIN labels written',
                     'Only old positive labels tested; misses do not prove no better aim exists at longer delay'])
    with(output/'result.json').open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();audit(args.data,args.output)
