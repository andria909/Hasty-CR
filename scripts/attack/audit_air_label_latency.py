"""Reproduce and recheck existing TRAIN Musketeer defense targets at live delays."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

from scripts.kaggle.local_artifact_transfer import file_sha
from sim.adapter import grid_to_point
from sim.continuous_push import DECKS
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26,ClashEnv
from sim.match import Match
from sim.runner import resolve_deck


def opening(record,cards,spells):
    match=Match(cards=cards,spells=spells,decks=(list(DECK_26),list(DECKS['air'])),seed=record['seed'],geometry='native_anchors_candidate_v1')
    match.players[1].elixir=int(record['initial_own_elixir']*1000);match.players[-1].elixir=10000
    hp={k:t.hitpoints for k,t in match.towers[1].items()}
    x=3 if record['lane']=='left'else 14
    if not match.play_card(-1,'balloon',grid_to_point(17-x,16,-1)):raise ValueError('Original paid Balloon unavailable')
    while match.elapsed_ms<500:match.step(50)
    if record['support'] and not match.play_card(-1,'baby_dragon',grid_to_point(17-x,18,-1)):raise ValueError('Paid support unavailable')
    if record['decision_ms']!=500+record['reaction_ms']:raise ValueError('Original decision time differs')
    while match.elapsed_ms<record['decision_ms']:match.step(50)
    return match,hp


def outcome(match,hp,action,delay):
    if type(delay)is not int or not 0<=delay<=10000:raise ValueError('Finite delay required')
    trial=deepcopy(match);decision=match.elapsed_ms;slot,x,y=ClashEnv.decode(int(action))
    if trial.players[1].hand[slot]!='musketeer':raise ValueError('Original Musketeer slot required')
    while trial.elapsed_ms<decision+delay:trial.step(50)
    if not trial.play_card(1,'musketeer',grid_to_point(x,y,1)):raise ValueError('Paid Musketeer unavailable')
    placed=trial.elapsed_ms
    while not trial.finished and trial.elapsed_ms<decision+60000:trial.step(50)
    remaining=sum(e.alive and e.side==-1 and not e.is_tower for e in trial.battle.entities.values())
    damage=sum(value-trial.towers[1][key].hitpoints for key,value in hp.items())
    return dict(action=int(action),cell=[x,y],success=remaining==0 and damage==0,own_damage=damage,remaining_enemies=remaining,placed_ms=placed)


def audit(data,output):
    data,output=Path(data),Path(output);manifest=json.loads((data/'manifest.json').read_text())
    if manifest['split']!='train' or file_sha(data/'examples.npz')!=manifest['data_sha256']:raise ValueError('Verified TRAIN data required')
    with np.load(data/'examples.npz',allow_pickle=False)as arrays:masks=arrays['success_masks'].copy()
    records=[r for r in manifest['records']if r.get('skill')=='air_defense' and r['accepted']]
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for record in records:
        match,hp=opening(record,cards,spells);stored={r['action']:r for r in record['trials']}
        actions=np.flatnonzero(masks[record['array_index']]);results={225:[],1350:[],2025:[]}
        for action in actions:
            base=outcome(match,hp,action,225)
            if not base['success'] or base['own_damage']!=stored[int(action)]['own_damage'] or not stored[int(action)]['success']:
                raise ValueError('Original air label failed225ms reproduction')
            results[225].append(base)
            for delay in (1350,2025):results[delay].append(outcome(match,hp,action,delay))
        rows.append(dict({k:v for k,v in record.items()if k!='trials'},original_actions=len(actions),by_delay={str(d):dict(
            successful_actions=sum(v['success']for v in values),success_action_ids=[v['action']for v in values if v['success']],
            mean_own_damage=float(np.mean([v['own_damage']for v in values])),outcomes=values)for d,values in results.items()}))
        print(json.dumps(dict(completed=len(rows),total=len(records))),flush=True)
    summary={str(d):dict(actions=sum(r['original_actions']for r in rows),successful_actions=sum(r['by_delay'][str(d)]['successful_actions']for r in rows),
        fixtures=len(rows),fixtures_with_any_success=sum(r['by_delay'][str(d)]['successful_actions']>0 for r in rows))for d in(225,1350,2025)}
    report=dict(schema='air-train-label-latency-audit-v1',summary=summary,rows=rows,data_sha256=manifest['data_sha256'],
        source_manifest_sha256=file_sha(data/'manifest.json'),audit_source_sha256=file_sha(__file__),
        source_data=str(data),original225ms_labels_reproduced=True,training_data_modified=False,quality_accepted=False,
        limitations=['Existing single-Musketeer positive labels only, not every possible defense',
                    'Cannon+spirit/spells and reserve sequences are not tested here',
                    'Targets share48fixtures and3seeds; not independent games or winrates'])
    with(output/'result.json').open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.data,a.output)
