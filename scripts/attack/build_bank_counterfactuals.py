"""Paired TRAIN-only bank decisions with a common bounded continuation search.

Each first action starts from the identical snapshot. Waiting is labelled only
when it beats the alternatives under the declared tower-loss/clear/cost score.
These are local synthetic exercises, not full-match expert demonstrations.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np

from sim.adapter import grid_to_point
from sim.continuous_push import ContinuousPushEnv
from sim.env import ACTIONS, DECK_26


def opening(seed, lane, bank, enemy_y):
    env=ContinuousPushEnv(action_delay_ms=1350,continuous_profile=dict(
        cannon_center_only=1,tower_damage_all=1,source_radius_reach=1,hog_sight_clipping=1))
    env.reset(seed=seed,scenario_id='defense::air/push')
    own=env.match.players[1]
    own.hand=['skeletons','electro_spirit','musketeer','cannon']
    own.queue=[c for c in DECK_26 if c not in own.hand]
    own.elixir=bank
    enemy=env.match.players[-1];enemy.elixir=10000
    rest=[c for c in enemy.deck if c!='balloon']
    enemy.hand=['balloon',*rest[:3]];enemy.queue=rest[3:]
    column=2+seed%3
    x=column if lane=='left' else 17-column
    if not env.match.play_card(-1,'balloon',grid_to_point(x,enemy_y,1)):
        raise ValueError('Paid opponent opening failed')
    # Vary how far the paid threat has already progressed. No reinforcements.
    for _ in range((seed%3)*10):env.match.step(50)
    own.elixir=bank
    env.next_enemy=10**9  # local exercise: no reinforcement after paid opening
    env._previous=env._metrics()
    return env


def rollout(env, first, cell, horizon=45000):
    env=deepcopy(env);hp={k:t.hitpoints for k,t in env.match.towers[1].items()}
    start=env.match.elapsed_ms
    env.step(first);sent=bool(first and (first-1)//576==2)
    while not env.done and env.match.elapsed_ms<start+horizon:
        live=any(e.alive and e.side==-1 and not e.is_tower for e in env.match.battle.entities.values())
        # Continue through the horizon: Balloon death bombs and in-flight
        # shots can still damage a tower after the last enemy body dies.
        action=0
        if not sent and live and 'musketeer' in env.match.players[1].hand:
            slot=env.match.players[1].hand.index('musketeer');x,y=cell
            candidate=1+slot*576+y*18+x
            if env._mask()[candidate]:action=candidate;sent=True
        env.step(action)
    damage=sum(value-env.match.towers[1][k].hitpoints for k,value in hp.items())
    remaining=sum(e.alive and e.side==-1 and not e.is_tower for e in env.match.battle.entities.values())
    spent=sum(env.cards[a['card']].cost for a in env.actions)
    return dict(first_action=first,followup_cell=list(cell),own_damage=damage,
        remaining_enemies=remaining,spent_elixir=spent,commands=env.actions,
        end_ms=env.match.elapsed_ms)


def score(row):return row['own_damage'],row['remaining_enemies'],row['spent_elixir']


def label(env,lane):
    mask=env._mask();x=3 if lane=='left' else 14
    actions=[0]
    for slot in (0,1):
        for y in (18,27):
            a=1+slot*576+y*18+x
            if mask[a]:actions.append(a)
    for cx,y in [(x,20),(6 if lane=='left'else 11,22),(x,24)]:
        a=1+2*576+y*18+cx
        if mask[a]:actions.append(a)
    outcomes=[]
    for action in actions:
        trials=[rollout(env,action,(cx,y)) for cx,y in
                [(x,20),(6 if lane=='left'else 11,22),(x,24)]]
        outcomes.append(min(trials,key=score))
    best=min(map(score,outcomes));success=np.zeros(ACTIONS,dtype=np.bool_)
    for row in outcomes:
        if score(row)==best:success[row['first_action']]=True
    # A tie between WAIT and every deployment teaches no distinction.
    accepted=bool(success.any() and not all(success[a]for a in actions))
    return accepted,success,outcomes


def build(output,seeds=(352,353,354),banks=(2000,3000,4000),enemy_rows=(9,12,15)):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    arrays={k:[] for k in ('planes','scalars','masks','success_masks')};records=[]
    for seed in seeds:
        for lane in ('left','right'):
            for bank in banks:
                for y in enemy_rows:
                    env=opening(seed,lane,bank,y)
                    accepted,success,outcomes=label(env,lane)
                    record=dict(seed=seed,lane=lane,bank=bank,enemy_y=y,accepted=accepted,
                        alternatives=outcomes,skill='bank_counterfactual',action=0 if success[0]else int(np.flatnonzero(success)[0]))
                    if accepted:
                        obs=env._observe();record['array_index']=len(arrays['planes'])
                        for k,v in dict(planes=obs['planes'],scalars=obs['scalars'],masks=env._mask(),success_masks=success).items():arrays[k].append(v)
                    records.append(record)
                    print(json.dumps(dict(contexts=len(records),accepted=len(arrays['planes']),seed=seed)),flush=True)
    if not arrays['planes']:raise ValueError('No discriminative teacher examples')
    np.savez_compressed(output/'examples.npz',**{k:np.stack(v)for k,v in arrays.items()})
    identities={}
    for record in records:
        if not record['accepted']:continue
        index=record['array_index']
        digest=hashlib.sha256(b''.join(arrays[k][index].tobytes()for k in ('planes','scalars','masks'))).hexdigest()
        if digest in identities and identities[digest]!=record['seed']:
            raise ValueError('Identical policy inputs cross seed groups')
        identities[digest]=record['seed'];record['input_sha256']=digest
    names=['sim/continuous_push.py','sim/engine.py','sim/match.py','sim/gamedata.py',
           'sim/public_gamedata.py','sim/entities.py','sim/adapter.py','sim/env.py','sim/projectile_calibration.py']
    manifest=dict(schema='bank-counterfactual-labels-v1',split='train',seeds=list(seeds),
        examples=len(arrays['planes']),records=records,command_delay_ms=1350,
        data_sha256=hashlib.sha256((output/'examples.npz').read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physics_pins={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()for n in names},
        score_order=['own tower HP lost','remaining enemy bodies','spent elixir'],
        limitations=['Synthetic bank/hand starts','One paid Balloon, no reinforcements',
            'Bounded three-cell Musketeer continuation, not globally optimal',
            'Correlated seed repeats; holdout must use whole seed',
            'Not full-game quality or a live win rate'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    result=build(a.output,**(dict(seeds=(352,),banks=(3000,),enemy_rows=(12,))if a.smoke else {}))
    print(json.dumps(dict(examples=result['examples'])))
