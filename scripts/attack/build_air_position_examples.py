"""Collect successful local-defense actions from TRAIN-only paid fixtures.

Labels are sets of successful legal Musketeer cells, not claims of optimal
full-game play. Observations are public and taken after every enemy play.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from sim.adapter import grid_to_point
from sim.continuous_push import DECKS
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26,ACTIONS
from sim.eval_defense_duel import policy_observation,mechanical_mask,JOINT_MODE
from sim.match import Match
from sim.runner import resolve_deck


def build(output,seeds=None):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    def make(seed):return Match(cards=cards,spells=spells,decks=(list(DECK_26),list(DECKS['air'])),seed=seed,geometry='native_anchors_candidate_v1')
    if seeds is None:
        seeds=[]
        for seed in range(300001,301001):
            m=make(seed)
            if 'musketeer'in m.players[1].hand and {'balloon','baby_dragon'}<=set(m.players[-1].hand):seeds.append(seed)
            if len(seeds)==3:break
    else:
        seeds=list(seeds)
        if len(set(seeds))!=len(seeds):raise ValueError('Distinct fixture seeds required')
        for seed in seeds:
            m=make(seed)
            if 'musketeer'not in m.players[1].hand or not {'balloon','baby_dragon'}<=set(m.players[-1].hand):
                raise ValueError('Supplied seed lacks legal opening cards')
    if len(seeds)!=3:raise ValueError('Insufficient legal opening hands')
    planes=[];scalars=[];masks=[];labels=[];records=[]
    for seed in seeds:
        for lane in ('left','right'):
            for support in (False,True):
                for bank in (5000,10000):
                    for reaction_ms in (0,1500):
                        m=make(seed);m.players[1].elixir=bank;m.players[-1].elixir=10000
                        x0=3 if lane=='left'else 14
                        original_hp={k:t.hitpoints for k,t in m.towers[1].items()}
                        assert m.play_card(-1,'balloon',grid_to_point(17-x0,16,-1))
                        while m.elapsed_ms<500:m.step(50)
                        if support:assert m.play_card(-1,'baby_dragon',grid_to_point(17-x0,18,-1))
                        decision_ms=500+reaction_ms
                        while m.elapsed_ms<decision_ms:m.step(50)
                        obs=policy_observation(m,1,JOINT_MODE);mask=mechanical_mask(m,cards,spells,1)
                        slot=m.players[1].hand.index('musketeer');safe=np.zeros(ACTIONS,dtype=np.bool_);trials=[]
                        for cx0 in (1,3,6):
                            for y in (17,20,23,26):
                                x=cx0 if lane=='left'else 17-cx0;action=1+slot*576+y*18+x
                                if not mask[action]:continue
                                trial=deepcopy(m)
                                while trial.elapsed_ms<decision_ms+225:trial.step(50)
                                assert trial.play_card(1,'musketeer',grid_to_point(x,y,1))
                                while not trial.finished and trial.elapsed_ms<decision_ms+60000:trial.step(50)
                                remaining=sum(e.alive and e.side==-1 and not e.is_tower for e in trial.battle.entities.values())
                                damage=sum(hp-trial.towers[1][k].hitpoints for k,hp in original_hp.items())
                                success=remaining==0 and damage==0
                                safe[action]=success;trials.append(dict(action=action,x=x,y=y,success=success,own_damage=damage))
                        accepted=bool(safe.any())
                        record=dict(seed=seed,lane=lane,support=support,initial_own_elixir=bank/1000,reaction_ms=reaction_ms,
                            decision_ms=decision_ms,accepted=accepted,trials=trials)
                        if accepted:
                            record['array_index']=len(planes);planes.append(obs['planes']);scalars.append(obs['scalars']);masks.append(mask);labels.append(safe)
                        records.append(record)
    if not planes:raise ValueError('No successful local defense examples')
    np.savez_compressed(root/'examples.npz',planes=np.stack(planes),scalars=np.stack(scalars),masks=np.stack(masks),success_masks=np.stack(labels))
    report=dict(schema='public-air-position-examples-v1',split='train',seeds=seeds,fixtures=len(records),examples=len(planes),
        candidates_checked=sum(len(r['trials'])for r in records),records=records,
        data_sha256=hashlib.sha256((root/'examples.npz').read_bytes()).hexdigest(),
        scope='Successful single-action local defenses, not complete expert trajectories or full-game wins',
        own_inputs='Public20-plane309-scalar observation; no future schedule or hidden enemy bank supplied',
        command_delay_ms=225,physics_changed=False,live_games_started=False)
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=build(a.output)
    print(json.dumps({k:r[k]for k in('fixtures','examples','candidates_checked','data_sha256')}))
