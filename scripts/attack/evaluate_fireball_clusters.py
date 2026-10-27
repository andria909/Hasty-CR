"""Finite offline Fireball placement diagnostic; no policy overrides in live play.

Free card selection and conditional Fireball aim are reported separately.
The reference is a bounded search around visible groups, not a global oracle.
Fixture outcomes are validation diagnostics, never training demonstrations.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sim.adapter import grid_to_point
from sim.continuous_push import CONTRACT,DECKS
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26,ClashEnv
from sim.eval_defense_duel import JOINT_MODE,mechanical_mask,policy_observation
from sim.match import Match
from sim.runner import resolve_deck
from sim.tactical_rewards import fireball_outcomes
from sim.train_attack_ppo import initialize_attack_policy

GROUPS=(('barbarians',),('minions',),('barbarians','musketeer'),('barbarians','bomber'))
COMMAND_DELAY_MS=225


def fixture_cases(seed_start=7000000,seeds=None):
    cards,spells,_,_=load_defense_assets()
    cards.update(resolve_deck(cards,list(DECK_26)))
    for group_index,group in enumerate(GROUPS):
        enemy=list(DECKS['hog'])
        for index,card in enumerate(group):enemy[index]=card
        # Avoid duplicate entries after replacing the first slots.
        unique=[]
        for card in enemy+['archers','spear_goblins','minions','giant']:
            if card not in unique:unique.append(card)
        enemy=unique[:8]
        def make(seed):
            return Match(cards=cards,spells=spells,decks=(list(DECK_26),enemy),seed=seed,
                         geometry='native_anchors_candidate_v1')
        selected=[]
        candidates=seeds if seeds is not None else range(seed_start+group_index*1000,seed_start+1000+group_index*1000)
        for seed in candidates:
            m=make(seed)
            if 'fireball'in m.players[1].hand and set(group)<=set(m.players[-1].hand):selected.append(seed)
            if len(selected)==3:break
        if len(selected)!=3 or len(set(selected))!=3:raise ValueError('Three legal hand groups required')
        for seed in selected:
            for lane in ('left','right'):
                for age_ms in (1500,4500):
                    m=make(seed);m.players[1].elixir=10000;m.players[-1].elixir=10000
                    x=3 if lane=='left'else 14
                    for index,card in enumerate(group):
                        while m.elapsed_ms<index*500:m.step(50)
                        assert m.play_card(-1,card,grid_to_point(17-x,18+index,-1))
                    while m.elapsed_ms<age_ms:m.step(50)
                    assert not m.finished
                    label=dict(id=f"{'+'.join(group)}_s{seed}_{lane}_age{age_ms}",seed=seed,
                               group=list(group),lane=lane,age_ms=age_ms,
                               enemy_elixir_paid=m.players[-1].elixir_spent)
                    yield label,m,cards,spells


def cast_outcome(match,action):
    trial=deepcopy(match)
    slot,x,y=ClashEnv.decode(int(action))
    if trial.players[1].hand[slot]!='fireball':raise ValueError('Expected a legal Fireball action')
    requested_ms=trial.elapsed_ms
    while trial.elapsed_ms<requested_ms+COMMAND_DELAY_MS:trial.step(50)
    if not trial.play_card(1,'fireball',grid_to_point(x,y,1)):
        raise ValueError('Diagnostic cast became illegal')
    cast=trial.battle.spell_cast_log[-1]
    while not cast['resolved'] and not trial.finished and trial.elapsed_ms<requested_ms+10000:
        trial.step(50)
    if not cast['resolved']:raise ValueError('Diagnostic cast did not resolve within finite horizon')
    hits=[h for h in cast['hits']if h['target_side']==-1 and h['damage']>0]
    units=[h for h in hits if not h['is_tower']]
    return dict(action=int(action),cell=[x,y],**fireball_outcomes([cast]),
                damaged_units=len(units),unit_damage=sum(h['damage']for h in units),
                tower_damage=sum(h['damage']for h in hits if h['is_tower']),
                hit_uids=sorted(h['target_uid']for h in units),
                empty=not bool(cast['hits']),request_ms=requested_ms,
                cast_ms=cast['cast_ms'],impact_ms=max((h['hit_ms']for h in cast['hits']),default=trial.elapsed_ms))


def reference_candidates(match,mask):
    slot=match.players[1].hand.index('fireball')
    enemies=[e for e in match.battle.entities.values()if e.alive and e.side==-1 and not e.is_tower]
    centers=[(e.pos.x/1000,e.pos.y/1000)for e in enemies]
    if enemies:centers.append((np.mean([x for x,y in centers]),np.mean([y for x,y in centers])))
    cells=set()
    for cx,cy in centers:
        for dx in(-1,0,1):
            for dy in(0,1,2):
                x,y=round(cx)+dx,round(cy)+dy
                if 0<=x<18 and 0<=y<32:
                    action=1+slot*576+y*18+x
                    if mask[action]:cells.add(action)
    if not cells:raise ValueError('No visible group and legal reference target')
    return sorted(cells)


def summarize(records):
    def subset(rows):
        n=len(rows)
        return dict(fixtures=n,free_card_choices=dict(Counter(r['free_card']for r in rows)),
            free_fireballs=sum(r['free_card']=='fireball'for r in rows),
            conditional_group_hits=sum(r['conditional_cast']['damaged_units']>=3 for r in rows),
            conditional_empty=sum(r['conditional_cast']['empty']for r in rows),
            mean_conditional_damaged_units=sum(r['conditional_cast']['damaged_units']for r in rows)/n,
            mean_conditional_exchange_value=sum(r['conditional_cast']['fireball_exchange_value']for r in rows)/n,
            reference_group_opportunities=sum(r['reference_best']['damaged_units']>=3 and r['reference_best']['fireball_exchange_value']>0 for r in rows))
    return dict(all=subset(records),by_lane={lane:subset([r for r in records if r['lane']==lane])for lane in('left','right')},
                by_group={'+'.join(group):subset([r for r in records if r['group']==list(group)])for group in GROUPS})


def evaluate(checkpoints,output):
    output=Path(output)
    if output.exists():raise ValueError('Preserve existing diagnostic output; choose a new path')
    torch.set_num_threads(1)
    nets=[]
    for path in checkpoints:
        path=Path(path);blob=torch.load(path,map_location='cpu',weights_only=False)
        if blob['state_dict']['conv.0.weight'].shape[1]!=20 or blob['state_dict']['scalar.0.weight'].shape[1]!=309:
            raise ValueError('Exact public20-plane309-scalar checkpoint required')
        net,_,_=initialize_attack_policy(blob,device='cpu',num_scalars=309,observation_contract=CONTRACT)
        net.eval();nets.append((path,net,hashlib.sha256(path.read_bytes()).hexdigest(),[]))
    fixture_rows=[]
    for index,(label,m,cards,spells)in enumerate(fixture_cases(),1):
        obs=policy_observation(m,1,JOINT_MODE);mask=mechanical_mask(m,cards,spells,1)
        slot=m.players[1].hand.index('fireball');cache={}
        def resolved(action):
            action=int(action)
            if action not in cache:cache[action]=cast_outcome(m,action)
            return cache[action]
        references=[resolved(action)for action in reference_candidates(m,mask)]
        best=max(references,key=lambda r:(r['fireball_exchange_value'],r['damaged_units'],-r['action']))
        fixture_rows.append(dict(label,reference_best=best,candidates_checked=len(references)))
        for path,net,digest,records in nets:
            with torch.inference_mode():
                logits,_=net(torch.as_tensor(obs['planes'])[None],torch.as_tensor(obs['scalars'])[None])
                values=logits[0].numpy()
            if not np.isfinite(values[mask]).all():raise ValueError('Nonfinite legal model logits')
            action=int(np.where(mask,values,-np.inf).argmax())
            conditional_mask=mask.copy();conditional_mask[:1+slot*576]=False;conditional_mask[1+(slot+1)*576:]=False
            aim=int(np.where(conditional_mask,values,-np.inf).argmax())
            free_card='hold'if action==0 else m.players[1].hand[(action-1)//576]
            records.append(dict(label,free_action=action,free_card=free_card,
                                conditional_cast=resolved(aim),reference_best=best))
        print(json.dumps(dict(event='FIREBALL_DIAGNOSTIC_PROGRESS',completed=index,total=48)),flush=True)
    result=dict(schema='fireball-cluster-placement-validation-v1',split='validation_diagnostic',
        models=[dict(checkpoint=str(path),checkpoint_sha256=digest,summary=summarize(records),records=records)
                for path,net,digest,records in nets],fixtures=fixture_rows,
        command_delay_ms=COMMAND_DELAY_MS,own_initial_elixir=10,
        reference='Bounded search around currently visible units; not globally optimal',
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()for name in
            (__file__,'sim/match.py','sim/engine.py','sim/spells.py','sim/entities.py','sim/adapter.py',
             'sim/env.py','sim/eval_defense_duel.py','sim/tactical_rewards.py','sim/public_gamedata.py')},
        quality_accepted=False,used_for_training=False,live_games_started=False,
        limitations=['Single-cast fixtures, not whole-match wins or proof that Fireball is the best card',
            'Conditional aiming forces Fireball only inside this offline diagnostic; free choice reported separately',
            'Symmetric lanes share a seed;48 rows are not48 independent opening hands',
            'Native timing calibrated but all simulator mechanics are not proven equivalent to the game'])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weights',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();evaluate(a.weights,a.output)
