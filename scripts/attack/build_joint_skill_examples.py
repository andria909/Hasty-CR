"""Fresh TRAIN-only air/Fireball labels with a shared grouped holdout.

Positive Fireballs are verified by delayed resolved damage. Empty-board
restraint examples accept ANY legal non-Fireball action, not only WAIT.
Nothing is copied from the separately measured validation fixtures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.attack.build_air_position_examples import build as build_air
from scripts.attack.evaluate_fireball_clusters import fixture_cases,cast_outcome,reference_candidates
from sim.continuous_push import DECKS
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26,ACTIONS
from sim.eval_defense_duel import JOINT_MODE,mechanical_mask,policy_observation
from sim.match import Match
from sim.runner import resolve_deck


def common_seeds():
    cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    chosen=[]
    for seed in range(3800000,3801000):
        m=Match(cards=cards,spells=spells,decks=(list(DECK_26),list(DECKS['air'])),seed=seed,geometry='native_anchors_candidate_v1')
        if {'fireball','musketeer'}<=set(m.players[1].hand) and {'balloon','baby_dragon'}<=set(m.players[-1].hand):chosen.append(seed)
        if len(chosen)==3:return chosen
    raise ValueError('Three shared legal seed groups required')


def build(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);seeds=common_seeds()
    air=build_air(root/'air',seeds=seeds)
    with np.load(root/'air/examples.npz',allow_pickle=False)as arrays:
        packed={key:list(arrays[key])for key in('planes','scalars','masks','success_masks')}
    records=[dict(r,skill='air_defense')for r in air['records']]
    for label,m,cards,spells in fixture_cases(seeds=seeds):
        mask=mechanical_mask(m,cards,spells,1);success=np.zeros(ACTIONS,dtype=np.bool_)
        outcomes=[]
        for action in reference_candidates(m,mask):
            result=cast_outcome(m,action);outcomes.append(result)
            if result['damaged_units']>=3 and result['fireball_exchange_value']>0:success[action]=True
        accepted=bool(success.any());row=dict(label,skill='fireball_group',accepted=accepted,candidate_outcomes=outcomes)
        if accepted:
            obs=policy_observation(m,1,JOINT_MODE);row['array_index']=len(packed['planes'])
            for key,value in [('planes',obs['planes']),('scalars',obs['scalars']),('masks',mask),('success_masks',success)]:packed[key].append(value)
        records.append(row)
    cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    for seed in seeds:
        for bank in(5000,10000):
            for age in(1500,4500):
                m=Match(cards=cards,spells=spells,decks=(list(DECK_26),list(DECKS['air'])),seed=seed,geometry='native_anchors_candidate_v1')
                m.players[1].elixir=bank
                while m.elapsed_ms<age:m.step(50)
                mask=mechanical_mask(m,cards,spells,1);success=mask.copy();slot=m.players[1].hand.index('fireball')
                success[1+slot*576:1+(slot+1)*576]=False
                assert success.any() and not any(e.alive and e.side==-1 and not e.is_tower for e in m.battle.entities.values())
                obs=policy_observation(m,1,JOINT_MODE);row=dict(seed=seed,skill='fireball_restraint',accepted=True,
                    initial_own_elixir=bank/1000,age_ms=age,array_index=len(packed['planes']))
                for key,value in [('planes',obs['planes']),('scalars',obs['scalars']),('masks',mask),('success_masks',success)]:packed[key].append(value)
                records.append(row)
    np.savez_compressed(root/'examples.npz',**{key:np.stack(value)for key,value in packed.items()})
    counts={skill:sum(r['accepted']and r['skill']==skill for r in records)for skill in('air_defense','fireball_group','fireball_restraint')}
    result=dict(schema='joint-public-skill-labels-v1',split='train',seeds=seeds,examples=len(packed['planes']),records=records,by_skill=counts,
        data_sha256=hashlib.sha256((root/'examples.npz').read_bytes()).hexdigest(),heldout_seed=max(seeds),
        heldout_scope='Largest whole seed group excluded across ALL skills, including restraint',
        scope='Successful local actions and narrow empty-board restraint, not optimal complete expert trajectories',
        command_delay_ms=225,own_inputs='Public20-plane309-scalar only; future damage is a label, never an input',
        validation_data_reused=False,quality_accepted=False,live_games_started=False,
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()for name in
            (__file__,'scripts/attack/build_air_position_examples.py','scripts/attack/evaluate_fireball_clusters.py',
             'sim/match.py','sim/engine.py','sim/spells.py','sim/entities.py','sim/env.py','sim/eval_defense_duel.py')})
    (root/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(examples=result['examples'],seeds=seeds,by_skill=counts)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();build(a.output)
