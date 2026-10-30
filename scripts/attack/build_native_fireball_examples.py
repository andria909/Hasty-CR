"""Fresh grouped Fireball examples at measured latency and current physics.

Only development seeds are used. Future simulated positions generate labels,
never policy inputs. Candidate search is bounded and not an optimality proof.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scripts.attack.evaluate_fireball_clusters import GROUPS,fixture_cases,reference_candidates
from scripts.attack.audit_fireball_label_latency import delayed_cast,succeeds
from scripts.attack.sequence_physics import physics_pins
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.continuous_push import DECKS
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26,ACTIONS
from sim.eval_defense_duel import JOINT_MODE,mechanical_mask,policy_observation
from sim.match import Match
from sim.runner import resolve_deck


def common_seeds(count=12):
    cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    decks=[]
    for group in GROUPS:
        enemy=list(DECKS['hog'])
        for i,card in enumerate(group):enemy[i]=card
        unique=[]
        for card in enemy+['archers','spear_goblins','minions','giant']:
            if card not in unique:unique.append(card)
        decks.append((group,unique[:8]))
    selected=[]
    for seed in range(3810000,3811000):
        legal=True
        for group,enemy in decks:
            m=Match(cards=cards,spells=spells,decks=(list(DECK_26),enemy),seed=seed,geometry='native_anchors_candidate_v1')
            if 'fireball' not in m.players[1].hand or not set(group)<=set(m.players[-1].hand):legal=False;break
        if legal:selected.append(seed)
        if len(selected)==count:return selected
    raise ValueError('Insufficient legal development seeds')


def build(output,seeds=None,central_cannon=False,restrain_zap=False):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    pins=physics_pins();seeds=common_seeds() if seeds is None else list(seeds);heldout=max(seeds)
    if len(seeds)<3 or len(set(seeds))!=len(seeds):raise ValueError('Three or more distinct development seeds required')
    packed={k:[] for k in ('planes','scalars','masks','success_masks')};records=[]
    def append(label,m,cards,spells,success):
        if central_cannon and (m.battle.normal_range_mode!='edge_to_edge_v1' or m.battle.hog_sight_mode!='side_clip_v1'):
            raise ValueError('Current mixed curriculum requires corrected combat modes')
        mask=mechanical_mask(m,cards,spells,1)
        if np.any(success&~mask):raise ValueError('Illegal label')
        if central_cannon:
            from sim.cannon_placement import restrict_cannon,CENTRAL_MODE
            mask=restrict_cannon(mask,m.players[1].hand,CENTRAL_MODE)
            success=success&mask
        row=dict(label,accepted=bool(success.any()))
        if row['accepted']:
            row['array_index']=len(packed['planes']);obs=policy_observation(m,1,JOINT_MODE)
            for k,v in [('planes',obs['planes']),('scalars',obs['scalars']),('masks',mask),('success_masks',success)]:packed[k].append(v)
        records.append(row)
    for offset in range(0,len(seeds),3):
        for label,m,cards,spells in fixture_cases(seeds=seeds[offset:offset+3]):
            mask=mechanical_mask(m,cards,spells,1);candidates=set(reference_candidates(m,mask))
            future=deepcopy(m)
            while future.elapsed_ms<m.elapsed_ms+1350:future.step(50)
            candidates.update(reference_candidates(future,mask))
            outcomes=[delayed_cast(m,a,1350) for a in sorted(candidates)]
            success=np.zeros(ACTIONS,dtype=np.bool_)
            for outcome in outcomes:
                if succeeds(outcome):success[outcome['action']]=True
            append(dict(label,skill='fireball_group',candidate_outcomes=outcomes),m,cards,spells,success)
            print(json.dumps(dict(fixtures=len(records),accepted=len(packed['planes']))),flush=True)
    # Empty-board examples use the same seed groups, making both skills present
    # in the heldout group. All legal non-Fireball choices are accepted.
    cards,spells,_,_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    for seed in seeds:
        for bank in (5000,10000):
            for age in (1500,4500):
                m=Match(cards=cards,spells=spells,decks=(list(DECK_26),list(DECKS['hog'])),seed=seed,geometry='native_anchors_candidate_v1')
                m.players[1].elixir=bank
                while m.elapsed_ms<age:m.step(50)
                success=mechanical_mask(m,cards,spells,1)
                for slot,card in enumerate(m.players[1].hand):
                    if card=='fireball' or (restrain_zap and card=='zap'):
                        success[1+slot*576:1+(slot+1)*576]=False
                append(dict(seed=seed,skill='fireball_restraint',age_ms=age,initial_own_elixir=bank/1000),m,cards,spells,success)
    assert pins==physics_pins()
    for skill in ('fireball_group','fireball_restraint'):
        if not any(r['accepted'] and r['skill']==skill and r['seed']==heldout for r in records):raise ValueError('Skill lacks heldout coverage')
    np.savez_compressed(root/'examples.npz',**{k:np.stack(v) for k,v in packed.items()})
    result=dict(schema='native-fireball-development-examples-v1',split='train',seeds=seeds,heldout_seed=heldout,
        records=records,examples=len(packed['planes']),command_delay_ms=1350,physics_pins=pins,
        cannon_placement_mode='central_defense_v1' if central_cannon else 'mechanical_all_legal',
        empty_board_forbidden_spells=['fireball','zap'] if restrain_zap else ['fireball'],
        data_sha256=file_sha(root/'examples.npz'),builder_sha256=file_sha(__file__),
        helper_sha256={p:file_sha(p) for p in ('scripts/attack/evaluate_fireball_clusters.py','scripts/attack/audit_fireball_label_latency.py')},
        quality_accepted=False,used_in_training=False,scope='Paid narrow Fireball development fixtures; maximum seed held out for both skills; projected candidates only affect labels, never inputs.')
    (root/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();build(a.output)
