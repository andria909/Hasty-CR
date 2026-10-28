"""Paid edge, pocket and late-response probes; candidates are not live approved."""
import argparse
from itertools import product
import json
from pathlib import Path
from scripts.attack.audit_cannon_tower_states import run_case,CANDIDATES
from scripts.attack.sequence_physics import physics_pins
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.runner import resolve_deck

CELLS=dict(CANDIDATES,bridge_near=(4,18),king_front=(7,26))


def audit(output):
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    contexts=[]
    for own,lane,delay in product(range(4),('left','right'),(1350,2025)):
        contexts.append(dict(own_mask=own,enemy_mask=3,lane=lane,threat='balloon',delay=delay,route='edge',decision_ms=1500))
        for threat in ('hog_rider','giant','balloon'):
            contexts.append(dict(own_mask=own,enemy_mask=3,lane=lane,threat=threat,delay=delay,route='bridge',decision_ms=5500))
            if not own & (1 if lane=='left'else 2):
                contexts.append(dict(own_mask=own,enemy_mask=3,lane=lane,threat=threat,delay=delay,route='pocket',decision_ms=0))
    rows=[]
    for index,context in enumerate(contexts):
        group=[run_case(cards,spells,**context,choice=choice,candidate_cells=CELLS)for choice in ('none',*CELLS)]
        baseline=group[0]['total_damage']
        for row in group:
            row.update(context=index,damage_prevented_vs_no_card=baseline-row['total_damage'])
        rows.extend(group)
        if (index+1)%20==0:print(json.dumps(dict(contexts=index+1,total=len(contexts))),flush=True)
    result=dict(schema='cannon-stress-candidate-audit-v1',episodes=len(rows),contexts=len(contexts),
        physics_pins=physics_pins(),geometry='native_anchors_candidate_v1',rows=rows,
        live_mask_enabled=False,quality_accepted=False,
        limitations=['Pocket rectangle is explicitly unverified in simulator geometry; results conditional on it',
                     'Single paid attacker only; no spell/support/split/occupied-cell coverage',
                     'Fixed full-health kings; six Cannon candidates not professional grid calibration',
                     'TRAIN diagnostic, not a win rate or live targeting validation'])
    with Path(output).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.output);print(json.dumps(dict(episodes=result['episodes'],contexts=result['contexts'])))
