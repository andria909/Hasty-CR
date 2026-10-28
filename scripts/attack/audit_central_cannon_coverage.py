"""Enumerate the actual allowed Cannon cells in paid TRAIN diagnostic scenes.

Best-cell summaries are hindsight oracles, never model win rates or live labels.
"""
import argparse
from itertools import product
import json
from pathlib import Path

from scripts.attack.audit_cannon_tower_states import run_case
from scripts.attack.sequence_physics import physics_pins
from sim.cannon_placement import CENTRAL_CELLS
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.runner import resolve_deck

CELLS={f'x{x}_y{y}':(x,y)for x,y in CENTRAL_CELLS}


def contexts():
    rows=[]
    for own,enemy,lane,delay in product(range(4),range(4),('left','right'),(1350,2025)):
        for threat,route,decision in [('hog_rider','bridge',1500),('balloon','edge',1500),
                                      ('hog_rider','bridge',5500)]:
            rows.append(dict(own_mask=own,enemy_mask=enemy,lane=lane,threat=threat,
                             delay=delay,route=route,decision_ms=decision))
        if not own&(1 if lane=='left'else 2):
            rows.append(dict(own_mask=own,enemy_mask=enemy,lane=lane,threat='hog_rider',
                             delay=delay,route='pocket',decision_ms=0))
    return rows


def summarize_group(context,rows):
    baseline=next(r for r in rows if r['choice']=='none')
    paid=[r for r in rows if r['choice']!='none']
    if (len(paid)!=36 or {tuple(r['cell'])for r in paid}!=set(CENTRAL_CELLS)
            or not all(r['paid_cannon']for r in paid)):
        raise ValueError('Actual central-cell coverage missing or illegal')
    best=min(r['total_damage']for r in paid)
    return dict(**context,baseline_damage=baseline['total_damage'],best_paid_damage=best,
        best_prevented=baseline['total_damage']-best,
        zero_damage_cells=sum(r['total_damage']==0 for r in paid),
        pull_cells=sum(r['pulled']for r in paid),
        improved_cells=sum(r['total_damage']<baseline['total_damage']for r in paid),
        worsening_cells=sum(r['total_damage']>baseline['total_damage']for r in paid),
        best_cells=[r['cell']for r in paid if r['total_damage']==best])


def mirror_key(context):
    def flip(mask):return ((mask&1)<<1)|((mask&2)>>1)
    return (flip(context['own_mask']),flip(context['enemy_mask']),
        'right'if context['lane']=='left'else'left',context['threat'],context['delay'],
        context['route'],context['decision_ms'])


def context_key(context):
    return tuple(context[k]for k in ('own_mask','enemy_mask','lane','threat','delay','route','decision_ms'))


def audit(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    spec=dict(schema='central-cannon-coverage-protocol-v1',contexts=contexts(),cells=CELLS,
        physics_pins=physics_pins(),horizon_ms=30000,split='TRAIN diagnostic',
        labels_verified=False,model_evaluation=False,live_games_started=False)
    (output/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n')
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    groups=[];outcomes={};mismatches=[];pairs=0
    with (output/'rows.jsonl').open('x') as f:
        for index,context in enumerate(spec['contexts']):
            rows=[run_case(cards,spells,**context,choice=choice,candidate_cells=CELLS)
                  for choice in ('none',*CELLS)]
            for row in rows:f.write(json.dumps(dict(context=index,**row))+'\n')
            f.flush()
            groups.append(summarize_group(context,rows))
            signature=[(r['choice'],r['total_damage'],r['pulled'],r['first_cannon_lock_ms'],r['king_dead'])for r in rows]
            other=outcomes.get(mirror_key(context))
            if other is not None:
                pairs+=len(rows)
                for left,right in zip(signature,other):
                    if left!=right:mismatches.append(dict(context=context,observed=left,mirror=right))
            outcomes[context_key(context)]=signature
            if (index+1)%8==0:
                print(json.dumps(dict(contexts=index+1,total=len(spec['contexts']),episodes=(index+1)*37)),flush=True)
    if physics_pins()!=spec['physics_pins']:raise ValueError('Physics changed during audit')
    summaries=[]
    keys=sorted({(r['threat'],r['route'],r['decision_ms'],r['delay'])for r in groups})
    for key in keys:
        selected=[r for r in groups if (r['threat'],r['route'],r['decision_ms'],r['delay'])==key]
        summaries.append(dict(threat=key[0],route=key[1],decision_ms=key[2],delay_ms=key[3],
            contexts=len(selected),any_pull=sum(r['pull_cells']>0 for r in selected),
            any_zero_damage=sum(r['zero_damage_cells']>0 for r in selected),
            any_improvement=sum(r['improved_cells']>0 for r in selected),
            all_cells_worse_than_wait=sum(r['worsening_cells']==36 for r in selected)))
    result=dict(schema='central-cannon-coverage-audit-v1',protocol=str(output/'protocol.json'),
        physics_pins=spec['physics_pins'],episodes=len(groups)*37,contexts=len(groups),
        mirror_pairs=pairs,mirror_mismatches=mismatches,summaries=summaries,groups=groups,
        mask_changed=False,quality_accepted=False,labels_verified=False,
        limitations=['Best cell is a hindsight oracle, not a learned policy.',
            'One paid unsupported attacker; Cannon alone, no defensive spell or spirit.',
            'Unoccupied board, full-health kings, missing princess wakes king; no extra activation while both princesses live.',
            'Pocket geometry and real target retention remain unverified; simulator coverage does not prove live defense.'])
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(episodes=result['episodes'],contexts=result['contexts'],
                         mirror_pairs=pairs,mirror_mismatches=len(mismatches))),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    args=p.parse_args();audit(args.output)
