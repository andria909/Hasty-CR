"""TRAIN-only paid Cannon candidate audit across all surviving tower patterns.

Grid candidates are hypotheses, not professional/live-validated placements.
This diagnostic does not alter action masks, rewards, or frozen GPU packages.
"""
import argparse
from itertools import product
import json
from pathlib import Path
from sim.adapter import grid_to_point
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.match import Match
from sim.runner import resolve_deck
from sim.tower_support import TowerSupport
from scripts.attack.sequence_physics import physics_pins

CANDIDATES={'central_near':(8,22),'central_far':(9,22),
            'central_high':(8,21),'lane_near':(6,22)}
ENEMY=('hog_rider','giant','balloon','knight','musketeer','skeletons','zap','fireball')


def ready_hand(player,card):
    rest=[c for c in player.deck if c!=card]
    player.hand=[card,*rest[:3]];player.queue=rest[3:];player.elixir=10000


def run_case(cards,spells,own_mask,enemy_mask,lane,threat,delay,choice,activate=False,*,
             route="bridge",decision_ms=1500,candidate_cells=None,trace=None):
    m=Match(cards=cards,spells=spells,decks=(list(DECK_26),list(ENEMY)),seed=17,
            geometry='native_anchors_candidate_v1')
    for side,bits in ((1,own_mask),(-1,enemy_mask)):
        for bit,key in enumerate(('left','right')):
            if not bits & (1<<bit):m.towers[side][key].hitpoints=0
    if activate:m.towers[1]['king'].target_only_buildings=False
    m._wake_kings()
    start_hp={key:t.hitpoints for key,t in m.towers[1].items()}
    ready_hand(m.players[-1],threat);ready_hand(m.players[1],'cannon')
    if route not in ('bridge','edge','pocket'):raise ValueError('Unknown attack route')
    if route=='edge' and threat!='balloon':raise ValueError('Edge probe is flying Balloon only')
    if route=='pocket' and own_mask & (1 if lane=='left' else 2):
        raise ValueError('Pocket attack requires destroyed own princess on that side')
    if not 0<=decision_ms<=10000 or delay not in (1350,2025):raise ValueError('Bounded measured timing required')
    x=(0 if route=='edge' else 3) if lane=='left' else (17 if route=='edge' else 14)
    world_y=23 if route=='pocket' else 15
    assert m.play_card(-1,threat,grid_to_point(x,world_y,1))
    tracker=TowerSupport(m.battle).attach();placed=False;pull=False;first_lock=None
    candidates=CANDIDATES if candidate_cells is None else candidate_cells
    if choice!='none' and choice not in candidates:raise ValueError('Unknown candidate')
    cell=candidates.get(choice)
    if cell and lane=='right':cell=(17-cell[0],cell[1])
    while not m.finished and m.elapsed_ms<30000:
        if cell and not placed and m.elapsed_ms>=decision_ms+delay:
            placed=m.play_card(1,'cannon',grid_to_point(*cell,1))
            if not placed:raise ValueError('Candidate is not mechanically legal')
        m.step(50)
        if trace is not None:trace(m)
        for unit in m.battle.entities.values():
            target=m.battle.entities.get(unit.target_uid)
            if unit.alive and unit.side==-1 and not unit.is_tower and target and target.name=='cannon' and target.side==1:
                pull=True
                if first_lock is None:first_lock=m.elapsed_ms
    damage={key:value-m.towers[1][key].hitpoints for key,value in start_hp.items()}
    return dict(own_mask=own_mask,enemy_mask=enemy_mask,king_activated_initially=activate or own_mask!=3,
        lane=lane,threat=threat,delay_ms=delay,choice=choice,cell=cell,paid_cannon=placed,
        route=route,decision_ms=decision_ms,enemy_cell=[x,world_y],
        paid_enemy=True,pulled=pull,first_cannon_lock_ms=first_lock,damage=damage,
        total_damage=sum(damage.values()),king_dead=not m.towers[1]['king'].alive,
        remaining_enemies=sum(e.alive and e.side==-1 and not e.is_tower for e in m.battle.entities.values()),
        **tracker.metrics())


def audit(output,*,quick=False):
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    missing=set(ENEMY)-set(cards)
    if missing:raise ValueError('Missing actual enemy card data: '+str(missing))
    rows=[];masks=[3]if quick else range(4)
    for own,enemy,lane,threat,delay in product(masks,masks,('left','right'),('hog_rider','giant','balloon'),(1350,2025)):
        for active in ((False,True)if own==3 else (True,)):
            for choice in ('none',*CANDIDATES):
                rows.append(run_case(cards,spells,own,enemy,lane,threat,delay,choice,active))
        if len(rows)%100==0:print(json.dumps(dict(episodes=len(rows))),flush=True)
    groups={}
    for row in rows:
        key=tuple(row[k]for k in ('own_mask','enemy_mask','lane','threat','delay_ms','king_activated_initially'))
        groups.setdefault(key,[]).append(row)
    for group in groups.values():
        baseline=next(r for r in group if r['choice']=='none')['total_damage']
        for row in group:row['damage_prevented_vs_no_card']=baseline-row['total_damage']
    report=dict(schema='cannon-tower-state-candidate-audit-v1',split='TRAIN diagnostic, not a win-rate test',
        physics_pins=physics_pins(),geometry='native_anchors_candidate_v1',decision_ms=1500,
        horizon_ms=30000,episodes=len(rows),contexts=len(groups),live_mask_enabled=False,
        professional_grid_mapping_verified=False,rows=rows,
        limitations=['One paid unsupported attacker per scenario; no split pushes, spells, occupied cells or pocket spawns',
        'Grid candidates have NOT been calibrated against professional screenshots or real-game targeting',
        'Simulator mechanics and native geometry remain candidates; no live correctness claim',
        'Both king towers initially alive; missing princess implies awake king; invalid sleeping-king states excluded'])
    with Path(output).open('x')as f:json.dump(report,f,indent=2);f.write('\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--quick',action='store_true')
    a=p.parse_args();r=audit(a.output,quick=a.quick);print(json.dumps(dict(episodes=r['episodes'],contexts=r['contexts'])))
