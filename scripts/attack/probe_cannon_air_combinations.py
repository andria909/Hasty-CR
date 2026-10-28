"""Bounded paid defense search after both princess towers are lost; TRAIN only."""
import argparse
from itertools import product
import json
from pathlib import Path
from scripts.attack.audit_cannon_tower_states import ENEMY,ready_hand
from scripts.attack.search_late_air_defense import rollout,rank
from scripts.attack.sequence_execution import EXECUTION
from scripts.attack.sequence_physics import physics_pins
from sim.adapter import grid_to_point
from sim.cannon_placement import CENTRAL_CELLS
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.match import Match
from sim.runner import resolve_deck


def opening(cards,spells,enemy_mask,lane):
    match=Match(cards=cards,spells=spells,decks=(list(DECK_26),list(ENEMY)),seed=17,
                geometry='native_anchors_candidate_v1')
    for side,mask in ((1,0),(-1,enemy_mask)):
        for bit,key in enumerate(('left','right')):
            if not mask&(1<<bit):match.towers[side][key].hitpoints=0
    match._wake_kings()
    player=match.players[1]
    player.hand=['cannon','electro_spirit','zap','fireball']
    player.queue=[c for c in player.deck if c not in player.hand];player.elixir=10000
    ready_hand(match.players[-1],'balloon')
    assert match.play_card(-1,'balloon',grid_to_point(0 if lane=='left'else 17,15,1))
    hp={k:t.hitpoints for k,t in match.towers[1].items()}
    while match.elapsed_ms<1500:match.step(50)
    return match,hp


def audit(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    spec=dict(schema='central-cannon-air-combination-probe-v1',physics_pins=physics_pins(),
        execution=EXECUTION,delays=[1350,2025],own_princess_mask=0,enemy_masks=list(range(4)),
        beam_cells=2,helper_offsets_ms=[1500,3000,5000],helper_variants=3,
        initial_hand=['cannon','electro_spirit','zap','fireball'],initial_elixir=10,
        horizon_ms=60000,split='TRAIN diagnostic',live_games_started=False)
    (output/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n')
    results=[];attempts=0
    with(output/'trials.jsonl').open('x')as f:
        for index,(enemy,lane,delay)in enumerate(product(range(4),('left','right'),spec['delays'])):
            match,hp=opening(cards,spells,enemy,lane);rows=[]
            def attempt(kind,first=None,second=None,third=None):
                nonlocal attempts
                result=rollout(match,hp,cards,spells,lane,first,second,third=third,delay_ms=delay)
                row=dict(context=index,kind=kind,first=first,second=second,third=third,**result)
                rows.append(row);attempts+=1;f.write(json.dumps(row)+'\n');return row
            attempt('none')
            singles=[attempt('cannon',dict(card='cannon',cell=[x if lane=='left'else 17-x,y]))for x,y in CENTRAL_CELLS]
            beams=sorted((r for r in singles if r['valid']),key=rank)[:2]
            for single in beams:
                first=single['first']
                for card in ('electro_spirit','zap','fireball'):
                    for offset,variant in product(spec['helper_offsets_ms'],range(3)):
                        attempt('cannon_'+card,first,dict(card=card,offset_ms=offset,variant=variant))
                for offset,sv,zv in product(spec['helper_offsets_ms'],range(3),range(3)):
                    attempt('cannon_spirit_zap',first,dict(card='electro_spirit',offset_ms=offset,variant=sv),
                            dict(card='zap',offset_ms=offset+250,variant=zv))
            best={}
            for kind in sorted({r['kind']for r in rows}):
                legal=[r for r in rows if r['kind']==kind and r['valid']]
                best[kind]=min(legal,key=rank)if legal else None
            results.append(dict(enemy_mask=enemy,lane=lane,delay_ms=delay,best=best,
                                attempts=len(rows),valid=sum(r['valid']for r in rows)))
            f.flush();print(json.dumps(dict(contexts=index+1,total=16,trials=attempts)),flush=True)
    if physics_pins()!=spec['physics_pins']:raise ValueError('Physics changed during search')
    summary={}
    for kind in results[0]['best']:
        choices=[r['best'][kind]for r in results if r['best'][kind]is not None]
        summary[kind]=dict(contexts_with_valid_plan=len(choices),zero_damage=sum(r['clear_without_damage']for r in choices),
            king_survives=sum(not r['own_king_dead']for r in choices),
            mean_damage=sum(r['total_damage']for r in choices)/len(choices),
            mean_paid_elixir=sum(r['elixir_spent']for r in choices)/len(choices))
    result=dict(protocol=spec,attempts=attempts,contexts=results,summary=summary,
        training_started=False,quality_accepted=False,
        limitations=['Hindsight bounded search over two best single-Cannon cells; not a learned policy or global optimum.',
            'Full ten-elixir bank and fixed legal hand; no enemy support, reinforcements or spells.',
            'Helpers use observed current target position plus bounded offsets, not future coordinates.',
            'Clear includes60s resolution; no-damage metric includes bomb impacts, not just first hits.',
            'Simulator mechanics remain unverified against real activation and target locks.'])
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.output)
