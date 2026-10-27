"""Bounded TRAIN counterfactual search with1350ms delivery and paid commands.

Other free slots remain usable while a command is in flight. This is a local
search over single cards and two-card continuations, never a live controller.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.attack.audit_air_label_latency import opening
from scripts.kaggle.local_artifact_transfer import file_sha
from sim.adapter import grid_to_point
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.eval_defense_duel import mechanical_mask
from sim.runner import resolve_deck


def helper_cell(match,lane,card,variant):
    x=3 if lane=='left' else 14
    center=6 if lane=='left' else 11
    if card in ('fireball','zap'):
        enemies=[e for e in match.battle.entities.values()if e.alive and e.side==-1 and not e.is_tower]
        if not enemies:return None
        target=min(enemies,key=lambda e:(e.name!='balloon',-e.pos.y,e.uid))
        return min(17,max(0,int(target.pos.x/1000))),min(31,max(0,int(target.pos.y/1000)+2*variant))
    if card=='cannon':return [(center,21),(center,23),(center,25)][variant]
    return [(x,21),(center,23),(x,25)][variant]


def rollout(initial,hp,cards,spells,lane,first=None,second=None):
    match=deepcopy(initial);start=match.elapsed_ms;pending=[];ready={};commands=[];invalid=[]
    decisions=[]
    if first:decisions.append((start,first['card'],first['cell'],None))
    if second:decisions.append((start+second['offset_ms'],second['card'],None,second['variant']))
    while not match.finished and match.elapsed_ms<start+60000:
        for event in list(decisions):
            if event[0]>match.elapsed_ms:continue
            decisions.remove(event);_,card,cell,variant=event;player=match.players[1]
            cell=cell if cell is not None else helper_cell(match,lane,card,variant)
            reserved=sum(cards[p['card']].cost*1000 for p in pending)
            if card not in player.hand or cell is None:
                invalid.append('card_or_target_unavailable');continue
            slot=player.hand.index(card);x,y=cell;action=1+slot*576+y*18+x
            if (any(p['slot']==slot for p in pending) or match.elapsed_ms<ready.get(slot,0)
                    or player.elixir-reserved<cards[card].cost*1000
                    or not mechanical_mask(match,cards,spells,1)[action]):
                invalid.append('not_legal_or_unreserved_elixir');continue
            command=dict(card=card,slot=slot,cell=list(cell),requested_ms=match.elapsed_ms,
                         due_ms=match.elapsed_ms+1350,cost=cards[card].cost)
            pending.append(command)
        for command in list(pending):
            if command['due_ms']>match.elapsed_ms:continue
            pending.remove(command)
            if not match.play_card(1,command['card'],grid_to_point(*command['cell'],1)):
                invalid.append('rejected_at_delivery');continue
            ready[command['slot']]=match.elapsed_ms+2100
            commands.append(dict(command,placed_ms=match.elapsed_ms))
        match.step(50)
    damage={k:value-match.towers[1][k].hitpoints for k,value in hp.items()}
    remaining=sum(e.alive and e.side==-1 and not e.is_tower for e in match.battle.entities.values())
    return dict(valid=not invalid and len(commands)==int(first is not None)+int(second is not None),
        invalid_reasons=invalid,commands=commands,total_damage=sum(damage.values()),own_tower_damage=damage,
        own_king_dead=not match.towers[1]['king'].alive,remaining_enemies=remaining,
        elixir_spent=sum(c['cost']for c in commands),clear_without_damage=remaining==0 and sum(damage.values())==0)


def rank(row):
    return row['own_king_dead'],row['remaining_enemies']>0,row['total_damage'],row['elixir_spent']


def search(data,output):
    data,output=Path(data),Path(output);manifest=json.loads((data/'manifest.json').read_text())
    if manifest['split']!='train' or file_sha(data/'examples.npz')!=manifest['data_sha256']:raise ValueError('Original TRAIN data required')
    contexts=[r for r in manifest['records']if r.get('skill')=='air_defense' and r['accepted'] and r['reaction_ms']==1500]
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(schema='late-air-native-delay-search-v1',source_data=str(data),data_sha256=manifest['data_sha256'],
        contexts=len(contexts),command_delay_ms=1350,slot_refill_ms=2100,beam_per_first_card=2,
        helper_offsets_ms=[250,2000,5000],helper_variants=3,helpers=['cannon','musketeer','electro_spirit','zap','fireball'],
        rank='King survival, enemies cleared, minimum own tower damage, then elixir spent',
        source_sha256=file_sha(__file__),training_data_modified=False,live_games_started=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    summaries=[];attempted=legal=0
    with(output/'trials.jsonl').open('x')as stream:
        for index,record in enumerate(contexts):
            match,hp=opening(record,cards,spells);records=[]
            def save(row,kind,first=None,second=None):
                nonlocal attempted,legal
                attempted+=1;legal+=row['valid']
                row=dict(row,context=index,kind=kind,first=first,second=second)
                stream.write(json.dumps(row)+'\n');records.append(row);return row
            save(rollout(match,hp,cards,spells,record['lane']),'no_cards')
            beams=[]
            for card in ('musketeer','cannon'):
                if card not in match.players[1].hand:continue
                choices=[]
                xs=range(1,9) if card=='musketeer' else range(5,9)
                ys=range(18,29,2) if card=='musketeer' else range(18,26,2)
                for x0 in xs:
                    for y in ys:
                        x=x0 if record['lane']=='left' else 17-x0;first=dict(card=card,cell=[x,y])
                        row=save(rollout(match,hp,cards,spells,record['lane'],first),'single',first)
                        if row['valid']:choices.append(row)
                beams.extend(sorted(choices,key=rank)[:2])
            for row in beams:
                first=row['first']
                for helper in protocol['helpers']:
                    if helper==first['card']:continue
                    for offset in protocol['helper_offsets_ms']:
                        for variant in range(3):
                            second=dict(card=helper,offset_ms=offset,variant=variant)
                            save(rollout(match,hp,cards,spells,record['lane'],first,second),'pair',first,second)
            valid=[r for r in records if r['valid']]
            best=min(valid,key=rank);singles=[r for r in valid if r['kind']=='single' and r['first']['card']=='musketeer']
            pairs=[r for r in valid if r['kind']=='pair']
            summaries.append(dict(context=index,opening={k:v for k,v in record.items()if k!='trials'},
                best=best,best_single_musketeer=min(singles,key=rank)if singles else None,
                best_pair=min(pairs,key=rank)if pairs else None,
                attempted=len(records),legal=len(valid)))
            stream.flush();print(json.dumps(dict(contexts_complete=index+1,total=len(contexts),trials=attempted)),flush=True)
    result=dict(protocol=protocol,contexts=summaries,attempted=attempted,legal=legal,
        zero_damage_contexts=sum(r['best']['clear_without_damage']for r in summaries),quality_accepted=False,
        limitations=['Bounded beam search, not globally optimal defense','No enemy reinforcements in this local fixture',
                    'Helper target uses current visible position plus bounded offsets, not future coordinates',
                    'Existing hand and paid elixir preserved; unavailable combinations are rejected'])
    with(output/'result.json').open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();search(a.data,a.output)
