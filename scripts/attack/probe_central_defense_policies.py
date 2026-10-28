"""Fixed TRAIN diagnostic suite for learned defense; no teacher actions injected."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import random
import torch

from scripts.attack.audit_cannon_tower_states import ENEMY,ready_hand
from scripts.attack.evaluate_continuous_pool import load,sha
from scripts.attack.probe_late_air_policies import episode
from scripts.attack.sequence_execution import EXECUTION
from scripts.attack.sequence_physics import physics_pins
from sim.adapter import grid_to_point
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26
from sim.match import Match
from sim.runner import resolve_deck


def contexts():
    return [dict(seed=seed,own_mask=mask,lane=lane,threat=threat,initial_own_elixir=bank,decision_ms=1500)
            for seed,mask,lane,threat,bank in product((814201,814219),range(4),('left','right'),
                                                    ('hog_rider','balloon'),(6,10))]


def opening(record,cards,spells):
    match=Match(cards=cards,spells=spells,decks=(list(DECK_26),list(ENEMY)),seed=record['seed'],
                geometry='native_anchors_candidate_v1')
    for bit,key in enumerate(('left','right')):
        if not record['own_mask']&(1<<bit):match.towers[1][key].hitpoints=0
    match._wake_kings()
    order=list(DECK_26);random.Random(record['seed']).shuffle(order)
    own=match.players[1];own.hand=order[:4];own.queue=order[4:];own.elixir=record['initial_own_elixir']*1000
    ready_hand(match.players[-1],record['threat'])
    x=0 if record['threat']=='balloon'else 3
    if record['lane']=='right':x=17-x
    assert match.play_card(-1,record['threat'],grid_to_point(x,15,1))
    hp={k:t.hitpoints for k,t in match.towers[1].items()}
    while match.elapsed_ms<record['decision_ms']:match.step(50)
    return match,hp


def summarize(rows):
    counts=Counter(a['card']for row in rows for a in row['actions'])
    return dict(contexts=len(rows),clear_without_damage=sum(r['clear_without_damage']for r in rows),
        mean_damage=sum(r['total_damage']for r in rows)/len(rows),
        mean_paid_elixir=sum(r['elixir_spent']for r in rows)/len(rows),
        mean_card_plays={c:counts[c]/len(rows)for c in DECK_26},
        rejected_commands=sum(r['rejections']for r in rows),
        cannon_outside_center=sum(a['card']=='cannon'and not(6<=a['cell'][0]<=11 and 18<=a['cell'][1]<=23)
                                 for r in rows for a in r['actions']))


def run(config,output):
    config=Path(config);spec=json.loads(config.read_text());output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    suite=contexts();pins=physics_pins()
    protocol=dict(schema='central-defense-policy-probe-v1',models=spec['models'],contexts=suite,
        execution=EXECUTION,physics_pins=pins,split='TRAIN development',horizon_ms=60000,
        actor_stops_after_last_enemy_body=True,world_continues_for_inflight_effects=True,
        sources={p:sha(p)for p in [__file__,'scripts/attack/probe_late_air_policies.py']},
        config_sha256=sha(config),quality_accepted=False,live_games_started=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    summaries={};completed=0
    with(output/'episodes.jsonl').open('x')as f:
        for model in spec['models']:
            net=load(dict(file=str(Path(model['path']).absolute()),sha256=model['sha256'],scalars=309),Path('/'),'cpu')
            rows=[]
            for record in suite:
                match,hp=opening(record,cards,spells)
                hand=list(match.players[1].hand)
                row=episode(net,record,cards,spells,initial=(match,hp));row.update(model=model['id'],initial_hand=hand)
                rows.append(row);f.write(json.dumps(row)+'\n');f.flush();completed+=1
                if completed%16==0:print(json.dumps(dict(completed=completed,total=len(suite)*len(spec['models']))),flush=True)
            summaries[model['id']]=dict(overall=summarize(rows),
                by_threat={t:summarize([r for r in rows if r['opening']['threat']==t])for t in ('hog_rider','balloon')},
                by_own_mask={str(mask):summarize([r for r in rows if r['opening']['own_mask']==mask])for mask in range(4)})
    if pins!=physics_pins():raise ValueError('Physics changed during probe')
    result=dict(protocol=protocol,summaries=summaries,episodes_sha256=sha(output/'episodes.jsonl'),
        quality_accepted=False,limitations=['Two shuffled-hand seeds; correlated development fixtures, not independent win rate.',
            'No enemy reinforcements; model acts only while enemy bodies exist, then pending effects resolve.',
            'Models evaluated with current central mask/timing regardless of their historical training configuration.'])
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.config,a.output)
