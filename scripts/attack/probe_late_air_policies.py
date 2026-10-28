"""Closed-loop TRAIN fixture diagnostic; no teacher actions or live promotion."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from scripts.attack.audit_air_label_latency import opening
from scripts.attack.evaluate_continuous_pool import load,sha
from sim.adapter import grid_to_point
from sim.continuous_push import ContinuousPushEnv
from sim.defense_assets import load_defense_assets
from sim.env import ClashEnv,DECK_26
from sim.runner import resolve_deck


def episode(net,record,cards,spells):
    match,hp=opening(record,cards,spells)
    env=object.__new__(ContinuousPushEnv)
    env.match,env.cards,env.spells=match,cards,spells
    env.profile={'cannon_center_only':1}
    env.pending,env.slot_ready,env.reflected=[],{},False
    start=match.elapsed_ms;actions=[];decisions=[];rejections=0;rejected_commands=[]
    while not match.finished and match.elapsed_ms<start+60000:
        now=match.elapsed_ms
        alive=sum(e.alive and e.side==-1 and not e.is_tower for e in match.battle.entities.values())
        if alive==0:break
        if (now-start)%250==0:
            mask=env._mask();obs=env._observe()
            with torch.inference_mode():
                logits,_=net(torch.from_numpy(obs['planes'])[None],torch.from_numpy(obs['scalars'])[None])
                support=torch.from_numpy(mask)[None]
                if not torch.isfinite(logits[support]).all():raise ValueError('Nonfinite policy logits')
                action=int(logits.masked_fill(~support,-torch.inf).argmax(-1).item())
            decisions.append(dict(at_ms=now,action=action,available_elixir=float(obs['scalars'][0])*10))
            if action:
                slot,x,y=ClashEnv.decode(action)
                if slot<0:raise ValueError('Unexpected ability action')
                env.pending.append((now+1350,match.players[1].hand[slot],slot,x,y))
        for event in list(env.pending):
            due,card,slot,x,y=event
            if due>now:continue
            if match.play_card(1,card,grid_to_point(x,y,1)):
                actions.append(dict(at_ms=now,card=card,cell=[x,y],cost=cards[card].cost))
            else:
                rejections+=1
                rejected_commands.append(dict(card=card,cell=[x,y],at_ms=now,hand=list(match.players[1].hand),elixir=match.players[1].elixir,can_play=match.can_play(1,card),has_spell_spec=card in spells))
            env.slot_ready[slot]=now+2100;env.pending.remove(event)
        match.step(50)
    damage={k:v-match.towers[1][k].hitpoints for k,v in hp.items()}
    remaining=sum(e.alive and e.side==-1 and not e.is_tower for e in match.battle.entities.values())
    return dict(opening=record,actions=actions,decisions=decisions,damage=damage,total_damage=sum(damage.values()),
        remaining_enemies=remaining,clear_without_damage=remaining==0 and sum(damage.values())==0,
        elixir_spent=sum(a['cost']for a in actions),rejections=rejections,rejected_commands=rejected_commands,
        wait_at_full_elixir=sum(d['action']==0 and d['available_elixir']>=9.99 for d in decisions),
        elapsed_ms=match.elapsed_ms-start,pending_at_stop=len(env.pending))


def run(config,output):
    config,output=Path(config),Path(output);protocol=json.loads(config.read_text())
    source=Path(protocol['search']);contexts=json.loads(source.read_text())['contexts']
    if sha(source)!=protocol['search_sha256']:raise ValueError('Fixture source changed')
    output.mkdir(parents=True,exist_ok=False)
    protocol.update(source_sha256=sha(__file__),match_source_sha256=sha('sim/match.py'),environment_source_sha256=sha('sim/continuous_push.py'),split='train',quality_accepted=False,live_games_started=False,
        command_delay_ms=1350,decision_interval_ms=250,no_enemy_reinforcement=True)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    torch.set_num_threads(1);cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
    summaries={};rows=[]
    with(output/'episodes.jsonl').open('x')as log:
        for model in protocol['models']:
            net=load(dict(file=str(Path(model['path']).absolute()),sha256=model['sha256'],scalars=309,mode='joint_enemy_presence_v1'),Path('.'),'cpu')
            own=[]
            for c in contexts:
                row=episode(net,c['opening'],cards,spells);row.update(model=model['id'],context=c['context'])
                own.append(row);rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
            counts=Counter(a['card']for r in own for a in r['actions'])
            summaries[model['id']]=dict(contexts=len(own),zero_damage=sum(r['clear_without_damage']for r in own),
                mean_damage=float(np.mean([r['total_damage']for r in own])),mean_elixir=float(np.mean([r['elixir_spent']for r in own])),
                mean_card_plays={k:counts[k]/len(own)for k in DECK_26},
                wait_at_full_elixir=sum(r['wait_at_full_elixir']for r in own),rejections=sum(r['rejections']for r in own))
            print(json.dumps(dict(model=model['id'],summary=summaries[model['id']])),flush=True)
    result=dict(protocol=protocol,summaries=summaries,rows=rows)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.config,a.output)
