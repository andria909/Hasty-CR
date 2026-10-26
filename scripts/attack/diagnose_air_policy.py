"""Bounded TRAIN-only diagnosis of anti-air availability and policy preference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
from sim.continuous_push import ContinuousPushEnv, CONTRACT
from sim.env import ClashEnv
from sim.train_attack_ppo import initialize_attack_policy


def diagnose(weight,seed,intervention='none',trace_musketeer=False):
    if intervention not in ('none','reserve','reserve_and_deploy'):raise ValueError('Unknown TRAIN intervention')
    torch.set_num_threads(1)
    blob=torch.load(weight,map_location='cpu',weights_only=False)
    net,_,_=initialize_attack_policy(blob,device='cpu',num_scalars=309,observation_contract=CONTRACT)
    net.eval();env=ContinuousPushEnv(seed=seed,split='train')
    obs,info=env.reset(seed=seed,scenario_id='defense::air/push')
    counts=Counter();mass=Counter();examples=[];tracked={};traces={}
    while not env.done:
        mask=torch.as_tensor(info['action_mask'])
        with torch.inference_mode():
            logits,_=net(torch.as_tensor(obs['planes'])[None],torch.as_tensor(obs['scalars'])[None])
            legal=logits[0].masked_fill(~mask,-torch.inf);probs=legal.softmax(0);action=int(legal.argmax())
        air=[e for e in env.match.battle.entities.values()if e.alive and e.side==-1 and e.flying and not e.is_tower]
        if air and intervention!='none' and 'musketeer'in env.match.players[1].hand:
            slot_m=env.match.players[1].hand.index('musketeer')
            start=1+slot_m*576;part=legal[start:start+576]
            original=action
            if intervention=='reserve_and_deploy' and torch.isfinite(part).any():
                action=start+int(part.argmax())
            elif env.match.players[1].elixir<4000:
                action=0
            if action!=original:counts['diagnostic_action_overrides']+=1
        if air:
            counts['decisions_with_enemy_air']+=1
            if not mask[1:].any():counts['air_forced_idle']+=1
            if 'musketeer'in env.match.players[1].hand:
                counts['air_musketeer_in_hand']+=1
                if env.match.players[1].elixir<4000:counts['air_musketeer_in_hand_bank_below_four']+=1
            if env.match.players[1].elixir>=4000:counts['air_bank_at_least_four']+=1
            for slot,card in enumerate(env.match.players[1].hand):
                support=mask[1+slot*576:1+(slot+1)*576]
                if support.any():
                    counts['air_legal_'+card]+=1;mass[card]+=float(probs[1+slot*576:1+(slot+1)*576].sum())
            if action:
                slot,x,y=ClashEnv.decode(action);card=env.match.players[1].hand[slot]
                counts['air_selected_'+card]+=1
                if card!='musketeer' and 'musketeer'in env.match.players[1].hand:
                    slot_m=env.match.players[1].hand.index('musketeer');part=legal[1+slot_m*576:1+(slot_m+1)*576]
                    if torch.isfinite(part).any()and len(examples)<12:
                        examples.append(dict(at_ms=env.match.elapsed_ms,selected=card,selected_logit=float(legal[action]),
                            best_musketeer_logit=float(part.max()),elixir=env.match.players[1].elixir/1000))
            else:counts['air_selected_idle']+=1
        obs,_,_,_,info=env.step(action)
        if trace_musketeer:
            now=env.match.elapsed_ms
            for entity in env.match.battle.entities.values():
                if entity.side==1 and entity.name=='musketeer' and entity.uid not in tracked:
                    tracked[entity.uid]=entity
                    traces[entity.uid]=dict(uid=entity.uid,first_observed_ms=now,spawn_position_mt=[entity.pos.x,entity.pos.y],
                        first_damage_observed_ms=None,death_observed_ms=None,target_samples={})
            for uid,entity in tracked.items():
                trace=traces[uid]
                if entity.damage_dealt and trace['first_damage_observed_ms']is None:trace['first_damage_observed_ms']=now
                if not entity.alive and trace['death_observed_ms']is None:trace['death_observed_ms']=now
                if entity.alive:
                    target=env.match.battle.entities.get(entity.target_uid)
                    label=('air'if target.flying else 'ground')if target else 'none'
                    trace['target_samples'][label]=trace['target_samples'].get(label,0)+1
            for uid,entity in tracked.items():traces[uid]['damage_dealt_all_targets']=entity.damage_dealt
    return dict(schema='air-policy-train-diagnostic-v1',weight_sha256=hashlib.sha256(Path(weight).read_bytes()).hexdigest(),
        seed=seed,intervention=intervention,scenario='defense::air/push',counts=dict(counts),policy_probability_mass_sums=dict(mass),examples=examples,
        musketeer_trace=list(traces.values()),musketeer_commands=[a for a in env.actions if a['card']=='musketeer']if trace_musketeer else [],
        final_metrics=info['metrics'],scope='One greedy TRAIN game; diagnostic only, no checkpoint selection or live claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weight',required=True);p.add_argument('--seed',type=int,default=781)
    p.add_argument('--output',required=True);p.add_argument('--compare-weight')
    p.add_argument('--trace-musketeer',action='store_true')
    p.add_argument('--intervention',choices=('none','reserve','reserve_and_deploy'),default='none')
    p.add_argument('--seeds',type=int,nargs='+',default=[781,782,783]);a=p.parse_args()
    if a.compare_weight:
        rows=[]
        for name,path in [('parent',a.weight),('strong',a.compare_weight)]:
            for seed in a.seeds:
                row=diagnose(path,seed,a.intervention,a.trace_musketeer);row['model']=name;rows.append(row)
        r=dict(schema='paired-air-policy-train-diagnostic-v1',rows=rows,
            scope='Paired TRAIN seeds, diagnostic not model selection',live_games_started=False)
    else:r=diagnose(a.weight,a.seed,a.intervention,a.trace_musketeer)
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r.get('counts',dict(games=len(r.get('rows',[]))))))
