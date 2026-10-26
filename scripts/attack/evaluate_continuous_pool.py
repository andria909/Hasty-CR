"""Frozen full-game test and round robin with independent slot execution."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from sim.adapter import grid_to_point
from sim.continuous_push import ContinuousPushEnv
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26,ACTIONS,NUM_PLANES,ClashEnv
from sim.eval_defense_duel import policy_observation,mechanical_mask,outcome,JOINT_MODE
from sim.match import Match
from sim.runner import resolve_deck
from sim.train_ppo import build_network,checkpoint_architecture,load_policy_state


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(entry,weights,device):
    path=Path(weights)/entry['file']
    if sha(path)!=entry['sha256']:raise ValueError('Evaluation weight changed')
    blob=torch.load(path,map_location='cpu',weights_only=False)
    count=blob['state_dict']['scalar.0.weight'].shape[1]
    if count!=entry['scalars']:raise ValueError('Declared input count differs')
    net=build_network(NUM_PLANES,count,ACTIONS,checkpoint_architecture(blob)).to(device)
    load_policy_state(net,blob['state_dict'])
    return net.eval().requires_grad_(False)


def choose(network,match,side,mode,mask,device,reserved_milli=0):
    if not mask[1:].any():return 0
    obs=policy_observation(match,side,mode)
    obs['scalars']=obs['scalars'].copy()
    obs['scalars'][0]=max(0,obs['scalars'][0]-reserved_milli/10000)
    with torch.inference_mode():
        logits,_=network(torch.as_tensor(obs['planes'],device=device)[None],
                         torch.as_tensor(obs['scalars'],device=device)[None])
        support=torch.as_tensor(mask,device=device)
        if not torch.isfinite(logits[0,support]).all():raise ValueError('Nonfinite legal logits')
        return int(logits[0].masked_fill(~support,-torch.inf).argmax())


def duel(a,b,device,seed,seat,cards,spells):
    m=Match(cards=cards,spells=spells,decks=(list(DECK_26),list(DECK_26)),seed=seed,
            geometry='native_anchors_candidate_v1')
    networks={seat:a[0],-seat:b[0]};modes={seat:a[1],-seat:b[1]}
    pending={1:[],-1:[]};ready={1:{},-1:{}};rejected=Counter();decision=0
    order=(1,-1)if seed%2 else(-1,1)
    while not m.finished and m.elapsed_ms<310000:
        if m.elapsed_ms>=decision:
            for side in order:
                mask=mechanical_mask(m,cards,spells,side)
                reserved=sum(cards[c[1]].cost*1000 for c in pending[side])
                busy={c[2]for c in pending[side]};available=m.players[side].elixir-reserved
                for slot,card in enumerate(m.players[side].hand):
                    if slot in busy or m.elapsed_ms<ready[side].get(slot,0)or cards[card].cost*1000>available:
                        mask[1+slot*576:1+(slot+1)*576]=False
                action=choose(networks[side],m,side,modes[side],mask,device,reserved)
                if action:
                    slot,x,y=ClashEnv.decode(action)
                    pending[side].append((m.elapsed_ms+225,m.players[side].hand[slot],slot,x,y))
            decision+=250
        for side in order:
            for command in list(pending[side]):
                if command[0]>m.elapsed_ms:continue
                _,card,slot,x,y=command
                accepted=m.play_card(side,card,grid_to_point(x,y,side));rejected[side]+=not accepted
                ready[side][slot]=m.elapsed_ms+2100;pending[side].remove(command)
        m.step(50)
    counts={str(s):dict(Counter(e['card']for e in m.play_log if e['side']==s))for s in(1,-1)}
    return dict(seed=seed,candidate_side=seat,outcome=outcome(m.result,seat,m.finished),elapsed_ms=m.elapsed_ms,
        accepted_cards=counts,rejected_commands=dict(rejected),
        elixir_overflow={str(s):m.players[s].natural_elixir_overflow_milli/1000 for s in(1,-1)},
        quality_flags=['simulator_mechanics_not_fully_validated','no_live_validation'])


def evaluate(protocol,weights,output):
    protocol=Path(protocol);p=json.loads(protocol.read_text());output=Path(output);output.mkdir(exist_ok=False,parents=True)
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices required')
    torch.set_num_threads(1);models=p['models'];pairs=list(itertools.combinations(models,2))
    def worker(index):
        device=f'cuda:{index}';nets={e['id']:load(e,weights,device)for e in models}
        cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
        for entry in models[index::2]:
            rows=[]
            for task in ('attack','defense'):
                for family in ('hog','giant','air','golem','swarm','bridge'):
                    for seed in p['test_seeds']:
                        env=ContinuousPushEnv(split='test',action_delay_ms=225,max_ms=310000)
                        env.reset(seed=seed,scenario_id=f'{task}::{family}/heldout_random')
                        for _ in range(1242):
                            reserved=sum(env.cards[c[1]].cost*1000 for c in env.pending)
                            action=choose(nets[entry['id']],env.match,1,entry['mode'],env._mask(),device,reserved)
                            _,_,done,truncated,info=env.step(action)
                            if done or truncated:break
                        else:raise RuntimeError('Nonterminal heldout game')
                        rows.append(dict(info['metrics'],seed=seed))
            counts=Counter(r['outcome']for r in rows)
            if counts['unknown']:raise ValueError('Unknown heldout outcome')
            result=dict(model=entry,counts=dict(counts),games=len(rows),
                win_rate=counts['win']/len(rows),
                mean_card_plays={card:float(np.mean([r['accepted_card_counts'].get(card,0)for r in rows]))for card in DECK_26},
                mean_elixir_overflow=float(np.mean([r['elixir_overflow']for r in rows])),rows=rows)
            (output/(entry['id']+'_test.json')).write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(event='POOL_TEST_COMPLETE',model=entry['id'],games=len(rows),wins=counts['win'])),flush=True)
        for a,b in pairs[index::2]:
            rows=[]
            for seed in p['duel_seeds']:
                for seat in(1,-1):rows.append(duel((nets[a['id']],a['mode']),(nets[b['id']],b['mode']),device,seed,seat,cards,spells))
            (output/(a['id']+'__'+b['id']+'_duels.json')).write_text(json.dumps(dict(pair=[a['id'],b['id']],games=len(rows),counts=dict(Counter(r['outcome']for r in rows)),rows=rows),indent=2)+'\n')
            print(json.dumps(dict(event='POOL_PAIR_COMPLETE',pair=[a['id'],b['id']],games=len(rows))),flush=True)
    print(json.dumps(dict(event='CUDA_READY',gpus=[torch.cuda.get_device_name(i)for i in(0,1)])),flush=True)
    with ThreadPoolExecutor(2)as pool:
        futures=[pool.submit(worker,i)for i in(0,1)]
        for f in futures:f.result()
    files={x.name:sha(x)for x in output.glob('*.json')}
    result=dict(schema='continuous-frozen-pool-evaluation-v1',status='complete',protocol_sha256=sha(protocol),
        models=models,test_games=len(models)*12*len(p['test_seeds']),duel_games=len(pairs)*2*len(p['duel_seeds']),
        files=files,quality_accepted=False,test_used_for_selection=False,live_checkpoint_changed=False)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--weights',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();evaluate(a.protocol,a.weights,a.output)
