"""Frozen full-game test and round robin with independent slot execution."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from scripts.attack.evaluation_execution import (LEGACY, execution_for, validate_execution,
    environment_kwargs, restrict_placements, check_row_execution, check_duel_execution,configure_battle)
from sim.adapter import grid_to_point
from sim.continuous_push import ContinuousPushEnv
from sim.defense_assets import load_defense_assets
from sim.env import DECK_26,ACTIONS,NUM_PLANES,ClashEnv
from sim.eval_defense_duel import policy_observation,mechanical_mask,outcome,JOINT_MODE
from sim.match import Match
from sim.runner import resolve_deck
from sim.train_ppo import build_network,checkpoint_architecture,load_policy_state


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_parts(protocol_path,protocol,weights,output):
    """Reuse only pinned complete parts from an interrupted identical schedule."""
    weights=Path(weights);output=Path(output);manifest_path=weights/'resume_manifest.json'
    if not manifest_path.exists():return {}
    manifest=json.loads(manifest_path.read_text())
    if manifest['protocol_sha256']!=sha(protocol_path):raise ValueError('Resume protocol differs')
    models={m['id']:m for m in protocol['models']}
    allowed={n+'_test.json':('test',n)for n in models}
    allowed.update({a+'__'+b+'_duels.json':('duel',(a,b))for a,b in itertools.combinations(models,2)})
    reports={}
    for name,digest in manifest['files'].items():
        if name not in allowed:raise ValueError('Resume part outside frozen schedule')
        source=weights/('resume__'+name)
        if source.is_symlink()or sha(source)!=digest:raise ValueError('Resume part changed')
        report=json.loads(source.read_text());kind,identity=allowed[name];rows=report['rows']
        if kind=='test':
            expected={(f'{t}::{f}/heldout_random',s)for t in('attack','defense')for f in('hog','giant','air','golem','swarm','bridge')for s in protocol['test_seeds']}
            if report['model']!=models[identity]or len(rows)!=len(expected)or {(r['scenario_id'],r['seed'])for r in rows}!=expected:
                raise ValueError('Resume test contexts differ')
        else:
            expected={(s,seat)for s in protocol['duel_seeds']for seat in(1,-1)}
            if report['pair']!=list(identity)or len(rows)!=len(expected)or {(r['seed'],r['candidate_side'])for r in rows}!=expected:
                raise ValueError('Resume duel contexts differ')
        if 'execution' in protocol:
            for row in rows:
                check_row_execution(row,execution_for(protocol))
                if kind=='duel':check_duel_execution(row,execution_for(protocol))
        counts=Counter(r['outcome']for r in rows)
        if set(counts)-{'win','loss','draw'}or dict(counts)!=report['counts']:raise ValueError('Resume outcomes differ')
        reports[name]=report
    for name in reports:shutil.copyfile(weights/('resume__'+name),output/name)
    return reports


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


def duel(a,b,device,seed,seat,cards,spells,*,on_complete=None,execution=None,on_command=None):
    execution=validate_execution(LEGACY if execution is None else execution)
    m=Match(cards=cards,spells=spells,decks=(list(DECK_26),list(DECK_26)),seed=seed,
            geometry=execution['geometry'])
    configure_battle(m.battle,execution)
    networks={seat:a[0],-seat:b[0]};modes={seat:a[1],-seat:b[1]}
    pending={1:[],-1:[]};ready={1:{},-1:{}};rejected=Counter();decision=0
    order=(1,-1)if seed%2 else(-1,1)
    command_delays=[];cannon_outside=0
    while not m.finished and m.elapsed_ms<execution['max_ms']:
        if m.elapsed_ms>=decision:
            for side in order:
                mask=mechanical_mask(m,cards,spells,side)
                reserved=sum(cards[c[1]].cost*1000 for c in pending[side])
                busy={c[2]for c in pending[side]};available=m.players[side].elixir-reserved
                for slot,card in enumerate(m.players[side].hand):
                    if slot in busy or m.elapsed_ms<ready[side].get(slot,0)or cards[card].cost*1000>available:
                        mask[1+slot*576:1+(slot+1)*576]=False
                mask=restrict_placements(mask,m.players[side].hand,execution)
                action=choose(networks[side],m,side,modes[side],mask,device,reserved)
                if action:
                    slot,x,y=ClashEnv.decode(action)
                    pending[side].append((m.elapsed_ms+execution['action_delay_ms'],m.players[side].hand[slot],slot,x,y))
            decision+=execution['decision_interval_ms']
        for side in order:
            for command in list(pending[side]):
                if command[0]>m.elapsed_ms:continue
                _,card,slot,x,y=command
                accepted=m.play_card(side,card,grid_to_point(x,y,side));rejected[side]+=not accepted
                requested=command[0]-execution['action_delay_ms']
                command_delays.append(m.elapsed_ms-requested)
                if accepted and card=='cannon' and not (6<=x<=11 and 18<=y<=23):cannon_outside+=1
                if on_command is not None:on_command(dict(side=side,card=card,slot=slot,x=x,y=y,
                    requested_ms=requested,executed_ms=m.elapsed_ms,accepted=accepted))
                ready[side][slot]=m.elapsed_ms+execution['action_cooldown_ms'];pending[side].remove(command)
        m.step(execution['physics_tick_ms'])
    counts={str(s):dict(Counter(e['card']for e in m.play_log if e['side']==s))for s in(1,-1)}
    result=dict(seed=seed,candidate_side=seat,outcome=outcome(m.result,seat,m.finished),elapsed_ms=m.elapsed_ms,
        normal_range_mode=m.battle.normal_range_mode,target_lock_mode=m.battle.target_lock_mode,
        accepted_cards=counts,rejected_commands=dict(rejected),execution=execution,cannon_outside_center=cannon_outside,
        executed_commands=len(command_delays),actual_command_delay_min_ms=min(command_delays)if command_delays else None,
        actual_command_delay_max_ms=max(command_delays)if command_delays else None,
        elixir_overflow={str(s):m.players[s].natural_elixir_overflow_milli/1000 for s in(1,-1)},
        quality_flags=['simulator_mechanics_not_fully_validated','no_live_validation'])
    if on_complete is not None:result['public_metrics']=on_complete(m)
    return result


def evaluate(protocol,weights,output):
    protocol=Path(protocol);p=json.loads(protocol.read_text());output=Path(output);output.mkdir(exist_ok=False,parents=True)
    if not torch.cuda.is_available()or torch.cuda.device_count()<2:raise RuntimeError('Two CUDA devices required')
    execution=execution_for(p)
    torch.set_num_threads(1);models=p['models'];pairs=list(itertools.combinations(models,2))
    resumed=restore_parts(protocol,p,weights,output);jobs=[]
    for entry in models:
        name=entry['id']+'_test.json'
        if name in resumed:
            row=resumed[name];print(json.dumps(dict(event='POOL_TEST_COMPLETE',model=entry['id'],games=len(row['rows']),wins=row['counts'].get('win',0),reused=True)),flush=True)
        else:jobs.append(('test',entry,12*len(p['test_seeds'])))
    for a,b in pairs:
        name=a['id']+'__'+b['id']+'_duels.json'
        if name in resumed:print(json.dumps(dict(event='POOL_PAIR_COMPLETE',pair=[a['id'],b['id']],games=len(resumed[name]['rows']),reused=True)),flush=True)
        else:jobs.append(('duel',(a,b),2*len(p['duel_seeds'])))
    schedules=[[],[]];loads=[0,0]
    for job in sorted(jobs,key=lambda r:-r[2]):
        index=min(range(2),key=lambda i:loads[i]);schedules[index].append(job);loads[index]+=job[2]
    def worker(index):
        device=f'cuda:{index}';nets={e['id']:load(e,weights,device)for e in models}
        cards,spells,*_=load_defense_assets();cards.update(resolve_deck(cards,list(DECK_26)))
        for entry in [job[1]for job in schedules[index]if job[0]=='test']:
            rows=[]
            for task in ('attack','defense'):
                for family in ('hog','giant','air','golem','swarm','bridge'):
                    for seed in p['test_seeds']:
                        env=ContinuousPushEnv(split='test',**environment_kwargs(execution))
                        env.reset(seed=seed,scenario_id=f'{task}::{family}/heldout_random')
                        for _ in range(execution['max_ms']//execution['decision_interval_ms']+2):
                            reserved=sum(env.cards[c[1]].cost*1000 for c in env.pending)
                            action=choose(nets[entry['id']],env.match,1,entry['mode'],env._mask(),device,reserved)
                            _,_,done,truncated,info=env.step(action)
                            if done or truncated:break
                        else:raise RuntimeError('Nonterminal heldout game')
                        rows.append(dict(info['metrics'],seed=seed,execution=execution))
            counts=Counter(r['outcome']for r in rows)
            if counts['unknown']:raise ValueError('Unknown heldout outcome')
            result=dict(model=entry,counts=dict(counts),games=len(rows),
                win_rate=counts['win']/len(rows),
                mean_card_plays={card:float(np.mean([r['accepted_card_counts'].get(card,0)for r in rows]))for card in DECK_26},
                mean_elixir_overflow=float(np.mean([r['elixir_overflow']for r in rows])),rows=rows)
            (output/(entry['id']+'_test.json')).write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(event='POOL_TEST_COMPLETE',model=entry['id'],games=len(rows),wins=counts['win'])),flush=True)
        for a,b in [job[1]for job in schedules[index]if job[0]=='duel']:
            rows=[]
            for seed in p['duel_seeds']:
                for seat in(1,-1):rows.append(duel((nets[a['id']],a['mode']),(nets[b['id']],b['mode']),device,seed,seat,cards,spells,execution=execution))
            (output/(a['id']+'__'+b['id']+'_duels.json')).write_text(json.dumps(dict(pair=[a['id'],b['id']],games=len(rows),counts=dict(Counter(r['outcome']for r in rows)),rows=rows),indent=2)+'\n')
            print(json.dumps(dict(event='POOL_PAIR_COMPLETE',pair=[a['id'],b['id']],games=len(rows))),flush=True)
    print(json.dumps(dict(event='CUDA_READY',gpus=[torch.cuda.get_device_name(i)for i in(0,1)])),flush=True)
    with ThreadPoolExecutor(2)as pool:
        futures=[pool.submit(worker,i)for i in(0,1)]
        for f in futures:f.result()
    files={x.name:sha(x)for x in output.glob('*.json')}
    result=dict(schema='continuous-frozen-pool-evaluation-v1',status='complete',protocol_sha256=sha(protocol),
        execution=execution,models=models,test_games=len(models)*12*len(p['test_seeds']),duel_games=len(pairs)*2*len(p['duel_seeds']),
        files=files,quality_accepted=False,test_used_for_selection=False,live_checkpoint_changed=False)
    if resumed:
        result.update(resumed_parts_sha256={n:sha(output/n)for n in resumed},
            resumed_games=sum(len(r['rows'])for r in resumed.values()),computed_games=sum(job[2]for job in jobs))
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--weights',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();evaluate(a.protocol,a.weights,a.output)
